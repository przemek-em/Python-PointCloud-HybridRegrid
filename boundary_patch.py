"""Complete bounded holes in local surface charts using their whole boundary.

Charts follow three principal directions. A measured tangent graph separates
sheets before projection. Quadratic trend plus a harmonic residual supplies a
continuous patch, rather than asking each new point for a locally planar fit.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from math import prod

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from cloud import Cloud, _find, check_cancel
from cpu_workers import ordered_batches, resolve_workers
from hybrid import _subset
from surface_fill import _support, _normals


@dataclass
class PatchResult:
    cloud: Cloud | None
    stats: dict = field(default_factory=dict)


def _design(uv):
    u, v = uv.T
    return np.column_stack((np.ones(len(u)), u, v, u*u, u*v, v*v))


def _components(xyz, normals, scale, cancel):
    """Connect short tangent edges; parallel sheets have no normal-directed edge."""
    tree = cKDTree(xyz)
    rows, columns = [], []
    for start in range(0, len(xyz), 8192):
        check_cancel(cancel)
        p = xyz[start:start+8192]
        distance, near = tree.query(p, k=min(16, len(xyz)), distance_upper_bound=3.5*scale, workers=1)
        valid = np.isfinite(distance) & (distance > 1e-10*scale)
        near = np.minimum(near, len(xyz)-1)
        delta = xyz[near]-p[:, None, :]
        n = normals[start:start+len(p)]
        depth = np.maximum(np.abs(np.einsum("nki,ni->nk", delta, n)), np.abs(np.einsum("nki,nki->nk", delta, normals[near])))
        valid &= depth <= .45*distance+.35*scale
        a, b = np.nonzero(valid)
        rows.append(a+start)
        columns.append(near[a, b])
    graph = sparse.coo_matrix((np.ones(sum(map(len, rows)), dtype=np.uint8), (np.concatenate(rows), np.concatenate(columns))), shape=(len(xyz), len(xyz))).tocsr()
    return connected_components(graph, directed=False)[1]


def _chart_jobs(xyz, normals, axes, scale, grid, cancel, stats):
    jobs = []
    for axis in range(3):
        check_cancel(cancel)
        selection = np.flatnonzero(np.abs(normals @ axes[:, axis]) >= .15)
        if len(selection) < 16:
            continue
        labels = _components(xyz[selection], normals[selection], scale, cancel)
        order = np.argsort(labels, kind="stable")
        count = np.bincount(labels)
        offset = np.r_[0, np.cumsum(count)]
        for label in np.flatnonzero(count >= 16):
            indices = selection[order[offset[label]:offset[label+1]]]
            basis = axes[:, [i for i in range(3) if i != axis]+[axis]]
            center = xyz[indices].mean(axis=0)
            local = (xyz[indices]-center) @ basis
            uv_origin = np.floor(local[:, :2].min(axis=0)/grid)*grid-grid
            keys = np.floor((local[:, :2]-uv_origin)/grid).astype(np.int64)
            shape = tuple(map(int,keys.max(axis=0)+2))
            if prod(shape) > 2_000_000:
                stats["oversized_charts"] += 1
                continue
            flat = np.ravel_multi_index(keys.T, shape)
            counts = np.bincount(flat, minlength=prod(shape)).reshape(shape)
            measured = counts > 0
            height = np.bincount(flat, weights=local[:, 2], minlength=prod(shape)).reshape(shape)/np.maximum(counts, 1)
            mean_uv = np.stack([np.bincount(flat, weights=local[:, i], minlength=prod(shape)).reshape(shape)/np.maximum(counts,1) for i in range(2)],axis=-1)
            low = np.full(prod(shape), np.inf)
            high = np.full(prod(shape), -np.inf)
            np.minimum.at(low, flat, local[:, 2])
            np.maximum.at(high, flat, local[:, 2])
            ambiguous = ((high-low) > 2.5*grid).reshape(shape) & measured
            # Seal one-cell sampling cracks for detection, then restore the rim.
            # Open regions reaching the chart exterior are never filled.
            covered = ndimage.binary_dilation(measured)
            core = ndimage.binary_fill_holes(covered) & ~covered
            holes = (ndimage.binary_dilation(core) | (ndimage.binary_fill_holes(measured) & ~measured)) & ~measured
            regions, _ = ndimage.label(holes)
            stats["charts"] += 1
            own_tree = None
            for hole_index, bounds in enumerate(ndimage.find_objects(regions), 1):
                check_cancel(cancel)
                if bounds is None:
                    continue
                hole = regions[bounds] == hole_index
                # A modest measured context ring supplies the boundary trend.
                padding = max(3, min(12, int(np.ceil(max(hole.shape)*.3))))
                expanded = tuple(slice(max(0, b.start-padding), min(s, b.stop+padding)) for b,s in zip(bounds,shape))
                domain = regions[expanded] == hole_index
                if domain.sum() < 4:
                    stats["small_sampling_gaps"] += 1
                    continue
                ring = ndimage.binary_dilation(domain, iterations=padding) & measured[expanded]
                mixed = ring & ambiguous[expanded]
                if ring.sum() < 8 or mixed.sum() > .15*ring.sum():
                    stats["ambiguous_holes"] += 1
                    continue
                stats["ignored_ambiguous_boundary_cells"] += int(mixed.sum())
                ring &= ~mixed
                if own_tree is None:
                    own_tree = cKDTree(xyz[indices])
                jobs.append((indices, basis, center, uv_origin, grid, expanded, domain, ring, height[expanded].copy(), mean_uv[expanded].copy(), own_tree))
    return jobs


def _make_patch(job, support, spacing, reach, minimum_distance, source_tree, cancel, max_candidates):
    check_cancel(cancel)
    indices, basis, center, uv_origin, grid, bounds, domain, ring, height, mean_uv, own_tree = job
    offset = np.array([s.start for s in bounds])
    cells = np.column_stack(np.nonzero(domain))
    # Reach is measured against the same sheet, and limits an entire patch's
    # depth into its hole. A larger hole is left intact rather than half filled.
    nearest = ndimage.distance_transform_edt(~ring, return_distances=False, return_indices=True)
    uv0 = uv_origin+(offset+.5)*grid
    uv = mean_uv[ring]
    uv_center = uv0+cells.mean(axis=0)*grid
    size = max(grid, float(np.ptp(uv, axis=0).max()))
    design = _design((uv-uv_center)/size)
    values = height[ring]
    coefficient = np.linalg.lstsq(design, values, rcond=1e-10)[0]
    for _ in range(3):
        residual = values-design @ coefficient
        deviation = max(.1*grid, 1.4826*float(np.median(np.abs(residual-np.median(residual)))))
        weights = np.minimum(1, 2.5*deviation/np.maximum(np.abs(residual), 1e-12))
        coefficient = np.linalg.lstsq(design*weights[:, None], values*weights, rcond=1e-10)[0]
    shape = domain.shape
    residual = np.zeros(shape)
    residual[ring] = values-design @ coefficient
    fixed = residual[tuple(nearest)]
    count = len(cells)
    lookup = np.full(shape, -1, dtype=np.int64)
    lookup[domain] = np.arange(count)
    row, column, data = [np.arange(count)], [np.arange(count)], [np.full(count, 4.)]
    rhs = np.zeros(count)
    for delta in [(-1,0),(1,0),(0,-1),(0,1)]:
        other = cells+delta
        inside = np.all((other >= 0) & (other < shape), axis=1)
        other = np.clip(other, 0, np.array(shape)-1)
        target = lookup[tuple(other.T)]
        unknown = inside & (target >= 0)
        row.append(np.flatnonzero(unknown)); column.append(target[unknown]); data.append(np.full(unknown.sum(), -1.))
        known = ~unknown
        rhs[known] += fixed[tuple(other[known].T)]
    matrix = sparse.coo_matrix((np.concatenate(data), (np.concatenate(row), np.concatenate(column))), shape=(count,count)).tocsr()
    solution, info = cg(matrix, rhs, rtol=1e-7, atol=1e-9, maxiter=2000, callback=lambda x: check_cancel(cancel))
    if info != 0:
        return None, {"unconverged": 1}
    field = fixed.copy()
    field[domain] = solution
    # Oversample the chart before selecting one world voxel representative.
    step = min(grid, .5*spacing)
    # Include the measured rim's cells too. A projected cell can contain both a
    # rim measurement and empty space; world-voxel deduplication resolves that
    # overlap without leaving an unfilled seam at the patch boundary.
    sampling_domain = ndimage.binary_dilation(domain)
    sample_cells = np.column_stack(np.nonzero(sampling_domain))
    minimum = uv_origin+(offset+sample_cells.min(axis=0))*grid
    maximum = uv_origin+(offset+sample_cells.max(axis=0)+1)*grid
    dimension_float = np.ceil((maximum-minimum)/step)
    if not np.isfinite(dimension_float).all():
        raise ValueError("Output spacing is too small for this boundary patch.")
    dimension = tuple(map(int,dimension_float))
    if prod(dimension) > max_candidates:
        raise ValueError("A boundary patch is too large at this output spacing. Increase spacing or crop the input.")
    fine = np.stack(np.meshgrid(*(minimum[i]+(np.arange(dimension[i])+.5)*step for i in range(2)), indexing="ij"),axis=-1).reshape(-1,2)
    pixel = np.floor((fine-uv0)/grid+.5).astype(np.int64)
    valid = np.all((pixel >= 0) & (pixel < shape), axis=1)
    pixel = np.clip(pixel, 0, np.array(shape)-1)
    valid &= sampling_domain[tuple(pixel.T)]
    fine = fine[valid]
    coordinates = ((fine-uv0)/grid).T
    depth = _design((fine-uv_center)/size) @ coefficient + ndimage.map_coordinates(field, coordinates, order=1, mode="nearest")
    xyz = np.column_stack((fine,depth)) @ basis.T+center
    distance, closest = own_tree.query(xyz, workers=1)
    if len(distance) and distance.max() > reach+1e-8*grid:
        return None, {"beyond_reach": 1,"tested_candidates":len(fine)}
    measured_distance, _ = source_tree.query(xyz, workers=1)
    keep = (distance <= reach) & (measured_distance >= minimum_distance) & np.isfinite(xyz).all(axis=1)
    xyz, closest = xyz[keep], indices[closest[keep]]
    if not len(xyz):
        return None, {"tested_candidates":len(fine)}
    values = {name: getattr(support,name)[closest] for name in ("rgb", "intensity", "classification") if getattr(support,name) is not None}
    check_cancel(cancel)
    return Cloud(xyz, **values, generated=np.ones(len(xyz),bool), source_indices=np.full(len(xyz),-1,dtype=np.int64), metadata=support.metadata), {"patched_holes":1,"tested_candidates":len(fine)}


def complete(cloud, spacing, reach, native, workers=0, cancel=None, progress=None, minimum_distance=0, max_candidates=4_000_000):
    check_cancel(cancel)
    workers = resolve_workers(workers)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="boundary-patch") if workers>1 else nullcontext(None)
    progress = progress or (lambda f,m:None)
    stats = {"charts":0, "detected_holes":0, "patched_holes":0, "ambiguous_holes":0, "ignored_ambiguous_boundary_cells":0, "oversized_charts":0, "small_sampling_gaps":0, "beyond_reach":0, "unconverged":0,"tested_candidates":0}
    with threadpool_limits(limits=1,user_api="blas"), executor as pool:
        scale = max(native, .5*spacing)
        progress(.01,"Finding measured surface sheets for boundary patches")
        _, _, _, support = _support(cloud,scale)
        stats["support_points"] = len(support.xyz)
        tree = cKDTree(support.xyz)
        normals, good = _normals(support.xyz,tree,scale,cancel,lambda f,m:progress(.05+.2*f,m),pool,workers)
        stats["reliable_normals"] = int(good.sum())
        if good.sum()<16:
            return PatchResult(None,stats)
        support, normals = _subset(support,good), normals[good]
        xyz = support.xyz
        _, axes = np.linalg.eigh(np.cov((xyz-xyz.mean(axis=0)).T))
        progress(.25,"Detecting bounded holes in connected surface charts")
        jobs = _chart_jobs(xyz,normals,axes,scale,scale,cancel,stats)
        stats["detected_holes"] = len(jobs)
        source_tree = cKDTree(cloud.xyz)
        chunks=[]
        proposal_count=0
        def run(index):
            return _make_patch(jobs[index],support,spacing,reach,minimum_distance,source_tree,cancel,max_candidates)
        for index,(patch,counts) in ordered_batches(run,len(jobs),1,pool,workers,cancel):
            for key,value in counts.items():
                stats[key] += value
            if stats["tested_candidates"] > max_candidates:
                raise ValueError("Boundary patches exceed the candidate budget. Increase spacing or crop the input.")
            if patch is not None:
                proposal_count += len(patch.xyz)
                if proposal_count > max_candidates:
                    raise ValueError("Boundary patches exceed the candidate budget. Increase spacing or crop the input.")
                chunks.append(patch)
            progress(.35+.65*(index+1)/max(len(jobs),1),f"Boundary patches: {index+1:,}/{len(jobs):,} surface patches")
        if not chunks:
            return PatchResult(None,stats)
        attributes={}
        for name in ("xyz","rgb","intensity","classification","generated","source_indices"):
            if getattr(chunks[0],name) is not None:
                attributes[name]=np.concatenate([getattr(chunk,name) for chunk in chunks])
        check_cancel(cancel)
        return PatchResult(Cloud(**attributes,metadata=cloud.metadata),stats)


def select_new_cells(patch, occupied, origin, spacing, limit, snap=False):
    """Choose an actual patch sample nearest each new world-cell centre."""
    if patch is None:
        return None, None
    keys = np.floor((patch.xyz-origin)/spacing).astype(np.int64)
    unique, groups = np.unique(keys,axis=0,return_inverse=True)
    groups = groups.reshape(-1)
    delta = patch.xyz-(origin+(keys+.5)*spacing)
    order = np.lexsort((np.arange(len(keys)),np.einsum("ij,ij->i",delta,delta),groups))
    first = np.r_[True,groups[order][1:]!=groups[order][:-1]]
    representatives = order[first]
    _, exists = _find(occupied,unique)
    keep = representatives[~exists]
    if len(keep)>limit:
        raise ValueError("Boundary patches exceed the output-point limit. Increase spacing or the point limit.")
    if not len(keep):
        return None, None
    output = _subset(patch,keep)
    new_keys = keys[keep]
    if snap:
        output.xyz = origin+(new_keys+.5)*spacing
    return output,new_keys
