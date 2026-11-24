"""Rebuild a complete surface point cloud on a sparse, uniform 3D voxel grid.

The fixed input is the reconstruction reference. Wider directional probe
neighbourhoods reach opposite hole rims without requiring a dense volume grid.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import product
from threading import Event

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from cloud import Cloud, Progress, _find, _mean, check_cancel, estimate_native_spacing
from cpu_workers import ordered_batches, resolve_workers
from hybrid import _aggregate, _grid, _join, _subset
from surface_fill import NEIGHBOURS, _fit_candidates, _normals, _support


@dataclass(frozen=True)
class Parameters:
    spacing: float | None = None
    positions: str = "grid"  # exact lattice centres, or surface-projected
    completion: str = "balanced"  # off / balanced / strong
    reach: float | None = None  # auto = 6 times native reference spacing
    max_output_points: int = 2_000_000
    workers: int = 0
    max_candidates: int = 20_000_000

    def validate(self):
        for name, value in [("Grid spacing", self.spacing), ("Reconstruction reach", self.reach)]:
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and greater than zero.")
        if self.positions not in {"grid", "surface"}:
            raise ValueError("Choose exact grid centres or surface-projected output.")
        if self.completion not in {"off", "balanced", "strong"}:
            raise ValueError("Gap completion must be Off, Balanced or Strong.")
        if not isinstance(self.workers, int) or not 0 <= self.workers <= 32:
            raise ValueError("CPU workers must be Auto (0) or an integer from 1 to 32.")
        if not isinstance(self.max_output_points, int) or self.max_output_points < 1:
            raise ValueError("Maximum output points must be a positive integer.")
        if not isinstance(self.max_candidates, int) or self.max_candidates < 1:
            raise ValueError("Candidate limit must be a positive integer.")


@dataclass
class Result:
    cloud: Cloud
    spacing: float
    native_spacing: float
    reach: float
    input_count: int
    occupied_cells: int
    added_count: int
    rounds: int
    tested_candidates: int
    reliable_supports: int
    workers: int
    origin: np.ndarray
    stop_reason: str
    boundary_patches: dict | None = None


@dataclass(frozen=True)
class _PatchSettings:
    max_angle_deg: float
    layer_tolerance: float
    interpolation: str = "quadratic"


def _concat(chunks):
    first = chunks[0]
    values = {}
    for name in ("xyz", "rgb", "intensity", "classification", "generated", "source_indices"):
        if getattr(first, name) is not None:
            values[name] = np.concatenate([getattr(c, name) for c in chunks])
    return Cloud(**values, metadata=first.metadata)


def _candidate_tiles(front_keys, front_xyz, front_normals, occupied, origin, spacing, limit, cancel):
    """Stream disjoint 32^3 tiles, including a one-cell frontier halo.

    Ownership is by target cell, so a boundary candidate is tested once. The
    temporary neighbour arrays stay bounded even for a million-point input;
    the candidate limit counts unique cells rather than duplicate proposals.
    """
    tile_keys, inverse = np.unique(front_keys // 32, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(tile_keys))) ]
    groups = {tuple(key): order[offsets[i]:offsets[i+1]] for i, key in enumerate(tile_keys)}
    neighbours = np.array(list(product((-1, 0, 1), repeat=3)), dtype=np.int64)
    tiles = np.unique((tile_keys[:, None, :]+neighbours[None, :, :]).reshape(-1, 3), axis=0)
    total = 0
    for tile_index, tile in enumerate(tiles):
        check_cancel(cancel)
        parts = [groups[tuple(key)] for key in tile+neighbours if tuple(key) in groups]
        indices = np.concatenate(parts)
        low, high = tile*32, (tile+1)*32
        keys = front_keys[indices]
        indices = indices[np.all((keys >= low-1) & (keys <= high), axis=1)]
        if not len(indices):
            continue
        keys, xyz, normal = front_keys[indices], front_xyz[indices], front_normals[indices]
        cells = keys[:, None, :]+NEIGHBOURS[None, :, :]
        center_offset = origin+(keys+.5)*spacing-xyz
        depth = np.einsum("ni,ni->n", center_offset, normal)[:, None] + spacing*(normal @ NEIGHBOURS.T)
        valid = (np.abs(depth) <= .85*spacing) & np.all((cells >= low) & (cells < high), axis=2)
        cells = np.unique(cells[valid], axis=0)
        if len(cells):
            _, present = _find(occupied, cells)
            cells = cells[~present]
        if not len(cells):
            continue
        total += len(cells)
        if total > limit:
            raise ValueError(f"Rebuilding exceeds {limit:,} candidate cells in one round. Increase grid spacing or crop the cloud.")
        yield cells, (tile_index+1)/len(tiles), total


def _directional_support(centers, support, tree, normals, native, reach):
    """Near samples plus 12 directions at three radii, instead of nearest-only.

    All query results are still actual fixed reference points. Probe positions
    are only lookup locations and never become interpolation evidence.
    """
    near_distance, near = tree.query(centers, k=24, workers=1)
    anchor_indices = near[:, 0]
    reference = normals[anchor_indices]
    anchor = support.xyz[anchor_indices]
    axis = np.zeros_like(reference)
    use_z = np.abs(reference[:, 2]) < .9
    axis[use_z, 2], axis[~use_z, 1] = 1, 1
    tangent_u = np.cross(reference, axis)
    tangent_u /= np.maximum(np.linalg.norm(tangent_u, axis=1)[:, None], 1e-12)
    tangent_v = np.cross(reference, tangent_u)
    projected = centers - np.einsum("ij,ij->i", centers-anchor, reference)[:, None]*reference
    angles = np.arange(12)*2*np.pi/12
    directions = np.cos(angles)[None, :, None]*tangent_u[:, None, :] + np.sin(angles)[None, :, None]*tangent_v[:, None, :]
    radii = np.array([.5, 1, 2])*reach
    probes = projected[:, None, None, :] + radii[None, :, None, None]*directions[:, None, :, :]
    _, ring = tree.query(probes.reshape(-1, 3), k=2, distance_upper_bound=reach+native, workers=1)
    indices = np.concatenate((near, ring.reshape(len(centers), -1)), axis=1)
    # De-duplicate within each row without changing the nearest anchor at slot 0.
    previous = indices[:, :, None] == indices[:, None, :]
    duplicate = (previous & np.tril(np.ones((indices.shape[1], indices.shape[1]), dtype=bool), -1)[None, :, :]).any(axis=2)
    valid = (indices < len(support.xyz)) & ~duplicate
    indices = np.minimum(indices, len(support.xyz)-1)
    points = support.xyz[indices]
    distance = np.linalg.norm(points-centers[:, None, :], axis=2)
    valid &= distance <= 2*reach+native
    depth = np.einsum("nki,ni->nk", points-anchor[:, None, :], reference)
    # Split separated depth layers and select the component containing the
    # nearest reference point. Curved connected depths can remain in one layer.
    depth_order = np.argsort(np.where(valid, depth, np.inf), axis=1, kind="stable")
    sorted_depth = np.take_along_axis(depth, depth_order, axis=1)
    sorted_valid = np.take_along_axis(valid, depth_order, axis=1)
    breaks = np.ones_like(sorted_valid)
    breaks[:, 1:] = (~sorted_valid[:, 1:]) | (~sorted_valid[:, :-1]) | (np.diff(sorted_depth, axis=1) > .6*native)
    labels = np.cumsum(breaks, axis=1)
    inverse = np.argsort(depth_order, axis=1)
    labels = np.take_along_axis(labels, inverse, axis=1)
    valid &= labels == labels[:, :1]
    return (np.where(valid, distance, np.inf), indices), valid


def rebuild(cloud: Cloud, params: Parameters, cancel: Event | None = None, progress: Progress | None = None) -> Result:
    params.validate()
    check_cancel(cancel)
    workers = resolve_workers(params.workers)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="uniform-rebuild") if workers > 1 else nullcontext(None)
    with threadpool_limits(limits=1, user_api="blas"), executor as pool:
        return _rebuild(cloud, params, cancel, progress or (lambda f, m: None), pool, workers)


def _rebuild(cloud, params, cancel, progress, pool, workers):
    progress(.01, "Measuring native input spacing")
    native = estimate_native_spacing(cloud.xyz, cancel, workers)
    spacing = native if params.spacing is None else params.spacing
    reach = 6*native if params.reach is None else params.reach
    origin, occupied, groups, counts = _grid(cloud.xyz, spacing)
    if len(occupied) > params.max_output_points:
        raise ValueError(f"The observed grid already needs {len(occupied):,} points, above the output limit. Increase grid spacing or the output limit.")
    centroid = _mean(cloud.xyz-origin, groups, counts)+origin
    positions = origin+(occupied+.5)*spacing if params.positions == "grid" else centroid
    output = _aggregate(cloud, groups, counts, positions)
    initial = len(occupied)
    check_cancel(cancel)
    if params.completion == "off":
        progress(1, f"New grid cloud: {len(output.xyz):,} occupied surface cells")
        return Result(output, spacing, native, reach, len(cloud.xyz), initial, 0, 0, 0, 0, workers, origin, "Completion disabled")

    # In this rebuild mode the provided file is the fixed geometric reference,
    # including points marked synthetic by earlier tools. New rounds never add
    # output samples to this reference or to its normal-estimation model.
    reference = Cloud(cloud.xyz, cloud.rgb, cloud.intensity, cloud.classification,
                      np.zeros(len(cloud.xyz), dtype=bool), cloud.source_indices, cloud.metadata)
    _, _, _, support = _support(reference, native)
    tree = cKDTree(support.xyz)
    progress(.07, "Estimating reference surface directions")
    normals, reliable = _normals(support.xyz, tree, 2*native, cancel, lambda f, m: progress(f, m), pool, workers)
    if reliable.sum() < 6:
        raise ValueError("Not enough reliable surface support for reconstruction. Use completion Off to resample only, or try a larger input region.")
    support, normals = _subset(support, reliable), normals[reliable]
    tree, source_tree = cKDTree(support.xyz), cKDTree(cloud.xyz)
    reliable = np.ones(len(support.xyz), dtype=bool)
    _, nearest = tree.query(centroid, workers=workers)
    front_keys, front_xyz, front_normals = occupied, centroid, normals[nearest]
    # Balanced keeps a smaller normal-depth band. Strong permits rougher and
    # more curved patches; depth components still protect separated sheets.
    angle = 55 if params.completion == "balanced" else 75
    depth_limit = max(.75*native, (.15 if params.completion == "balanced" else .35)*reach)
    fit_scale = max(native, reach*(.2 if params.completion == "balanced" else .4))
    patch = _PatchSettings(angle, depth_limit/spacing)
    max_rounds = min(512, max(2, 2*int(np.ceil(reach/spacing))+2))
    added = tested = rounds = 0
    stop = "No more surface-supported cells"
    for iteration in range(max_rounds):
        check_cancel(cancel)
        stage = .25+.65*iteration/max_rounds
        progress(stage, f"Rebuilding uniform surface: round {iteration+1}")
        chunks, normal_chunks = [], []
        for candidates, tile_fraction, tile_tested in _candidate_tiles(front_keys, front_xyz, front_normals, occupied, origin, spacing, params.max_candidates, cancel):
            tested += len(candidates)
            def fit(start):
                check_cancel(cancel)
                centers = origin+(candidates[start:start+512]+.5)*spacing
                near, mask = _directional_support(centers, support, tree, normals, native, reach)
                result = _fit_candidates(centers, support, tree, source_tree, normals, reliable, spacing, reach, patch,
                                         neighbours=near, support_mask=mask, fit_spacing=fit_scale, minimum_distance=0)
                check_cancel(cancel)
                return result
            for start, (accepted, fitted) in ordered_batches(fit, len(candidates), 512, pool, workers, cancel):
                if accepted is not None:
                    chunks.append(accepted)
                    normal_chunks.append(fitted)
                progress(stage+.65/max_rounds*tile_fraction, f"Round {iteration+1}: {tile_tested:,} grid candidates; {len(output.xyz):,} output points")
        if not chunks:
            break
        new = _concat(chunks)
        fitted = np.concatenate(normal_chunks)
        # Projection can cross a cell boundary. Re-key before merging so every
        # output cell is emitted exactly once, including observed input cells.
        new_keys = np.floor((new.xyz-origin)/spacing).astype(np.int64)
        unique, first = np.unique(new_keys, axis=0, return_index=True)
        _, exists = _find(occupied, unique)
        keep = first[~exists]
        if not len(keep):
            break
        if len(output.xyz)+len(keep) > params.max_output_points:
            raise ValueError(f"Rebuilding would exceed {params.max_output_points:,} output points. Increase grid spacing, reduce reach, or increase the limit.")
        new = _subset(new, keep)
        front_xyz = new.xyz.copy()
        front_normals = fitted[keep]
        front_keys = np.floor((front_xyz-origin)/spacing).astype(np.int64)
        if params.positions == "grid":
            new.xyz = origin+(front_keys+.5)*spacing
        occupied = np.unique(np.concatenate((occupied, front_keys)), axis=0)
        output = _join(output, new)
        added += len(new.xyz)
        rounds += 1
    else:
        stop = "Reached reconstruction round limit; increase spacing if coverage is incomplete"
    boundary_stats = None
    if params.completion == "strong":
        from boundary_patch import complete, select_new_cells
        patches = complete(reference, spacing, reach, native, workers, cancel,
                           lambda f,m: progress(.9+.09*f,m), max_candidates=params.max_candidates)
        boundary_stats = patches.stats
        extra, _ = select_new_cells(patches.cloud, occupied, origin, spacing,
                                    params.max_output_points-len(output.xyz), params.positions == "grid")
        boundary_stats["added_output_cells"] = 0 if extra is None else len(extra.xyz)
        if extra is not None:
            output = _join(output,extra)
            added += len(extra.xyz)
        stop += "; bounded-hole patch stage completed"
    check_cancel(cancel)
    progress(1, f"New uniform cloud: {len(output.xyz):,} points, {added:,} inferred surface cells")
    return Result(output, spacing, native, reach, len(cloud.xyz), initial, added, rounds, tested,
                  len(support.xyz), workers, origin, stop, boundary_stats)
