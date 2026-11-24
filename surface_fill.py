"""Orientation-independent, conservative local point-cloud gap interpolation.

Only measured input samples support a fit. Generated samples can extend the
candidate frontier, but never become evidence for subsequent interpolation.
"""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import combinations, product
from threading import Event

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from cpu_workers import ordered_batches, resolve_workers

from cloud import Cloud, Progress, _find, _majority, auto_voxel_size, check_cancel, estimate_native_spacing
from hybrid import _grid, _join, _subset

NEIGHBOURS = np.array([v for v in product((-1, 0, 1), repeat=3) if v != (0, 0, 0)], dtype=np.int64)


@dataclass(frozen=True)
class Parameters:
    spacing: float | None = None
    max_gap_distance: float | None = None  # None = 3 times spacing
    passes: int = 3
    max_angle_deg: float = 35.0
    layer_tolerance: float = .5  # times spacing, measured from nearest support plane
    interpolation: str = "quadratic"
    max_added_points: int = 100_000
    max_candidates: int = 1_000_000
    workers: int = 0  # 0 = automatic, otherwise 1..32
    method: str = "local"  # local / boundary

    def validate(self):
        if self.method not in {"local", "boundary"}:
            raise ValueError("Surface filling method must be local or boundary.")
        if not isinstance(self.workers, int) or not 0 <= self.workers <= 32:
            raise ValueError("CPU workers must be Auto (0) or an integer from 1 to 32.")
        for name, value in [("Point spacing", self.spacing), ("Maximum gap distance", self.max_gap_distance)]:
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and greater than zero.")
        if not isinstance(self.passes, int) or not 1 <= self.passes <= 8:
            raise ValueError("Surface fill passes must be from 1 to 8.")
        if not np.isfinite(self.max_angle_deg) or not 5 <= self.max_angle_deg <= 75:
            raise ValueError("Normal angle tolerance must be from 5 to 75 degrees.")
        if not np.isfinite(self.layer_tolerance) or not .05 <= self.layer_tolerance <= 2:
            raise ValueError("Layer tolerance must be from 0.05 to 2 times spacing.")
        if self.interpolation not in {"plane", "quadratic"}:
            raise ValueError("Choose plane or quadratic surface interpolation.")
        if self.max_added_points < 1 or self.max_candidates < 1:
            raise ValueError("Point and candidate limits must be positive.")


@dataclass
class Result:
    cloud: Cloud
    spacing: float
    max_gap_distance: float
    input_count: int
    added_count: int
    passes: int
    support_points: int
    reliable_normals: int
    tested_candidates: int
    workers: int
    boundary_patches: dict | None = None


def _support(cloud, spacing):
    origin, occupied, _, _ = _grid(cloud.xyz, spacing)
    measured = np.flatnonzero(~cloud.generated)
    if not len(measured):
        raise ValueError("The input has no unmarked measured points to support interpolation.")
    keys, groups = np.unique(np.floor((cloud.xyz[measured]-origin)/spacing).astype(np.int64), axis=0, return_inverse=True)
    groups = groups.reshape(-1)
    centers = origin + (keys + .5) * spacing
    delta = cloud.xyz[measured] - centers[groups]
    order = np.lexsort((np.arange(len(groups)), np.einsum("ij,ij->i", delta, delta), groups))
    first = np.r_[True, groups[order][1:] != groups[order][:-1]]
    # Select an actual measured point, rather than averaging nearby sheets into
    # an invented intermediate support surface inside a shared voxel.
    indices = measured[order[first]]
    return origin, occupied, keys, _subset(cloud, indices)


def _normal_batch(xyz, tree, spacing, start, cancel):
    check_cancel(cancel)
    query = xyz[start:start + 1024]
    distance, indices = tree.query(query, k=32, distance_upper_bound=3 * spacing, workers=1)
    valid = np.isfinite(distance)
    indices = np.minimum(indices, len(xyz) - 1)
    count = valid.sum(axis=1)
    offsets = xyz[indices] - query[:, None, :]
    mean = (offsets * valid[:, :, None]).sum(axis=1) / np.maximum(count[:, None], 1)
    offsets = (offsets - mean[:, None, :]) * valid[:, :, None]
    covariance = np.einsum("nki,nkj->nij", offsets, offsets) / np.maximum(count[:, None, None], 1)
    values, vectors = np.linalg.eigh(covariance)
    good = (count >= 6) & (values[:, 1] > values[:, 2] * .05) & (values[:, 0] <= values.sum(axis=1) * .08)
    ambiguous = np.flatnonzero(~good & (count >= 6))
    if len(ambiguous):
        # Nearby cave sheets can make a single all-neighbour covariance
        # look volumetric. Deterministically test planes through the query
        # and pairs of close samples, then refit only the agreeing layer.
        raw = xyz[indices[ambiguous]] - query[ambiguous, None, :]
        pairs = np.array(list(combinations(range(1, 9), 2)))
        options = np.cross(raw[:, pairs[:, 0]], raw[:, pairs[:, 1]])
        length = np.linalg.norm(options, axis=2)
        options /= np.maximum(length[:, :, None], spacing*spacing*1e-12)
        depth = np.abs(np.einsum("nki,nji->njk", raw, options))
        inliers = (depth <= .35*spacing) & valid[ambiguous, None, :]
        inliers &= (length > spacing*spacing*1e-8)[:, :, None]
        weight = 1/(1 + (distance[ambiguous]/spacing)**2)
        score = (inliers*weight[:, None, :]).sum(axis=2)
        selected = np.argmax(score, axis=1)
        mask = inliers[np.arange(len(ambiguous)), selected]
        support_count = mask.sum(axis=1)
        center = (raw*mask[:, :, None]).sum(axis=1)/np.maximum(support_count[:, None], 1)
        centered = (raw-center[:, None, :])*mask[:, :, None]
        covariance = np.einsum("nki,nkj->nij", centered, centered)/np.maximum(support_count[:, None, None], 1)
        value, vector = np.linalg.eigh(covariance)
        recovered = (support_count >= 6) & (value[:, 1] > value[:, 2]*.05) & (value[:, 0] <= value.sum(axis=1)*.08)
        vectors[ambiguous] = vector
        good[ambiguous] = recovered
    check_cancel(cancel)
    return vectors[:, :, 0], good


def _normals(xyz, tree, spacing, cancel, progress, pool, workers):
    normals = np.zeros_like(xyz)
    reliable = np.zeros(len(xyz), dtype=bool)
    def run(start):
        return _normal_batch(xyz, tree, spacing, start, cancel)
    for start, (batch_normals, good) in ordered_batches(run, len(xyz), 1024, pool, workers, cancel):
        normals[start:start+len(good)] = batch_normals
        reliable[start:start+len(good)] = good
        progress(.08 + .17*(start+len(good))/len(xyz), "Estimating local surface directions")
    return normals, reliable


def _candidate_cells(front_keys, front_xyz, front_normals, occupied, origin, spacing, limit, cancel):
    chunks, total = [], 0
    # Keep neighbour cells intersecting the local tangent slab. This removes
    # most off-surface cells before the more expensive neighbourhood fitting.
    for delta in NEIGHBOURS:
        for start in range(0, len(front_keys), 32_768):
            check_cancel(cancel)
            target = slice(start, start + 32_768)
            cells = front_keys[target] + delta
            center = origin + (cells + .5) * spacing
            tangent = np.abs(np.einsum("ij,ij->i", center - front_xyz[target], front_normals[target])) <= .85 * spacing
            cells = cells[tangent]
            _, present = _find(occupied, cells)
            cells = cells[~present]
            if len(cells):
                chunks.append(cells)
                total += len(cells)
                if total > 4 * limit:
                    raise ValueError("Too many surface-fill candidates. Crop the cloud or increase point spacing.")
    if not chunks:
        return np.empty((0, 3), dtype=np.int64)
    candidates = np.unique(np.concatenate(chunks), axis=0)
    if len(candidates) > limit:
        raise ValueError(f"Surface filling exceeds {limit:,} candidate cells. Crop the cloud or increase point spacing.")
    return candidates


def _surrounded(u, v, valid, max_angle_gap=175):
    # In a local tangent plane, all supports on one side leave a >=180-degree
    # angular gap. Require slightly more coverage to avoid expanding open edges.
    angles = np.sort(np.where(valid, np.mod(np.arctan2(v, u), 2*np.pi), 2*np.pi), axis=1)
    count = valid.sum(axis=1)
    first, last = angles[:, 0], angles[np.arange(len(angles)), np.maximum(count - 1, 0)]
    differences = np.diff(angles, axis=1)
    differences = np.where(np.arange(angles.shape[1] - 1)[None, :] < count[:, None] - 1, differences, 0)
    max_gap = np.maximum(differences.max(axis=1), first + 2*np.pi - last)
    return (count >= 6) & (max_gap < np.deg2rad(max_angle_gap))


def _fit_candidates(centers, support, tree, source_tree, normals, reliable, spacing, gap, params, existing_tree=None,
                    *, neighbours=None, support_mask=None, fit_spacing=None, minimum_distance=None):
    distance, indices = tree.query(centers, k=96, distance_upper_bound=2*gap + spacing, workers=1) if neighbours is None else neighbours
    fit_spacing = spacing if fit_spacing is None else fit_spacing
    valid = np.isfinite(distance)
    indices = np.minimum(indices, len(support.xyz) - 1)
    anchor_index = indices[:, 0]
    anchor = support.xyz[anchor_index]
    reference = normals[anchor_index]
    points = support.xyz[indices]
    angle_ok = np.abs(np.einsum("nki,ni->nk", normals[indices], reference)) >= np.cos(np.deg2rad(params.max_angle_deg))
    depth = np.einsum("nki,ni->nk", points - anchor[:, None, :], reference)
    valid &= angle_ok & reliable[indices] & (np.abs(depth) <= params.layer_tolerance * spacing)
    if support_mask is not None:
        valid &= support_mask
    count = valid.sum(axis=1)
    offsets = points - anchor[:, None, :]
    mean = anchor + (offsets * valid[:, :, None]).sum(axis=1) / np.maximum(count[:, None], 1)
    offsets = (points - mean[:, None, :]) * valid[:, :, None]
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", offsets, offsets) / np.maximum(count[:, None, None], 1))
    normal, tangent_u, tangent_v = vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
    local = points - centers[:, None, :]
    u = np.einsum("nki,ni->nk", local, tangent_u) / fit_spacing
    v = np.einsum("nki,ni->nk", local, tangent_v) / fit_spacing
    w = np.einsum("nki,ni->nk", points - mean[:, None, :], normal) / fit_spacing
    surrounded = _surrounded(u, v, valid)
    good = reliable[anchor_index] & (count >= 6) & surrounded
    good &= (values[:, 1] > values[:, 2] * .03)
    good &= values[:, 0] <= (.35*fit_spacing)**2
    good &= np.abs(np.einsum("ij,ij->i", normal, reference)) >= np.cos(np.deg2rad(params.max_angle_deg))
    # Local plane is the stable fallback. Quadratic fitting only changes the
    # normal displacement at the candidate's projected tangent coordinates.
    height = np.zeros(len(centers))
    if params.interpolation == "quadratic":
        design = np.stack((np.ones_like(u), u, v, u*u, u*v, v*v), axis=2)
        weight = valid / (1 + u*u + v*v)
        gram = np.einsum("nki,nk,nkj->nij", design, weight, design)
        rhs = np.einsum("nki,nk,nk->ni", design, weight, w)
        condition = np.linalg.cond(gram)
        solve = good & (count >= 8) & np.isfinite(condition) & (condition < 1e8)
        if solve.any():
            coefficients = np.linalg.solve(gram[solve], rhs[solve, :, None])[..., 0]
            # All candidate tangent coordinates are zero in this design.
            height[solve] = coefficients[:, 0] * fit_spacing
        good &= np.abs(height) <= np.maximum(fit_spacing, 2*np.max(np.abs(w) * valid, axis=1)*fit_spacing)
    projected = centers - np.einsum("ij,ij->i", centers - mean, normal)[:, None] * normal + height[:, None] * normal
    measured_distance, _ = source_tree.query(projected, workers=1)
    lower_distance = .7*spacing if minimum_distance is None else minimum_distance
    good &= (measured_distance >= lower_distance) & (measured_distance <= gap)
    if existing_tree is not None:
        existing_distance, _ = existing_tree.query(projected, workers=1)
        good &= existing_distance >= .7*spacing
    good &= np.abs(np.einsum("ij,ij->i", projected - anchor, reference)) <= 1.5 * params.layer_tolerance * spacing
    good &= np.isfinite(projected).all(axis=1)
    # Attribute interpolation is based on the same accepted measured supports.
    safe_count = np.maximum(count, 1)
    rgb = (support.rgb[indices] * valid[:, :, None]).sum(axis=1) / safe_count[:, None] if support.rgb is not None else None
    intensity = (support.intensity[indices] * valid).sum(axis=1) / safe_count if support.intensity is not None else None
    classification = None
    if support.classification is not None:
        rows = np.repeat(np.arange(len(centers)), indices.shape[1])[valid.ravel()]
        classification = _majority(support.classification[indices[valid]], rows, len(centers)) if len(rows) else np.zeros(len(centers), dtype=np.uint8)
    accepted = Cloud(projected[good], rgb[good] if rgb is not None else None,
                     intensity[good] if intensity is not None else None,
                     classification[good] if classification is not None else None,
                     np.ones(good.sum(), dtype=bool), np.full(good.sum(), -1, dtype=np.int64), support.metadata) if good.any() else None
    return accepted, normal[good]


def _separate(points, minimum_distance, cancel=None):
    # Greedy deterministic spatial buckets: cap close samples across voxel
    # boundaries without materialising a potentially large all-pairs graph.
    buckets, keep = {}, []
    square = minimum_distance * minimum_distance
    for index, point in enumerate(points):
        if index % 1024 == 0:
            check_cancel(cancel)
        key = tuple(np.floor(point / minimum_distance).astype(np.int64))
        too_close = False
        for delta in product((-1, 0, 1), repeat=3):
            for other in buckets.get(tuple(k+d for k, d in zip(key, delta)), ()):
                diff = point - points[other]
                if np.dot(diff, diff) < square:
                    too_close = True
                    break
            if too_close:
                break
        if not too_close:
            keep.append(index)
            buckets.setdefault(key, []).append(index)
    return np.array(keep, dtype=np.int64)


def fill_gaps(cloud: Cloud, params: Parameters, cancel: Event | None = None, progress: Progress | None = None) -> Result:
    params.validate()
    check_cancel(cancel)
    workers = resolve_workers(params.workers)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="surface-fill") if workers > 1 else nullcontext(None)
    # BLAS settings are process-wide. The GUI runs one processing job at a time;
    # cap nested BLAS threads once around all worker batches, then restore them.
    with threadpool_limits(limits=1, user_api="blas"), executor as pool:
        return _fill_gaps(cloud, params, cancel, progress, pool, workers)


def _fill_gaps(cloud, params, cancel, progress, pool, workers) -> Result:
    params.validate()
    progress = progress or (lambda fraction, message: None)
    check_cancel(cancel)
    progress(.01, "Preparing measured surface supports")
    spacing = auto_voxel_size(cloud.xyz, cancel, workers=workers) if params.spacing is None else params.spacing
    gap = 3*spacing if params.max_gap_distance is None else params.max_gap_distance
    if gap < .7*spacing:
        raise ValueError("Maximum gap distance must be at least 0.7 times point spacing.")
    if params.method == "boundary":
        from boundary_patch import complete, select_new_cells
        measured = cloud.xyz[~cloud.generated]
        if not len(measured):
            raise ValueError("The input has no unmarked measured points to support interpolation.")
        native = estimate_native_spacing(measured,cancel,workers)
        patches = complete(cloud,spacing,gap,native,workers,cancel,progress,
                           minimum_distance=.25*spacing,max_candidates=4*params.max_candidates)
        origin,occupied,_,_ = _grid(cloud.xyz,spacing)
        extra,_ = select_new_cells(patches.cloud,occupied,origin,spacing,params.max_added_points)
        if extra is not None:
            extra = _subset(extra,_separate(extra.xyz,.25*spacing,cancel))
        added = 0 if extra is None else len(extra.xyz)
        patches.stats["added_output_cells"] = added
        check_cancel(cancel)
        output = cloud if extra is None else _join(cloud,extra)
        return Result(output,spacing,gap,len(cloud.xyz),added,int(added>0),
                      patches.stats.get("support_points",0),patches.stats.get("reliable_normals",0),
                      patches.stats.get("tested_candidates",0),workers,patches.stats)
    origin, occupied, support_keys, support = _support(cloud, spacing)
    tree, source_tree = cKDTree(support.xyz), cKDTree(cloud.xyz[~cloud.generated])
    existing_tree = cKDTree(cloud.xyz) if cloud.generated.any() else None
    normals, reliable = _normals(support.xyz, tree, spacing, cancel, progress, pool, workers)
    if reliable.sum() < 6:
        raise ValueError("Too few reliable local surface directions. Try a different spacing or a larger surface crop.")
    front_keys, front_xyz, front_normals = support_keys[reliable], support.xyz[reliable], normals[reliable]
    output, generated_xyz, added, passes, tested = cloud, None, 0, 0, 0
    for pass_index in range(params.passes):
        check_cancel(cancel)
        progress(.25 + .65*pass_index/params.passes, f"3D surface fill pass {pass_index+1}: finding gaps")
        candidates = _candidate_cells(front_keys, front_xyz, front_normals, occupied, origin, spacing, params.max_candidates, cancel)
        tested += len(candidates)
        chunks, normal_chunks = [], []
        def run_fit(start):
            check_cancel(cancel)
            centers = origin + (candidates[start:start+1024] + .5)*spacing
            result = _fit_candidates(centers, support, tree, source_tree, normals, reliable, spacing, gap, params, existing_tree)
            check_cancel(cancel)
            return result
        for start, (accepted, fitted_normals) in ordered_batches(run_fit, len(candidates), 1024, pool, workers, cancel):
            if accepted is not None:
                chunks.append(accepted)
                normal_chunks.append(fitted_normals)
            done = min(start+1024, len(candidates))
            progress(.25 + .65*(pass_index + done/max(len(candidates), 1))/params.passes, f"Checking {len(candidates):,} 3D gap candidates")
        if not chunks:
            break
        new = chunks[0]
        for chunk in chunks[1:]:
            new = _join(new, chunk)
        new_normals = np.concatenate(normal_chunks)
        keys = np.floor((new.xyz - origin)/spacing).astype(np.int64)
        keys, first = np.unique(keys, axis=0, return_index=True)
        _, present = _find(occupied, keys)
        keep = first[~present]
        if generated_xyz is not None and len(keep):
            distance, _ = cKDTree(generated_xyz).query(new.xyz[keep], workers=workers)
            keep = keep[distance >= .7*spacing]
        if not len(keep):
            break
        keep = keep[_separate(new.xyz[keep], .7*spacing, cancel)]
        check_cancel(cancel)
        if added + len(keep) > params.max_added_points:
            raise ValueError(f"3D filling would exceed {params.max_added_points:,} added points. Reduce passes, crop the scan, or increase the limit.")
        if not len(keep):
            break
        new = _subset(new, keep)
        front_normals = new_normals[keep]
        front_xyz = new.xyz
        front_keys = np.floor((front_xyz - origin)/spacing).astype(np.int64)
        occupied = np.unique(np.concatenate((occupied, front_keys)), axis=0)
        output = _join(output, new)
        generated_xyz = new.xyz if generated_xyz is None else np.concatenate((generated_xyz, new.xyz))
        added += len(new.xyz)
        passes += 1
    check_cancel(cancel)
    progress(1, f"Ready: {added:,} estimated 3D surface points added")
    return Result(output, spacing, gap, len(cloud.xyz), added, passes, len(support.xyz), int(reliable.sum()), tested, workers)
