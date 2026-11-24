"""Port of HybridRegridFilter.cpp: heightfield plus classified 3D voxels."""
from dataclasses import dataclass
from itertools import product
from threading import Event

import numpy as np

from cloud import Cloud, Progress, _find, _majority, _mean, auto_voxel_size, check_cancel

NEIGHBOURS = np.array([p for p in product((-1, 0, 1), repeat=2) if p != (0, 0)], dtype=np.int64)


@dataclass(frozen=True)
class Parameters:
    cell_size: float | None = None
    voxel_size: float | None = None  # None means same as heightfield cell
    height_stat: str = "median"
    fill_passes: int = 1
    wall_slope_deg: float = 45.0
    keep_below_surface: bool = True
    below_delta_mul: float = 1.5
    max_added_points: int = 200_000

    def validate(self):
        for name, value in [("Heightfield cell size", self.cell_size), ("Voxel size", self.voxel_size)]:
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite number greater than zero.")
        if self.height_stat not in {"median", "mean", "max", "min"}:
            raise ValueError("Height statistic must be median, mean, max or min.")
        if not isinstance(self.fill_passes, int) or not 0 <= self.fill_passes <= 8:
            raise ValueError("Heightfield hole-fill passes must be from 0 to 8.")
        if not np.isfinite(self.wall_slope_deg) or not 0 <= self.wall_slope_deg <= 89.5:
            raise ValueError("Wall slope threshold must be from 0 to 89.5 degrees.")
        if not np.isfinite(self.below_delta_mul) or not 0 <= self.below_delta_mul <= 20:
            raise ValueError("Cave delta must be from 0 to 20 times the cell size.")
        if not isinstance(self.max_added_points, int) or self.max_added_points < 1:
            raise ValueError("Maximum added points must be a positive integer.")


@dataclass
class Result:
    cloud: Cloud
    cell_size: float
    voxel_size: float
    input_count: int
    heightfield_count: int
    occupied_voxels: int
    added_count: int
    passes: int
    walls_kept: int
    caves_kept: int
    fringe_kept: int
    heightfield_origin: np.ndarray
    voxel_origin: np.ndarray


def _grid(xyz, size):
    minimum = xyz.min(axis=0) / size
    if not np.isfinite(minimum).all() or np.max(np.abs(minimum)) > 2**50:
        raise ValueError("Cell size is too small for the coordinate magnitudes.")
    origin = np.floor(minimum) * size
    scaled = (xyz - origin) / size
    if not np.isfinite(scaled).all() or np.max(scaled) > 2**50:
        raise ValueError("Cell size is too small for the input extent.")
    keys, groups, counts = np.unique(np.floor(scaled).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    return origin, keys, groups.reshape(-1), counts


def _aggregate(cloud, groups, counts, xyz):
    distance = np.einsum("ij,ij->i", cloud.xyz - xyz[groups], cloud.xyz - xyz[groups])
    order = np.lexsort((np.arange(len(groups)), distance, groups))
    first = np.r_[True, groups[order][1:] != groups[order][:-1]]
    representative = order[first]
    return Cloud(xyz, _mean(cloud.rgb, groups, counts), _mean(cloud.intensity, groups, counts),
                 _majority(cloud.classification, groups, len(counts)) if cloud.classification is not None else None,
                 np.bincount(groups, weights=cloud.generated, minlength=len(counts)) > 0,
                 cloud.source_indices[representative], cloud.metadata)


def _join(a, b, order=None):
    values = {}
    for name in ("xyz", "rgb", "intensity", "classification", "generated", "source_indices"):
        left, right = getattr(a, name), getattr(b, name)
        if left is not None:
            combined = np.concatenate((left, right))
            values[name] = combined if order is None else combined[order]
    return Cloud(**values, metadata=a.metadata)


def _subset(cloud, indices):
    return Cloud(**{name: getattr(cloud, name)[indices] for name in ("xyz", "rgb", "intensity", "classification", "generated", "source_indices") if getattr(cloud, name) is not None}, metadata=cloud.metadata)


def _gap_cells(keys, cancel):
    chunks = []
    for axis in range(2):
        for start in range(0, len(keys), 32_768):
            check_cancel(cancel)
            part = keys[start:start + 32_768]
            far = part.copy()
            far[:, axis] += 2
            _, paired = _find(keys, far)
            candidate = part[paired].copy()
            candidate[:, axis] += 1
            _, present = _find(keys, candidate)
            if np.any(~present):
                chunks.append(candidate[~present])
    return np.unique(np.concatenate(chunks), axis=0) if chunks else np.empty((0, 2), dtype=np.int64)


def _height_slopes(keys, heights, cell_size, cancel):
    """Actual C++ rule: steepest of eight neighbour gradients, then atan."""
    slope = np.zeros(len(keys))
    for delta in NEIGHBOURS:
        for start in range(0, len(keys), 65_536):
            check_cancel(cancel)
            target = slice(start, start + 65_536)
            index, valid = _find(keys, keys[target] + delta)
            difference = np.abs(heights[index] - heights[target]) / (cell_size * np.linalg.norm(delta))
            slope[target] = np.maximum(slope[target], np.where(valid, difference, 0))
    return np.degrees(np.arctan(slope))


def _classify(voxel_xyz, keys, heights, slopes, origin, cell_size, params):
    projected = np.floor((voxel_xyz[:, :2] - origin) / cell_size).astype(np.int64)
    indices, present = _find(keys, projected)
    fringe = ~present
    wall = present & (slopes[indices] >= params.wall_slope_deg)
    cave = present & ~wall & params.keep_below_surface & ((heights[indices] - voxel_xyz[:, 2]) > params.below_delta_mul * cell_size)
    return wall, cave, fringe


def regrid(cloud: Cloud, params: Parameters, cancel: Event | None = None, progress: Progress | None = None) -> Result:
    params.validate()
    progress = progress or (lambda fraction, message: None)
    check_cancel(cancel)
    progress(.01, "Estimating cell size" if params.cell_size is None else "Building heightfield")
    cell = auto_voxel_size(cloud.xyz, cancel) if params.cell_size is None else params.cell_size
    voxel = cell if params.voxel_size is None else params.voxel_size

    # A: one robust height and mean XY/attributes per occupied XY column.
    origin, keys, groups, counts = _grid(cloud.xyz[:, :2], cell)
    xy = _mean(cloud.xyz[:, :2] - origin, groups, counts) + origin
    order = np.lexsort((cloud.xyz[:, 2], groups))
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    z_sorted = cloud.xyz[order, 2]
    if params.height_stat == "median":
        # C++ nth_element returns the upper middle sample for even counts.
        height = z_sorted[starts + counts // 2]
    elif params.height_stat == "min":
        height = z_sorted[starts]
    elif params.height_stat == "max":
        height = z_sorted[starts + counts - 1]
    else:
        z_origin = float(cloud.xyz[:, 2].min())
        height = _mean(cloud.xyz[:, 2] - z_origin, groups, counts) + z_origin
    hf = _aggregate(cloud, groups, counts, np.column_stack((xy, height)))
    original_hf, added, passes = len(keys), 0, 0
    check_cancel(cancel)
    progress(.25, f"Heightfield: {original_hf:,} occupied XY cells")

    # A.5: synchronous 2D sandwich fill. Each added cell is at its XY centre.
    for pass_index in range(params.fill_passes):
        candidates = _gap_cells(keys, cancel)
        if not len(candidates):
            break
        if added + len(candidates) > params.max_added_points:
            raise ValueError(f"Heightfield filling would exceed {params.max_added_points:,} added points. Reduce passes, crop the scan, or increase the limit.")
        new_xyz = np.empty((len(candidates), 3))
        new_rgb = np.empty_like(new_xyz) if hf.rgb is not None else None
        new_intensity = np.empty(len(candidates)) if hf.intensity is not None else None
        # Like C++, newly filled cells have unclassified LAS class 0.
        new_class = np.zeros(len(candidates), dtype=np.uint8) if hf.classification is not None else None
        for start in range(0, len(candidates), 8192):
            check_cancel(cancel)
            cells = candidates[start:start + 8192]
            indices, valid = _find(keys, (cells[:, None, :] + NEIGHBOURS).reshape(-1, 2))
            indices, valid = indices.reshape(-1, 8), valid.reshape(-1, 8)
            support = valid.sum(axis=1)
            target = slice(start, start + len(cells))
            new_xyz[target, :2] = origin + (cells + .5) * cell
            # Equal weight per neighbouring heightfield cell, matching C++.
            z_anchor = hf.xyz[indices[:, 0], 2]
            new_xyz[target, 2] = z_anchor + ((hf.xyz[indices, 2] - z_anchor[:, None]) * valid).sum(axis=1) / support
            # Propagate attributes from earlier fill passes too, avoiding the
            # black/uninitialised infill possible in the original implementation.
            if new_rgb is not None:
                new_rgb[target] = (hf.rgb[indices] * valid[:, :, None]).sum(axis=1) / support[:, None]
            if new_intensity is not None:
                new_intensity[target] = (hf.intensity[indices] * valid).sum(axis=1) / support
        new = Cloud(new_xyz, new_rgb, new_intensity, new_class, np.ones(len(candidates), dtype=bool), np.full(len(candidates), -1, dtype=np.int64), cloud.metadata)
        keys = np.concatenate((keys, candidates))
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys = keys[order]
        hf = _join(hf, new, order)
        added += len(candidates)
        passes += 1
        progress(.25 + .15 * (pass_index + 1) / params.fill_passes, f"Heightfield fill pass {pass_index + 1}: {len(candidates):,} added cells")

    # B: a separate voxel grid always comes from the original input points.
    check_cancel(cancel)
    progress(.45, "Building 3D voxel grid from input")
    voxel_origin, voxel_keys, vgroups, vcounts = _grid(cloud.xyz, voxel)
    centroids = _mean(cloud.xyz - voxel_origin, vgroups, vcounts) + voxel_origin
    voxels = _aggregate(cloud, vgroups, vcounts, centroids)

    # C/D: append voxel representatives for steep cells, below-surface points,
    # and points outside the heightfield's XY coverage, in that priority order.
    progress(.68, "Measuring heightfield slopes")
    slope = _height_slopes(keys, hf.xyz[:, 2], cell, cancel)
    check_cancel(cancel)
    progress(.85, "Keeping wall, cave and fringe voxel points")
    wall, cave, fringe = _classify(voxels.xyz, keys, hf.xyz[:, 2], slope, origin, cell, params)
    keep = wall | cave | fringe
    # As in the editor, emit all HF points plus retained voxels. They may overlap
    # on steep surfaces; there is no unrelated global smoothing or 3D dilation.
    output = _join(hf, _subset(voxels, keep)) if keep.any() else hf
    check_cancel(cancel)
    progress(1, f"Ready: {len(output.xyz):,} points")
    return Result(output, cell, voxel, len(cloud.xyz), original_hf, len(voxel_keys), added, passes,
                  int(wall.sum()), int(cave.sum()), int(fringe.sum()), origin, voxel_origin)
