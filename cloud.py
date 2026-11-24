"""Shared point-cloud data, spacing estimation and sparse-grid helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable

import numpy as np

Progress = Callable[[float, str], None]


class Cancelled(Exception):
    """A cooperative cancellation, with no output committed."""


def check_cancel(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled("Cancelled")


@dataclass
class Cloud:
    xyz: np.ndarray
    rgb: np.ndarray | None = None  # floating point RGB in [0, 1]
    intensity: np.ndarray | None = None
    classification: np.ndarray | None = None
    generated: np.ndarray | None = None
    source_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not len(self.xyz):
            raise ValueError("The input must contain at least one XYZ point.")
        if not np.isfinite(self.xyz).all():
            raise ValueError("The input contains invalid coordinates (NaN or infinity).")
        for name in ("rgb", "intensity", "classification", "generated", "source_indices"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value)
                if len(value) != len(self.xyz):
                    raise ValueError(f"{name}: attribute count differs from point count.")
                if not np.isfinite(value).all():
                    raise ValueError(f"{name}: invalid attribute values.")
                setattr(self, name, value)
        if self.rgb is not None and (self.rgb.shape != self.xyz.shape or np.any((self.rgb < 0) | (self.rgb > 1))):
            raise ValueError("RGB must have three channels in the range 0 to 1.")
        if self.classification is not None:
            if np.any((self.classification < 0) | (self.classification > 255) | (self.classification != np.floor(self.classification))):
                raise ValueError("Classification must contain integer values from 0 to 255.")
        for name in ("intensity", "classification", "generated", "source_indices"):
            value = getattr(self, name)
            if value is not None and value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional array.")
        if self.generated is None:
            self.generated = np.zeros(len(self.xyz), dtype=bool)
        else:
            self.generated = self.generated.astype(bool)
        if self.source_indices is None:
            self.source_indices = np.arange(len(self.xyz), dtype=np.int64)


def auto_voxel_size(xyz: np.ndarray, cancel: Event | None = None, workers: int = -1) -> float:
    """2.5 times median nearest-neighbour spacing, as in the editor.

    Sample queries are deterministic; their neighbours come from the full cloud.
    Duplicates are skipped rather than interpreted as zero spacing.
    """
    from scipy.spatial import cKDTree

    check_cancel(cancel)
    span = np.ptp(xyz, axis=0)
    diagonal = float(np.linalg.norm(span))
    if diagonal == 0:
        return 1.0
    sample = xyz[np.linspace(0, len(xyz) - 1, min(2048, len(xyz)), dtype=np.int64)]
    tree = cKDTree(xyz)
    check_cancel(cancel)
    distances, _ = tree.query(sample, k=min(8, len(xyz)), workers=workers)
    if distances.ndim == 1:
        distances = distances[:, None]
    positive = np.where(distances > 0, distances, np.inf).min(axis=1)
    positive = positive[np.isfinite(positive)]
    if len(positive):
        estimate = float(np.median(positive)) * 2.5
    else:
        estimate = diagonal / max(1, np.sqrt(len(xyz)))
    check_cancel(cancel)
    return float(np.clip(estimate, diagonal * 1e-5, diagonal * 0.05))


def estimate_native_spacing(xyz, cancel=None, workers=1):
    """Median non-zero input spacing, independent of output grid density."""
    from scipy.spatial import cKDTree
    check_cancel(cancel)
    sample = xyz[np.linspace(0, len(xyz)-1, min(2048, len(xyz)), dtype=np.int64)]
    distance, _ = cKDTree(xyz).query(sample, k=min(8, len(xyz)), workers=workers)
    if distance.ndim == 1:
        distance = distance[:, None]
    nearest = np.where(distance > 0, distance, np.inf).min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    check_cancel(cancel)
    return float(np.median(nearest)) if len(nearest) else 1.0


def _view(keys: np.ndarray) -> np.ndarray:
    dtype = np.dtype([(f"c{i}", "<i8") for i in range(keys.shape[1])])
    return np.ascontiguousarray(keys, dtype=np.int64).view(dtype).reshape(-1)


def _find(sorted_keys: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact membership in a sparse sorted grid, without a dense bounding volume."""
    key_view, query_view = _view(sorted_keys), _view(queries)
    indices = np.searchsorted(key_view, query_view)
    indices = np.minimum(indices, len(sorted_keys) - 1)
    return indices, key_view[indices] == query_view


def _mean(values: np.ndarray | None, groups: np.ndarray, count: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    if values.ndim == 2:
        return np.column_stack([np.bincount(groups, weights=values[:, i], minlength=len(count)) / count for i in range(values.shape[1])])
    return np.bincount(groups, weights=values, minlength=len(count)) / count


def _majority(values: np.ndarray, groups: np.ndarray, size: int) -> np.ndarray:
    """Point majority; ties choose the lowest class for reproducible output."""
    pairs, frequency = np.unique(np.column_stack((groups, values)), axis=0, return_counts=True)
    order = np.lexsort((pairs[:, 1], -frequency, pairs[:, 0]))
    pairs = pairs[order]
    first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
    out = np.zeros(size, dtype=np.uint8)
    out[pairs[first, 0].astype(np.int64)] = pairs[first, 1]
    return out


