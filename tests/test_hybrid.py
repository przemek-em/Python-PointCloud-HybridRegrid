from itertools import product
from pathlib import Path
from threading import Event
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud import Cancelled, Cloud
from hybrid import Parameters, _classify, _height_slopes, regrid


def reference(xyz, p):
    """Small independent dictionary implementation of the actual C++ geometry."""
    cell, voxel = p.cell_size, p.voxel_size or p.cell_size
    columns, voxels = {}, {}
    for point in xyz:
        columns.setdefault(tuple(np.floor(point[:2] / cell).astype(int)), []).append(point)
        voxels.setdefault(tuple(np.floor(point / voxel).astype(int)), []).append(point)
    hf = {}
    xy = {}
    for key, points in columns.items():
        values = sorted(point[2] for point in points)
        hf[key] = {"median": values[len(values)//2], "mean": np.mean(values), "max": max(values), "min": min(values)}[p.height_stat]
        xy[key] = np.mean(points, axis=0)[:2]
    neighbours = [v for v in product((-1, 0, 1), repeat=2) if v != (0, 0)]
    for _ in range(p.fill_passes):
        candidates = set()
        for x, y in hf:
            candidates.update([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        added = {}
        for x, y in candidates - hf.keys():
            if not (((x-1, y) in hf and (x+1, y) in hf) or ((x, y-1) in hf and (x, y+1) in hf)):
                continue
            added[x, y] = np.mean([hf[x+dx, y+dy] for dx, dy in neighbours if (x+dx, y+dy) in hf])
            xy[x, y] = (np.array([x, y]) + .5) * cell
        hf.update(added)
    slopes = {}
    for (x, y), z in hf.items():
        differences = [abs(hf[x+dx, y+dy] - z) / (cell*np.hypot(dx, dy)) for dx, dy in neighbours if (x+dx, y+dy) in hf]
        slopes[x, y] = np.degrees(np.arctan(max(differences, default=0)))
    out = [np.r_[xy[key], z] for key, z in hf.items()]
    walls = caves = fringe = 0
    for points in voxels.values():
        center = np.mean(points, axis=0)
        key = tuple(np.floor(center[:2] / cell).astype(int))
        if key not in hf:
            fringe += 1
        elif slopes[key] >= p.wall_slope_deg:
            walls += 1
        elif p.keep_below_surface and hf[key] - center[2] > p.below_delta_mul * cell:
            caves += 1
        else:
            continue
        out.append(center)
    return np.array(out), (walls, caves, fringe)


def sorted_xyz(xyz):
    return xyz[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))]


class HybridTests(unittest.TestCase):
    def test_all_height_statistics_and_upper_median(self):
        cloud = Cloud([[.1, .2, 0], [.2, .3, 2], [.3, .4, 8], [.4, .5, 10]])
        for statistic, expected in [("median", 8), ("mean", 5), ("min", 0), ("max", 10)]:
            with self.subTest(statistic=statistic):
                result = regrid(cloud, Parameters(1, height_stat=statistic, fill_passes=0, keep_below_surface=False))
                self.assertEqual(len(result.cloud.xyz), 1)
                np.testing.assert_allclose(result.cloud.xyz[0], [.25, .35, expected])

    def test_flat_ground_uses_heightfield(self):
        xyz = [[x+.2+dx, y+.2+dy, 5] for x in range(4) for y in range(4) for dx in [0, .2] for dy in [0, .2]]
        result = regrid(Cloud(xyz), Parameters(1, voxel_size=.1, fill_passes=0))
        self.assertEqual(len(result.cloud.xyz), 16)
        self.assertEqual(result.walls_kept + result.caves_kept + result.fringe_kept, 0)

    def test_steep_ramp_retains_voxels(self):
        xyz = [[x+.2, y+.2, 2*x+.2] for x in range(4) for y in range(3)]
        result = regrid(Cloud(xyz), Parameters(1, fill_passes=0))
        self.assertEqual(result.walls_kept, len(xyz))
        self.assertEqual(len(result.cloud.xyz), len(xyz)*2)  # Same append semantics as editor.

    def test_cave_switch_and_depth_threshold(self):
        xyz = [[x+.2, y+.2, z] for x in range(3) for y in range(3) for z in [0.2, 10.2]]
        result = regrid(Cloud(xyz), Parameters(1, fill_passes=0))
        self.assertEqual(result.caves_kept, 9)
        self.assertEqual(len(result.cloud.xyz), 18)
        disabled = regrid(Cloud(xyz), Parameters(1, fill_passes=0, keep_below_surface=False))
        self.assertEqual(len(disabled.cloud.xyz), 9)
        shallow = regrid(Cloud(xyz), Parameters(1, fill_passes=0, below_delta_mul=10))
        self.assertEqual(shallow.caves_kept, 0)  # C++ uses strictly greater than.

    def test_isolated_vertical_wall_limitation(self):
        cloud = Cloud([[.2, .2, z+.2] for z in range(5)])
        result = regrid(cloud, Parameters(1, fill_passes=0, keep_below_surface=False))
        self.assertEqual(len(result.cloud.xyz), 1)
        self.assertEqual(result.walls_kept, 0)

    def test_gap_fill_at_xy_center_and_8_neighbour_mean(self):
        xyz = np.array([[x+.1, y+.2, x+y] for x in range(3) for y in range(3) if (x, y) != (1, 1)])
        cloud = Cloud(xyz, np.full_like(xyz, .4), intensity=np.full(len(xyz), 50), classification=np.full(len(xyz), 7))
        result = regrid(cloud, Parameters(1, wall_slope_deg=89, keep_below_surface=False))
        new = result.cloud.generated
        self.assertEqual(result.added_count, 1)
        np.testing.assert_allclose(result.cloud.xyz[new], [[1.5, 1.5, 2]])
        np.testing.assert_allclose(result.cloud.rgb[new], .4)
        np.testing.assert_allclose(result.cloud.intensity[new], 50)
        self.assertEqual(result.cloud.classification[new][0], 0)
        self.assertEqual(result.cloud.source_indices[new][0], -1)

    def test_multiple_synchronous_passes_propagate_colour(self):
        # An L-shaped three-cell gap: two arms fill first, then the corner.
        missing = {(2, 2), (3, 2), (2, 3)}
        xyz = [[x+.2, y+.2, 3] for x in range(5) for y in range(5) if (x, y) not in missing]
        cloud = Cloud(xyz, np.full((len(xyz), 3), .75), intensity=np.full(len(xyz), 100))
        result = regrid(cloud, Parameters(1, fill_passes=8))
        self.assertGreater(result.passes, 1)
        self.assertGreater(result.added_count, 0)
        np.testing.assert_allclose(result.cloud.rgb[result.cloud.generated], .75)
        np.testing.assert_allclose(result.cloud.intensity[result.cloud.generated], 100)
        expected, _ = reference(np.array(xyz), Parameters(1, fill_passes=8))
        np.testing.assert_allclose(sorted_xyz(result.cloud.xyz), sorted_xyz(expected))

    def test_separate_voxel_size_for_cave_detail(self):
        xyz = [[x+.2+dx, .2, z] for x in range(3) for dx in [0, .3] for z in [.2, 10.2]]
        coarse = regrid(Cloud(xyz), Parameters(1, fill_passes=0))
        fine = regrid(Cloud(xyz), Parameters(1, voxel_size=.2, fill_passes=0))
        self.assertEqual(coarse.caves_kept, 3)
        self.assertEqual(fine.caves_kept, 6)

    def test_slope_includes_diagonals_and_single_neighbour(self):
        keys = np.array([[0, 0], [1, 1]], dtype=np.int64)
        slopes = _height_slopes(keys, np.array([0, 2]), 1, None)
        np.testing.assert_allclose(slopes, np.degrees(np.arctan(2/np.sqrt(2))))

    def test_fringe_classification(self):
        wall, cave, fringe = _classify(np.array([[3.2, 3.2, 0]]), np.array([[0, 0]], dtype=np.int64), np.array([1]), np.array([0]), np.array([0, 0]), 1, Parameters(1))
        self.assertTrue(fringe[0])
        self.assertFalse(wall[0] or cave[0])

    def test_limits_and_cancellation(self):
        xyz = [[x+.2, y+.2, 0] for x in range(5) for y in range(5) if x in [0, 4] or y in [0, 4] or x == y]
        with self.assertRaisesRegex(ValueError, "exceed"):
            regrid(Cloud(xyz), Parameters(1, fill_passes=8, max_added_points=1))
        event = Event()
        event.set()
        with self.assertRaises(Cancelled):
            regrid(Cloud(xyz), Parameters(1), event)
        for params in [Parameters(0), Parameters(height_stat="bad"), Parameters(wall_slope_deg=float("nan")), Parameters(below_delta_mul=-1)]:
            with self.assertRaises(ValueError):
                params.validate()

    def test_matches_independent_cpp_reference(self):
        rng = np.random.default_rng(123)
        xyz = np.round(rng.uniform([-4, -4, -3], [5, 5, 12], (350, 3)), 3) + .0001
        for statistic in ["median", "mean", "min", "max"]:
            for fill in [0, 2]:
                with self.subTest(statistic=statistic, fill=fill):
                    params = Parameters(1, voxel_size=.5, height_stat=statistic, fill_passes=fill, wall_slope_deg=50, below_delta_mul=1.2)
                    expected, counts = reference(xyz, params)
                    actual = regrid(Cloud(xyz), params)
                    np.testing.assert_allclose(sorted_xyz(actual.cloud.xyz), sorted_xyz(expected), atol=1e-12)
                    self.assertEqual((actual.walls_kept, actual.caves_kept, actual.fringe_kept), counts)


if __name__ == "__main__":
    unittest.main()
