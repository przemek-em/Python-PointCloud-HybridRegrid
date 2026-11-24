from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, enumerate as threads
import json
import sys
import unittest

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud import Cancelled, Cloud
from cloud_io import read_cloud, write_cloud
from cpu_workers import resolve_workers
from jobs import process_rebuild_file
from uniform_rebuild import Parameters, _candidate_tiles, estimate_native_spacing, rebuild
from surface_fill import _candidate_cells


def plane_with_gap(large=False):
    size, hole = (30, 12) if large else (5, 1)
    return np.array([[x+.2, y+.2, .2] for x in range(-size, size+1) for y in range(-size, size+1) if not(abs(x) <= hole and abs(y) <= hole)])


class UniformRebuildTests(unittest.TestCase):
    def test_streamed_candidates_match_reference_across_tile_boundaries(self):
        rng = np.random.default_rng(17)
        keys = np.unique(rng.integers(-35, 36, size=(1600, 3)), axis=0)
        xyz = keys+.2
        normals = rng.normal(size=xyz.shape)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        expected = _candidate_cells(keys, xyz, normals, keys, np.zeros(3), 1, 100_000, None)
        chunks = list(_candidate_tiles(keys, xyz, normals, keys, np.zeros(3), 1, 100_000, None))
        streamed = np.concatenate([cells for cells, _, _ in chunks])
        self.assertEqual(len(streamed), len(np.unique(streamed, axis=0)))
        np.testing.assert_array_equal(np.unique(streamed, axis=0), expected)
        with self.assertRaisesRegex(ValueError, "candidate cells"):
            list(_candidate_tiles(keys, xyz, normals, keys, np.zeros(3), 1, 1, None))

    def test_off_resamples_into_exact_grid_and_averages(self):
        xyz = [[.1, .1, .1], [.3, .3, .3], [2.2, .2, .2]]
        cloud = Cloud(xyz, [[0, 0, 0], [1, 1, 1], [1, 0, 0]], [10, 30, 50], [2, 2, 7])
        result = rebuild(cloud, Parameters(1, completion="off"))
        np.testing.assert_array_equal(result.cloud.xyz, [[.5, .5, .5], [2.5, .5, .5]])
        np.testing.assert_allclose(result.cloud.rgb, [[.5, .5, .5], [1, 0, 0]])
        np.testing.assert_array_equal(result.cloud.intensity, [20, 50])
        np.testing.assert_array_equal(result.cloud.classification, [2, 7])
        self.assertEqual(result.added_count, 0)
        self.assertEqual(len(result.cloud.xyz), 2)

    def test_off_surface_positions_are_voxel_centroids(self):
        result = rebuild(Cloud([[.1, .2, .3], [.3, .4, .5]]), Parameters(1, positions="surface", completion="off"))
        np.testing.assert_allclose(result.cloud.xyz, [[.2, .3, .4]])

    def test_large_hole_fully_closes_including_centre(self):
        result = rebuild(Cloud(plane_with_gap(True)), Parameters(1, reach=18, workers=4))
        self.assertEqual(result.added_count, 625)
        self.assertGreater(result.rounds, 8)
        cells = np.floor((result.cloud.xyz-result.origin)/result.spacing).astype(np.int64)
        actual = {tuple(row) for row in cells}
        for x in range(-12, 13):
            for y in range(-12, 13):
                self.assertIn((x+30, y+30, 0), actual)
        np.testing.assert_allclose((result.cloud.xyz-result.origin)/result.spacing-.5, cells, atol=1e-12)
        self.assertEqual(len(np.unique(cells, axis=0)), len(cells))

    def test_finer_grid_independent_of_input_density_vertical_wall(self):
        xyz = plane_with_gap()[:, [2, 0, 1]]
        result = rebuild(Cloud(xyz), Parameters(.25, positions="surface", reach=4, workers=4))
        self.assertEqual(result.native_spacing, 1)
        self.assertGreater(len(result.cloud.xyz), 5*len(xyz))
        np.testing.assert_allclose(result.cloud.xyz[:, 0], .2, atol=1e-10)
        cells = np.floor((result.cloud.xyz-result.origin)/result.spacing).astype(np.int64)
        self.assertEqual(len(cells), len(np.unique(cells, axis=0)))

    def test_parallel_sheets_stay_separate(self):
        xy = plane_with_gap()[:, :2]
        xyz = np.concatenate([np.column_stack((xy, np.full(len(xy), z))) for z in [.2, 2.2]])
        result = rebuild(Cloud(xyz), Parameters(.5, positions="surface", completion="strong", reach=5, workers=4))
        self.assertGreater(result.added_count, 0)
        error = np.minimum(np.abs(result.cloud.xyz[:, 2]-.2), np.abs(result.cloud.xyz[:, 2]-2.2))
        self.assertLess(error.max(), 1e-8)

    def test_curved_surface_rebuild(self):
        xyz = np.array([[8*np.cos(t), 8*np.sin(t), z] for t in np.arange(-1, 1.001, .125) for z in np.arange(-5, 5.01, 1) if not(abs(t) <= .25 and abs(z) <= 1)])
        result = rebuild(Cloud(xyz), Parameters(.5, positions="surface", completion="strong", reach=4, workers=4))
        self.assertGreater(result.added_count, 0)
        error = np.abs(np.linalg.norm(result.cloud.xyz[:, :2], axis=1)-8)
        self.assertLess(np.quantile(error, .95), .15)

    def test_open_edges_do_not_expand_without_support(self):
        xyz = np.array([[x+.2, y+.2, .2] for x in range(10) for y in range(10)])
        result = rebuild(Cloud(xyz), Parameters(1, reach=6))
        self.assertEqual(result.added_count, 0)

    def test_reach_limits_surface_interpolation(self):
        xyz = plane_with_gap(True)
        result = rebuild(Cloud(xyz), Parameters(1, positions="surface", reach=2))
        added = result.cloud.xyz[result.cloud.generated]
        if len(added):
            distance, _ = cKDTree(xyz).query(added)
            self.assertTrue((distance <= 2+1e-10).all())
        self.assertLess(result.added_count, 625)

    def test_serial_parallel_equivalence(self):
        xyz = plane_with_gap(True)
        cloud = Cloud(xyz, np.full_like(xyz, .4), np.full(len(xyz), 100), np.full(len(xyz), 7))
        a = rebuild(cloud, Parameters(.5, reach=18, workers=1))
        b = rebuild(cloud, Parameters(.5, reach=18, workers=4))
        self.assertGreater(a.tested_candidates, 512)
        for name in ("xyz", "rgb", "intensity", "classification", "generated", "source_indices"):
            np.testing.assert_array_equal(getattr(a.cloud, name), getattr(b.cloud, name))
        self.assertFalse(any(t.name.startswith("uniform-rebuild") for t in threads()))

    def test_limits_cancel_and_invalid(self):
        cloud = Cloud(plane_with_gap())
        with self.assertRaisesRegex(ValueError, "output limit"):
            rebuild(cloud, Parameters(1, max_output_points=10))
        with self.assertRaisesRegex(ValueError, "exceed"):
            rebuild(cloud, Parameters(1, max_output_points=115))
        event = Event()
        def progress(fraction, message):
            if "Estimating local" in message:
                event.set()
        before = cloud.xyz.copy()
        with self.assertRaises(Cancelled):
            rebuild(cloud, Parameters(1, workers=4), event, progress)
        np.testing.assert_array_equal(cloud.xyz, before)
        self.assertFalse(any(t.name.startswith("uniform-rebuild") for t in threads()))
        for p in [Parameters(0), Parameters(completion="bad"), Parameters(positions="bad"), Parameters(workers=50)]:
            with self.assertRaises(ValueError):
                p.validate()

    def test_native_spacing_estimate_is_not_output_resolution(self):
        self.assertEqual(estimate_native_spacing(plane_with_gap()), 1)

    def test_file_job_rebuilds_and_reports(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output = folder/"source.ply", folder/"rebuilt.laz"
            write_cloud(source, Cloud(plane_with_gap()), 1)
            report = process_rebuild_file(source, output, Parameters(.5, reach=4, workers=2), Event(), lambda f, m: None, lambda m: None)
            actual = read_cloud(output)
            self.assertEqual(len(actual.xyz), report["output_points"])
            self.assertGreater(report["added_points"], 0)
            saved = json.loads(Path(str(output)+".regrid.json").read_text())
            self.assertEqual(saved["mode"], "uniform_rebuild_3d")
            self.assertEqual(saved["parameters"]["positions"], "grid")
            self.assertEqual(saved["worker_limit"], resolve_workers(2))
            local = (actual.xyz-np.array(saved["grid_origin"]))/saved["resolved_spacing"]-.5
            np.testing.assert_allclose(local, np.rint(local), atol=.002)


if __name__ == "__main__":
    unittest.main()
