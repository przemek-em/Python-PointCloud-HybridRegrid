from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import json
import sys
import unittest

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud import Cancelled, Cloud
from cloud_io import read_cloud, write_cloud
from jobs import process_surface_file
from surface_fill import Parameters, fill_gaps


def plane_with_gap(wall=False):
    xyz = np.array([[x+.2, y+.2, .2] for x in range(-5, 6) for y in range(-5, 6) if not (abs(x) <= 1 and abs(y) <= 1)])
    return xyz[:, [2, 0, 1]] if wall else xyz


class SurfaceFillTests(unittest.TestCase):
    def test_horizontal_gap_and_original_records(self):
        xyz = plane_with_gap()
        cloud = Cloud(xyz, np.full_like(xyz, .4), np.full(len(xyz), 200), np.full(len(xyz), 7))
        result = fill_gaps(cloud, Parameters(1))
        self.assertGreater(result.added_count, 0)
        np.testing.assert_array_equal(result.cloud.xyz[:len(xyz)], xyz)
        np.testing.assert_array_equal(result.cloud.source_indices[:len(xyz)], np.arange(len(xyz)))
        new = slice(len(xyz), None)
        np.testing.assert_allclose(result.cloud.xyz[new, 2], .2)
        np.testing.assert_allclose(result.cloud.rgb[new], .4)
        np.testing.assert_allclose(result.cloud.intensity[new], 200)
        np.testing.assert_array_equal(result.cloud.classification[new], 7)
        self.assertTrue(result.cloud.generated[new].all())
        self.assertTrue((result.cloud.source_indices[new] == -1).all())

    def test_vertical_wall_gap(self):
        xyz = plane_with_gap(True)
        result = fill_gaps(Cloud(xyz), Parameters(1))
        self.assertGreater(result.added_count, 0)
        np.testing.assert_allclose(result.cloud.xyz[len(xyz):, 0], .2, atol=1e-10)

    def test_arbitrary_orientation_and_large_coordinates(self):
        xyz = plane_with_gap()
        axis = np.array([1, 2, 3], dtype=float)
        axis /= np.linalg.norm(axis)
        a = .8
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3)*np.cos(a) + (1-np.cos(a))*np.outer(axis, axis) + np.sin(a)*cross
        offset = np.array([650000, 5100000, 100])
        transformed = xyz @ rotation.T + offset
        result = fill_gaps(Cloud(transformed), Parameters(1, interpolation="plane"))
        self.assertGreater(result.added_count, 0)
        back = (result.cloud.xyz[len(xyz):]-offset) @ rotation
        np.testing.assert_allclose(back[:, 2], .2, atol=1e-8)
        np.testing.assert_array_equal(result.cloud.xyz[:len(xyz)], transformed)

    def test_curved_cave_wall_quadratic(self):
        xyz = np.array([[8*np.cos(t), 8*np.sin(t), z] for t in np.arange(-1, 1.001, .125) for z in np.arange(-5, 5.01, 1) if not (abs(t) <= .25 and abs(z) <= 1)])
        result = fill_gaps(Cloud(xyz), Parameters(1, interpolation="quadratic"))
        new = result.cloud.xyz[len(xyz):]
        self.assertGreater(len(new), 0)
        self.assertLess(np.max(np.abs(np.linalg.norm(new[:, :2], axis=1)-8)), .03)

    def test_nearby_sheets_do_not_become_middle_surface(self):
        xy = plane_with_gap()[:, :2]
        xyz = np.concatenate([np.column_stack((xy, np.full(len(xy), z))) for z in [.2, 2.2]])
        result = fill_gaps(Cloud(xyz), Parameters(1, layer_tolerance=.3))
        new = result.cloud.xyz[len(xyz):]
        self.assertGreater(len(new), 0)
        distance = np.minimum(np.abs(new[:, 2]-.2), np.abs(new[:, 2]-2.2))
        self.assertLess(distance.max(), 1e-9)
        self.assertTrue((new[:, 2] < 1).any())
        self.assertTrue((new[:, 2] > 2).any())

    def test_full_surface_does_not_grow_open_edges(self):
        xyz = np.array([[x+.2, y+.2, .2] for x in range(-5, 6) for y in range(-5, 6)])
        result = fill_gaps(Cloud(xyz), Parameters(1, passes=4))
        self.assertEqual(result.added_count, 0)
        np.testing.assert_array_equal(result.cloud.xyz, xyz)

    def test_maximum_distance_always_uses_original_support(self):
        xyz = np.array([[x+.2, y+.2, .2] for x in range(-8, 9) for y in range(-8, 9) if not (abs(x) <= 3 and abs(y) <= 3)])
        result = fill_gaps(Cloud(xyz), Parameters(1, max_gap_distance=1.1, passes=8))
        new = result.cloud.xyz[len(xyz):]
        if len(new):
            distance, _ = cKDTree(xyz).query(new)
            self.assertTrue((distance <= 1.1+1e-10).all())
            self.assertFalse((np.linalg.norm(new[:, :2]-.2, axis=1) < 1).any())

    def test_limits_and_cancellation(self):
        with self.assertRaisesRegex(ValueError, "exceed"):
            fill_gaps(Cloud(plane_with_gap()), Parameters(1, max_added_points=1))
        event = Event()
        event.set()
        with self.assertRaises(Cancelled):
            fill_gaps(Cloud(plane_with_gap()), Parameters(1), event)
        with self.assertRaises(ValueError):
            Parameters(spacing=float("nan")).validate()

    def test_marked_estimates_never_become_measured_support(self):
        xyz = plane_with_gap()
        with self.assertRaisesRegex(ValueError, "no unmarked measured"):
            fill_gaps(Cloud(xyz, generated=np.ones(len(xyz), dtype=bool)), Parameters(1))
        original = Cloud(xyz)
        first = fill_gaps(original, Parameters(1))
        again = fill_gaps(first.cloud, Parameters(1))
        self.assertEqual(again.support_points, first.support_points)
        self.assertEqual(again.added_count, 0)

    def test_new_points_respect_minimum_spacing(self):
        result = fill_gaps(Cloud(plane_with_gap()), Parameters(1))
        new = result.cloud.xyz[result.input_count:]
        distance, _ = cKDTree(new).query(new, k=2)
        self.assertTrue((distance[:, 1] >= .7-1e-10).all())

    def test_file_job_laz_markers_and_report(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output = folder/"wall.ply", folder/"wall_filled.laz"
            xyz = plane_with_gap(True)
            write_cloud(source, Cloud(xyz), 1)
            report = process_surface_file(source, output, Parameters(1), Event(), lambda f, m: None, lambda m: None)
            actual = read_cloud(output)
            self.assertGreater(report["added_points"], 0)
            np.testing.assert_allclose(actual.xyz[:len(xyz)], xyz, atol=.001)
            self.assertEqual(int(actual.generated.sum()), report["added_points"])
            saved = json.loads(Path(str(output)+".regrid.json").read_text())
            self.assertEqual(saved["mode"], "surface_fill_3d")
            self.assertEqual(saved["input_points"], len(xyz))


if __name__ == "__main__":
    unittest.main()
