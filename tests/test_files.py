from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud import Cancelled, Cloud
from hybrid import Parameters, regrid
from cloud_io import read_cloud, write_cloud
from jobs import process_file


class FileTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def _las(self):
        import laspy
        from laspy.vlrs.known import WktCoordinateSystemVlr
        header = laspy.LasHeader(point_format=7, version="1.4")
        header.scales = [.0001] * 3
        header.offsets = [650000, 5100000, 100]
        header.vlrs.append(WktCoordinateSystemVlr('LOCAL_CS["test coordinates",UNIT["metre",1]]'))
        header.add_extra_dim(laspy.ExtraBytesParams(name="quality", type=np.uint16))
        data = laspy.LasData(header)
        data.x, data.y, data.z = np.array([[650000.2, 5100000.2, 100.2], [650000.4, 5100000.4, 100.4], [650002.2, 5100000.2, 100.2]]).T
        data.red = [10000, 30000, 50000]
        data.green = [20000, 40000, 60000]
        data.blue = [0, 0, 65535]
        data.intensity = [10, 30, 50]
        data.classification = [2, 2, 7]
        data.gps_time = [11, 12, 13]
        data.quality = [9, 8, 7]
        data.write(self.path / "source.las")
        return self.path / "source.las"

    def test_las_and_laz_preserve_coordinates_metadata_and_markers(self):
        import laspy
        cloud = read_cloud(self._las())
        result = regrid(cloud, Parameters(1, fill_passes=1))
        for suffix in [".las", ".laz"]:
            path = self.path / ("result" + suffix)
            write_cloud(path, result.cloud, 1)
            loaded = read_cloud(path)
            np.testing.assert_allclose(loaded.xyz, result.cloud.xyz, atol=.0001)
            np.testing.assert_allclose(loaded.rgb, result.cloud.rgb, atol=1/65535)
            np.testing.assert_array_equal(loaded.generated, result.cloud.generated)
            out = laspy.read(path)
            self.assertTrue(any(v.user_id == "LASF_Projection" for v in out.header.vlrs))
            added = np.asarray(out.regrid_generated).astype(bool)
            self.assertEqual(added.sum(), 1)
            self.assertEqual(np.asarray(out.synthetic)[added][0], 1)
            self.assertEqual(np.asarray(out.gps_time)[added][0], 0)
            self.assertEqual(np.asarray(out.quality)[added][0], 0)
            original = np.asarray(out.gps_time)[~added]
            self.assertTrue(set(original) <= {11, 12, 13})

    def test_ply_and_xyz_roundtrip(self):
        cloud = regrid(read_cloud(self._las()), Parameters(1, fill_passes=1)).cloud
        for suffix in [".ply", ".xyz", ".txt"]:
            path = self.path / ("out" + suffix)
            write_cloud(path, cloud, 1)
            out = read_cloud(path)
            np.testing.assert_allclose(out.xyz, cloud.xyz, atol=1e-10)
            np.testing.assert_allclose(out.rgb, cloud.rgb, atol=.5/255 + 1e-10)
            if suffix == ".ply":
                np.testing.assert_array_equal(out.generated, cloud.generated)
                np.testing.assert_array_equal(out.classification, cloud.classification)

    def test_ascii_ply_and_float_colours(self):
        path = self.path / "ascii.ply"
        path.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty double x\nproperty double y\nproperty double z\nproperty float red\nproperty float green\nproperty float blue\nend_header\n0 0 0 1 0 .5\n1 2 3 0 .5 1\n")
        cloud = read_cloud(path)
        np.testing.assert_allclose(cloud.rgb, [[1, 0, .5], [0, .5, 1]])

    def test_big_endian_ply_and_extended_class_las(self):
        path = self.path / "big.ply"
        header = b"ply\nformat binary_big_endian 1.0\nelement vertex 1\nproperty double x\nproperty double y\nproperty double z\nproperty uchar classification\nend_header\n"
        points = np.array([(650000.12345, 5100000.23456, 100.34567, 200)], dtype=[("x", ">f8"), ("y", ">f8"), ("z", ">f8"), ("classification", "u1")])
        path.write_bytes(header + points.tobytes())
        cloud = read_cloud(path)
        self.assertEqual(cloud.classification[0], 200)
        destination = self.path / "extended.las"
        write_cloud(destination, cloud, .01)
        out = read_cloud(destination)
        self.assertEqual(out.classification[0], 200)
        np.testing.assert_allclose(out.xyz, cloud.xyz, atol=1e-5)

    def test_truncated_ply_rejected(self):
        path = self.path / "truncated.ply"
        path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        with self.assertRaisesRegex(ValueError, "truncated"):
            read_cloud(path)

    def test_waveform_las_rejected(self):
        import laspy
        data = laspy.LasData(laspy.LasHeader(point_format=4, version="1.3"))
        data.x, data.y, data.z = [1], [2], [3]
        path = self.path / "waveform.las"
        data.write(path)
        with self.assertRaisesRegex(ValueError, "Waveform"):
            read_cloud(path)

    def test_copc_index_not_copied(self):
        import laspy
        from laspy.vlrs.vlr import VLR
        cloud = read_cloud(self._las())
        cloud.metadata["las_data"].header.vlrs.append(VLR(user_id="copc", record_id=42, record_data=b"obsolete index"))
        target = self.path / "ordinary.las"
        write_cloud(target, cloud, 1)
        self.assertFalse(any(v.user_id == "copc" for v in laspy.read(target).header.vlrs))

    def test_cancelled_write_preserves_existing_output(self):
        path = self.path / "old.xyz"
        path.write_bytes(b"original")
        event = Event()
        def progress(fraction):
            event.set()
        with self.assertRaises(Cancelled):
            write_cloud(path, Cloud([[1, 2, 3]]), 1, event, progress, overwrite=True)
        self.assertEqual(path.read_bytes(), b"original")
        self.assertFalse(list(self.path.glob(".voxel-regrid-*")))

    def test_no_overwrite_and_no_input_replacement(self):
        source = self._las()
        cloud = read_cloud(source)
        with self.assertRaises(ValueError):
            write_cloud(source, cloud, 1, overwrite=True)
        path = self.path / "exists.xyz"
        path.write_text("keep")
        with self.assertRaises(FileExistsError):
            write_cloud(path, cloud, 1)
        self.assertEqual(path.read_text(), "keep")

    def test_end_to_end_report(self):
        import json
        source = self._las()
        output = self.path / "processed.laz"
        report = process_file(source, output, Parameters(1, fill_passes=1), Event(), lambda f, m: None, lambda m: None)
        self.assertEqual(report["added_points"], 1)
        saved = json.loads(Path(str(output) + ".regrid.json").read_text())
        self.assertEqual(saved["output_points"], len(read_cloud(output).xyz))
        self.assertEqual(saved["resolved_voxel_size"], 1)

    def test_cancel_before_read(self):
        event = Event()
        event.set()
        with self.assertRaises(Cancelled):
            read_cloud(self.path / "absent.las", event)


if __name__ == "__main__":
    unittest.main()
