from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock, current_thread, enumerate as threads
from unittest.mock import patch
import sys
import time
import unittest

import numpy as np
from threadpoolctl import threadpool_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud import Cancelled, Cloud
from cpu_workers import ordered_batches, resolve_workers
import surface_fill
from surface_fill import Parameters, fill_gaps


def many_gaps():
    xyz = np.array([[x+.2, y+.2, .2] for x in range(121) for y in range(121)
                    if not (x % 9 in (3, 4, 5) and y % 9 in (3, 4, 5))])
    rgb = np.column_stack((xyz[:, 0]/122, xyz[:, 1]/122, np.full(len(xyz), .5)))
    return Cloud(xyz, rgb, xyz[:, 0]+xyz[:, 1], (np.floor(xyz[:, 0]/9) % 4).astype(np.uint8))


class CpuWorkerTests(unittest.TestCase):
    def test_config_validation_and_resolution(self):
        for value in (-1, 33, 1.5, "4"):
            with self.assertRaises(ValueError):
                Parameters(workers=value).validate()
        with patch("cpu_workers.os.cpu_count", return_value=32):
            self.assertEqual(resolve_workers(0), 8)
            self.assertEqual(resolve_workers(4), 4)
        with patch("cpu_workers.os.cpu_count", return_value=2):
            self.assertEqual(resolve_workers(0), 1)
            self.assertEqual(resolve_workers(32), 2)
        with patch("cpu_workers.os.cpu_count", return_value=None):
            self.assertEqual(resolve_workers(0), 1)

    def test_parallel_batches_yield_in_input_order(self):
        def run(start):
            time.sleep(.02 if start == 0 else .001)
            return start*2
        with ThreadPoolExecutor(max_workers=3) as pool:
            actual = list(ordered_batches(run, 12, 1, pool, 3, Event()))
        self.assertEqual(actual, [(i, 2*i) for i in range(12)])

    def test_completed_worker_timeout_is_propagated(self):
        def run(start):
            raise TimeoutError("failure inside worker")
        with ThreadPoolExecutor(max_workers=2) as pool:
            with self.assertRaisesRegex(TimeoutError, "failure inside worker"):
                list(ordered_batches(run, 3, 1, pool, 2, Event()))

    def test_parallel_output_matches_single_worker(self):
        cloud = many_gaps()
        serial = fill_gaps(cloud, Parameters(1, workers=1))
        parallel = fill_gaps(cloud, Parameters(1, workers=4))
        self.assertGreater(serial.tested_candidates, 1024)
        self.assertGreater(serial.added_count, 0)
        self.assertEqual(serial.added_count, parallel.added_count)
        self.assertEqual(serial.passes, parallel.passes)
        self.assertEqual(serial.reliable_normals, parallel.reliable_normals)
        for name in ("xyz", "rgb", "intensity", "classification", "generated", "source_indices"):
            np.testing.assert_array_equal(getattr(serial.cloud, name), getattr(parallel.cloud, name))
        self.assertFalse(any(t.name.startswith("surface-fill") for t in threads()))

    def test_normal_batches_really_execute_concurrently(self):
        if resolve_workers(2) < 2:
            self.skipTest("Requires two available CPUs")
        barrier = Barrier(2, timeout=5)
        observed, lock = set(), Lock()
        original = surface_fill._normal_batch
        def run(*args):
            start = args[3]
            if start in (0, 1024):
                with lock:
                    observed.add(current_thread().name)
                barrier.wait()
            return original(*args)
        with patch("surface_fill._normal_batch", side_effect=run):
            fill_gaps(many_gaps(), Parameters(1, workers=2, passes=1))
        self.assertEqual(len(observed), 2)

    def test_cancellation_stops_pending_batches(self):
        cancel = Event()
        cloud = many_gaps()
        before = cloud.xyz.copy()
        def progress(fraction, message):
            if "Estimating local" in message:
                cancel.set()
        with self.assertRaises(Cancelled):
            fill_gaps(cloud, Parameters(1, workers=4), cancel, progress)
        np.testing.assert_array_equal(cloud.xyz, before)
        self.assertFalse(any(t.name.startswith("surface-fill") for t in threads()))

    def test_worker_error_restores_blas_and_joins_threads(self):
        original = surface_fill._normal_batch
        before = {item["filepath"]: item["num_threads"] for item in threadpool_info() if item["user_api"] == "blas"}
        def run(*args):
            if args[3] == 1024:
                raise ValueError("deliberate batch failure")
            return original(*args)
        with patch("surface_fill._normal_batch", side_effect=run):
            with self.assertRaisesRegex(ValueError, "deliberate batch failure"):
                fill_gaps(many_gaps(), Parameters(1, workers=4))
        after = {item["filepath"]: item["num_threads"] for item in threadpool_info() if item["user_api"] == "blas"}
        self.assertEqual(before, after)
        self.assertFalse(any(t.name.startswith("surface-fill") for t in threads()))


if __name__ == "__main__":
    unittest.main()
