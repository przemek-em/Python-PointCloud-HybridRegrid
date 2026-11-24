"""Bounded CPU batches with stable result ordering and cooperative cancellation."""
from collections import deque
from concurrent.futures import Executor, TimeoutError
import os

from cloud import check_cancel


def resolve_workers(requested: int) -> int:
    available = max(1, os.cpu_count() or 1)
    return min(8, max(1, available-1)) if requested == 0 else min(requested, available)


def ordered_batches(function, count: int, batch_size: int, pool: Executor | None, workers: int, cancel):
    """Keep at most two batches per worker in flight; yield in input order.

    Waiting uses short timeouts so a cancellation is noticed even if the next
    batch is still in native numerical code. Running batches finish before
    their executor is closed; queued work is cancelled on errors or shutdown.
    """
    starts = iter(range(0, count, batch_size))
    if pool is None or count <= batch_size:
        for start in starts:
            check_cancel(cancel)
            result = function(start)
            check_cancel(cancel)
            yield start, result
        return
    pending = deque()
    try:
        for _ in range(2*workers):
            start = next(starts, None)
            if start is None:
                break
            check_cancel(cancel)
            pending.append((start, pool.submit(function, start)))
        while pending:
            check_cancel(cancel)
            start, future = pending[0]
            try:
                result = future.result(timeout=.05)
            except TimeoutError:
                # An operation may itself raise TimeoutError. Do not mistake
                # that completed failure for an unfinished future.
                if future.done():
                    result = future.result()
                else:
                    continue
            pending.popleft()
            check_cancel(cancel)
            yield start, result
            start = next(starts, None)
            if start is not None:
                check_cancel(cancel)
                pending.append((start, pool.submit(function, start)))
    finally:
        for _, future in pending:
            future.cancel()
