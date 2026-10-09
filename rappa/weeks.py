"""Running one function over many weeks at once, for either league's history (cfb.history, nfl.history).

Each week's fits are small, and numpy's threaded maths spends more time
coordinating its threads on them than working: a week of nfl.history's
score fits took 2.4 s on one thread and far longer on twenty. So every week
gets one thread, and the weeks are spread over worker processes instead.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from threadpoolctl import threadpool_limits

_FN = _ARGS = None


def _start(fn, args):
    global _FN, _ARGS
    _FN, _ARGS = fn, args
    threadpool_limits(1)


def _run(slot):
    return _FN(slot, *_ARGS)


def map_weeks(fn, slots, args=(), workers=None):
    """[fn(slot, *args) for slot in slots], over workers processes (default: every core) with one thread each.

    fn must be a module-level function, so the workers can find it; args go
    to each worker once rather than with every week.
    """
    slots = list(slots)
    workers = min(workers or os.cpu_count() or 1, len(slots))
    if workers <= 1:
        with threadpool_limits(1):
            return [fn(slot, *args) for slot in slots]
    # Forked, so the workers start with the caller's imports and sys.path (site/build.py runs by path).
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_start, initargs=(fn, args)) as pool:
        return list(pool.map(_run, slots))
