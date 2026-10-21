"""Run the vendored multi-turn benchmark with its exit hang fixed. Same arguments.

At the end of main_mp() the benchmark drains its task queue with `while not q.empty(): q.get()`
and then calls `q.join_thread()`. empty() can return True while the queue's feeder thread
still holds unsent conversations, and join_thread() then waits forever for a pipe nobody
reads, before stats.json is written. Any run that leaves conversations unused (every run with
-n) can hit it. In the main process the queue's leftover data is discarded anyway, so
join_thread() is told not to wait there. Children are untouched, so no results are lost.
"""

import asyncio
import multiprocessing as mp
import multiprocessing.queues as mpq
import sys
from pathlib import Path

_join_thread = mpq.Queue.join_thread


def _join_thread_main_skips(self: mpq.Queue) -> None:
    if mp.parent_process() is None:
        self.cancel_join_thread()
    _join_thread(self)


mpq.Queue.join_thread = _join_thread_main_skips
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "multi_turn"))

import benchmark_serving_multi_turn as bench  # noqa: E402

if __name__ == "__main__":
    asyncio.run(bench.main())
