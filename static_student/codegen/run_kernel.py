"""Run payloads through a compiled kernel, sharded across cores.

The kernel is a single-threaded scalar C program, so scoring a 20,000-payload calibration pool takes about six minutes
on one core. The work is embarrassingly parallel across payloads, and the result is independent of the split because a
payload's output depends only on itself. `score(exe, payloads, jobs)` shards, runs, and reassembles in input order.
"""
from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def default_jobs() -> int:
    return max(1, min(12, (os.cpu_count() or 4) - 2))


def _one(exe: str, payloads: list[bytes]) -> list[list[str]]:
    r = subprocess.run([exe], input=b"".join(p + b"\n" for p in payloads), capture_output=True, check=True)
    rows = [l.split() for l in r.stdout.decode().splitlines()]
    if len(rows) != len(payloads):
        raise RuntimeError(f"kernel returned {len(rows)} results for {len(payloads)} payloads")
    return rows


def score(exe, payloads: list[bytes], jobs: int | None = None) -> dict[str, np.ndarray]:
    jobs = jobs or default_jobs()
    n = len(payloads)
    if n == 0:
        raise ValueError("no payloads")
    jobs = max(1, min(jobs, n))
    bounds = [round(i * n / jobs) for i in range(jobs + 1)]
    shards = [payloads[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:          # the work is in the child processes, not in Python
        rows = [r for part in pool.map(lambda s: _one(str(exe), s), shards) for r in part]
    return {"ptr": np.array([[int(x) for x in r[:4]] for r in rows], dtype=np.int64),
            "unit": np.array([int(r[4]) for r in rows], dtype=np.int64),
            "score": np.array([float(r[5]) for r in rows], dtype=np.float32),
            "defer": np.array([float(r[6]) for r in rows], dtype=np.float32)}
