"""Families -> tensors. Pools are rendered in memory from families.jsonl, split by family, never by payload."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from static_student.student.model import CLS, MAX_LEN, PAD, UNIT_CLASSES
from static_student.tasks import telemetry as T


def load_families(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def split_families(families: list[dict], seed: int = 0, shares=(("train", 0.8), ("dev", 0.1), ("defer_fit", 0.1))) -> dict[str, list[dict]]:
    """Deterministic split by a hash of the family id: adding families never moves an existing one."""
    out: dict[str, list[dict]] = {name: [] for name, _ in shares}
    for f in families:
        u = int(hashlib.sha256(f"{seed}:{f['family_id']}".encode()).hexdigest()[:8], 16) / 2**32
        acc = 0.0
        for name, share in shares:
            acc += share
            if u < acc or name == shares[-1][0]:
                out[name].append(f)
                break
    return out


def encode(rows: list[T.Row]) -> dict[str, np.ndarray]:
    n = len(rows)
    ids = np.full((n, MAX_LEN), PAD, dtype=np.int64)
    ids[:, 0] = CLS
    ptr = np.zeros((n, 4), dtype=np.int64)       # 0 = absent
    unit = np.full(n, -100, dtype=np.int64)      # ignored by the loss when the row defers
    defer = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(rows):
        b = r.payload[: MAX_LEN - 1]
        ids[i, 1:1 + len(b)] = np.frombuffer(b, dtype=np.uint8)
        if r.defer:
            defer[i] = 1.0
        else:
            ptr[i] = (r.lat_span[0] + 1, r.lat_span[1], r.usr_span[0] + 1, r.usr_span[1])
            unit[i] = UNIT_CLASSES.index(r.unit)
    return {"ids": ids, "ptr": ptr, "unit": unit, "defer": defer}


def render_rows(families: list[dict], n: int, seed: str) -> list[T.Row]:
    rng = random.Random(seed)
    return [T.render(families[i % len(families)], rng) for i in range(n)]


def decode(payload: bytes, ptr: np.ndarray, unit: int) -> tuple[int, str] | None:
    """The deterministic struct conversion the generated C performs: spans and unit class -> (latency_us, user_id)."""
    ls, le, us, ue = (int(x) for x in ptr)
    if min(ls, le, us, ue) == 0 or le < ls or ue < us or len(payload) > T.MAX_PAYLOAD:
        return None
    lat = T.to_micros(payload[ls - 1:le], UNIT_CLASSES[unit])
    uid = payload[us - 1:ue]
    if lat is None or not uid or len(uid) > T.USER_ID_MAX or not T._ID_RE.match(uid):
        return None
    return lat, uid.decode("ascii")
