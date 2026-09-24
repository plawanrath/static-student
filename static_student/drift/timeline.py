"""Drift timelines: a deployment of several sources whose formats are rewritten by operators arriving over time.

A timeline is fully determined by its config (YAML under data/drift/): seed, track, number of steps and sources, the
Poisson arrival rate per source and step, and the payload count. `sample` returns, per step, the family of every
source plus the event log; `render_step` expands one step into labelled payloads.

Tracks: in_curriculum (D1-D8, D13, D14, in-curriculum parameters), heldout_params (D1-D7 with unseen parameter values),
heldout_operators (D9-D12).
"""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import yaml

from static_student.drift import heldout, operators
from static_student.tasks import telemetry as T

REPO = Path(__file__).resolve().parents[2]
TRACKS = {
    "in_curriculum": (list(operators.OPERATORS), None, operators.OPERATORS),
    "heldout_params": (["D1", "D2", "D3", "D4", "D5", "D6", "D7"], heldout.PARAMS, operators.OPERATORS),
    "heldout_operators": (list(heldout.OPERATORS), None, heldout.OPERATORS),
    "none": ([], None, {}),  # drift-free control, for false-alarm rates
}
DEFAULTS = {"steps": 27, "sources": 8, "rate": 0.1, "payloads_per_source": 2000, "sources_file": "data/drift/sources_v0.json",
            "only_operators": None}


def load_config(path: str | Path) -> dict:
    cfg = {**DEFAULTS, **yaml.safe_load(Path(path).read_text())}
    if cfg["track"] not in TRACKS:
        raise ValueError(f"track {cfg['track']!r}")
    return cfg


def _poisson(rng: random.Random, lam: float) -> int:
    k, p, limit = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def sample(cfg: dict, keep_families: bool = False) -> tuple[list[list[dict]], list[dict]]:
    """(families[step][source], events). Step 0 is the undrifted deployment. With keep_families, every applied event
    also carries the family before and after the rewrite (used to measure the regime the rewrite produced)."""
    rng = random.Random(f"timeline:{cfg['seed']}")
    pool = json.loads((REPO / cfg["sources_file"]).read_text())["sources"]
    current = [copy.deepcopy(f) for f in rng.sample(pool, cfg["sources"])]
    op_ids, params, table = TRACKS[cfg["track"]]
    op_ids = [o for o in op_ids if not cfg.get("only_operators") or o in cfg["only_operators"]]
    severity = [0.0] * len(current)
    steps, events = [copy.deepcopy(current)], []
    for t in range(1, cfg["steps"]):
        for i in range(len(current)):
            for _ in range(_poisson(rng, cfg["rate"]) if op_ids else 0):
                severity[i] = min(1.0, severity[i] + rng.uniform(0.1, 0.3))
                op = rng.choice(op_ids)
                out = operators.apply(op, current[i], rng, severity[i], params=params, operators=table)
                events.append({"step": t, "source": i, "source_id": current[i]["id"], "op": op,
                               "severity": round(severity[i], 3), "applied": out is not None})
                if out is not None:
                    if keep_families:
                        events[-1].update(before=current[i], after=out)
                    current[i] = out
        steps.append(copy.deepcopy(current))
    return steps, events


def render_step(cfg: dict, families: list[dict], step: int) -> list[tuple[int, T.Row]]:
    """[(source index, row)] for one step; independent of every other step's randomness."""
    out = []
    for i, fam in enumerate(families):
        rng = random.Random(f"render:{cfg['seed']}:{step}:{i}")
        out += [(i, T.render(fam, rng)) for _ in range(cfg["payloads_per_source"])]
    return out
