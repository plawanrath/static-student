"""Regime of every drift operator for the frozen legacy parser, measured one rewrite at a time.

    .venv/bin/python scripts/w02_operator_regimes.py [--instances 10] [--payloads 1000]    ->  results/w02_operator_regimes/

Each operator is applied to each step-0 source format `instances` times (different seeds and severities). For every
rewrite that applies, fresh payloads are rendered and the change in the legacy parser's no-match rate (payloads whose
gold is a struct) and silent-wrong rate (all payloads) is recorded. far: no-match up by >= 5 points; near:
silent-wrong up by >= 1 point; both; benign otherwise.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np

from static_student import legacy
from static_student.drift import heldout, operators
from static_student.tasks import telemetry as T

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/w02_operator_regimes"


def rates(fam: dict, n: int, seed: str) -> tuple[float, float]:
    rng = random.Random(seed)
    rows = [T.render(fam, rng) for _ in range(n)]
    o = legacy.outcomes(rows, legacy.parse_many([r.payload for r in rows]))
    struct = np.array([not r.defer for r in rows])
    return (float(1 - np.array(o["matched"])[struct].mean()) if struct.any() else 0.0), float(np.mean(o["wrong"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--payloads", type=int, default=1000)
    args = ap.parse_args()
    legacy.build()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = json.loads((REPO / "data/drift/sources_v0.json").read_text())["sources"]
    table = {**operators.OPERATORS, **heldout.OPERATORS}
    rows, summary = [], {}
    for op in sorted(table, key=lambda k: int(k[1:])):
        counts, sw = collections.Counter(), []
        for src in sources:
            for i in range(args.instances):
                rng = random.Random(f"{op}:{src['id']}:{i}")
                out = operators.apply(op, src, rng, rng.random(), operators=table)
                if out is None:
                    continue
                nm, w = rates(out, args.payloads, f"after:{op}:{src['id']}:{i}")
                far, near = nm >= 0.05, w >= 0.01  # step-0 sources have no-match 0 and silent-wrong 0 by the step-0 test
                regime = "both" if far and near else "far" if far else "near" if near else "benign"
                counts[regime] += 1
                sw.append(w)
                rows.append({"op": op, "source": src["id"], "instance": i, "no_match": round(nm, 4), "silent_wrong": round(w, 4), "regime": regime})
        n = sum(counts.values())
        summary[op] = {"instances": n, **dict(counts), "near_or_both_share": round((counts["near"] + counts["both"]) / n, 3) if n else None,
                       "mean_silent_wrong": round(float(np.mean(sw)), 4) if sw else None, "max_silent_wrong": round(float(np.max(sw)), 4) if sw else None}
        print(op, summary[op], flush=True)
    (OUT / "summary.json").write_text(json.dumps({"payloads_per_instance": args.payloads, "instances_per_source": args.instances, "operators": summary}, indent=1) + "\n")
    (OUT / "instances.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
