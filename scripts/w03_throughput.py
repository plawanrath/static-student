"""Q2: per-payload latency and single-core throughput of the hybrid binary against the legacy parser alone.

    .venv/bin/python scripts/w03_throughput.py --model S_w4 --n 200000     ->  results/w03_throughput/

The legacy parser runs first, so the student only costs anything on payloads the legacy parser rejects. The whole
shape of the result is therefore a function of the no-match rate, and that is what is swept: pools are built by mixing
undrifted payloads (which the legacy parser matches) with drifted ones (which it does not) in fixed proportions.
Both orders are measured, and the audit probability is swept separately because it puts the student on the hot path
for payloads the legacy parser already handled.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from static_student import stats
from static_student.drift import operators, timeline
from static_student.student import data

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/w03_throughput"


def pools(n: int, rate: float, seed: int = 0) -> list[bytes]:
    """A pool in which `rate` of the payloads come from families drifted far enough that the frozen parser fails."""
    rng = random.Random(f"pool:{seed}:{rate}")
    src = json.loads((REPO / "data/drift/sources_v0.json").read_text())["sources"]
    clean = data.render_rows(src, int(round(n * (1 - rate))), f"clean:{seed}") if rate < 1 else []
    drifted_fams = []
    while len(drifted_fams) < max(1, len(src)):
        f = rng.choice(src)
        for _ in range(3):
            f = operators.apply(rng.choice(["D1", "D2", "D6"]), f, rng, 0.9) or f
        if f.get("trace"):
            drifted_fams.append(f)
    drifted = data.render_rows(drifted_fams, n - len(clean), f"drift:{seed}") if rate > 0 else []
    rows = [r.payload for r in clean] + [r.payload for r in drifted]
    rng.shuffle(rows)
    return rows


def run(cmd: list[str], payloads: list[bytes]) -> tuple[np.ndarray, dict]:
    r = subprocess.run(cmd, input=b"".join(p + b"\n" for p in payloads), capture_output=True, check=True)
    us = np.array([float(x) for x in r.stdout.split()])
    info = dict(kv.split("=") for kv in r.stderr.decode().split() if "=" in kv)
    return us, info


def summarize(us: np.ndarray, info: dict) -> dict:
    return {"n": len(us), "p50_us": float(np.median(us)), "p99_us": float(np.percentile(us, 99)),
            "mean_us": float(us.mean()), "throughput_per_s": float(info.get("throughput_per_s", 0)),
            "counters": {k: int(v) for k, v in info.items() if k not in ("wall_s", "throughput_per_s")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="S_w4")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--rates", type=float, nargs="*", default=[0.0, 0.01, 0.10, 0.50])
    ap.add_argument("--audit", type=float, nargs="*", default=[0.0, 0.001, 0.01])
    ap.add_argument("--queue", type=int, default=4096, help="asynchronous audit queue capacity (ADR-0010)")
    ap.add_argument("--budgets", type=float, nargs="*", default=[1.0, 0.01, 0.001], help="share of payloads the student may run on")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hybrid = REPO / "build" / f"tel_{args.model}" / "ss_hybrid"
    legacy = REPO / "build" / "legacy_bench"
    if not hybrid.exists():
        raise SystemExit(f"{hybrid} not built")
    res = {"model": args.model, "n_per_pool": args.n, "by_no_match_rate": {}, "by_audit_probability": {}}
    for rate in args.rates:
        pool = pools(args.n, rate)
        lus, linfo = run([str(legacy)], pool)
        hus, hinfo = run([str(hybrid), "--bench"], pool)
        sus, sinfo = run([str(hybrid), "--bench", "--student-first"], pool)
        cell = {"legacy_only": summarize(lus, linfo), "hybrid_legacy_first": summarize(hus, hinfo), "student_first": summarize(sus, sinfo)}
        cell["hybrid_throughput_fraction_of_legacy"] = cell["hybrid_legacy_first"]["throughput_per_s"] / cell["legacy_only"]["throughput_per_s"]
        cell["ratio_of_medians_hybrid_over_legacy"] = stats.ratio_of_medians_ci(hus, lus).as_dict()
        res["by_no_match_rate"][f"{rate:.2f}"] = cell
        print(f"no-match {rate:>5.0%}  legacy {cell['legacy_only']['throughput_per_s']:>10,.0f}/s  hybrid {cell['hybrid_legacy_first']['throughput_per_s']:>10,.0f}/s"
              f"  ({cell['hybrid_throughput_fraction_of_legacy']:>6.1%} of legacy)  student-first {cell['student_first']['throughput_per_s']:>9,.0f}/s"
              f"  p50 {cell['hybrid_legacy_first']['p50_us']:.2f} us  p99 {cell['hybrid_legacy_first']['p99_us']:.1f} us", flush=True)
    pool = pools(args.n, 0.0)
    for p in args.audit:
        for mode, extra in (("synchronous", []), ("asynchronous", ["--audit-async", str(args.queue)])):
            hus, hinfo = run([str(hybrid), "--bench", "--audit", str(p), *extra], pool)
            cell = summarize(hus, hinfo)
            res["by_audit_probability"][f"{p}:{mode}"] = cell
            c = cell["counters"]
            print(f"audit p={p:<6} {mode:<13} throughput {cell['throughput_per_s']:>10,.0f}/s  p50 {cell['p50_us']:.2f} us  p99 {cell['p99_us']:.1f} us"
                  f"  audited {c.get('audited', 0):>6}  dropped {c.get('audit_dropped', 0):>6}", flush=True)
    res["by_budget"] = {}
    drift = pools(args.n, 0.10)
    for b in args.budgets:
        hus, hinfo = run([str(hybrid), "--bench", "--budget", str(b)], drift)
        cell = summarize(hus, hinfo)
        res["by_budget"][f"{b}"] = cell
        c = cell["counters"]
        print(f"budget {b:<7} (10% no-match)  throughput {cell['throughput_per_s']:>10,.0f}/s  p99 {cell['p99_us']:>9.1f} us"
              f"  rescued {c.get('student_accepted', 0):>6}  skipped {c.get('budget_skipped', 0):>6}", flush=True)
    (OUT / f"summary_{args.model}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
