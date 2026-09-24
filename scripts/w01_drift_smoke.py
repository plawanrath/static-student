"""W1 legacy-only drift smoke: the frozen legacy parser alone on drift timelines, per track.

    .venv/bin/python scripts/w01_drift_smoke.py [--seeds 5] [--payloads 500]     ->  results/w01_drift_smoke/

Reports, per track and step: no-match rate on payloads whose gold is a struct (loud failure), silent-wrong rate on
all payloads (the parser returned a struct that is not the gold struct, or returned one where gold defers), and yield.
Every applied operator instance is tagged with the regime it produced for the frozen parser, measured on fresh
payloads from the family before and after the rewrite: far (no-match up), near (silent-wrong up), both, or benign.
Rewrites of a source the parser had already lost (no-match >= 95% before the rewrite) are tagged already_broken.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
import yaml

from static_student import legacy, stats
from static_student.drift import timeline
from static_student.tasks import telemetry as T

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/w01_drift_smoke"
FAR_DELTA, NEAR_DELTA = 0.05, 0.01  # regime thresholds on the change in no-match / silent-wrong rate


def rates(rows: list[T.Row]) -> dict:
    o = legacy.outcomes(rows, legacy.parse_many([r.payload for r in rows]))
    struct = np.array([not r.defer for r in rows])
    matched, wrong, correct = (np.array(o[k]) for k in ("matched", "wrong", "correct"))
    return {"struct": struct, "no_match": (1 - matched)[struct], "silent_wrong": wrong, "yield": correct}


def regime(ev: dict, n: int) -> str:
    rng = random.Random(f"regime:{ev['step']}:{ev['source']}:{ev['op']}")
    before, after = (rates([T.render(ev[k], rng) for _ in range(n)]) for k in ("before", "after"))
    d_nm = after["no_match"].mean() - before["no_match"].mean() if len(after["no_match"]) and len(before["no_match"]) else 0.0
    d_sw = after["silent_wrong"].mean() - before["silent_wrong"].mean()
    ev.update(d_no_match=round(float(d_nm), 4), d_silent_wrong=round(float(d_sw), 4))
    if len(before["no_match"]) and before["no_match"].mean() >= 0.95:
        return "already_broken"
    far, near = d_nm >= FAR_DELTA, d_sw >= NEAR_DELTA
    return "both" if far and near else "far" if far else "near" if near else "benign"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--payloads", type=int, default=500)
    ap.add_argument("--rate", type=float, default=0.1, help="operator arrivals per source and step")
    args = ap.parse_args()
    tag = "" if args.rate == 0.1 else f"_rate{int(round(args.rate * 100)):03d}"
    out_dir = REPO / f"results/w01_drift_smoke{tag}"
    legacy.build()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = REPO / f"data/drift/smoke{tag}"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    summary = {"rate": args.rate, "n_timelines_per_track": args.seeds, "payloads_per_source_per_step": args.payloads, "regime_thresholds":
               {"far_delta_no_match": FAR_DELTA, "near_delta_silent_wrong": NEAR_DELTA}, "tracks": {}}
    regimes = collections.defaultdict(collections.Counter)
    event_rows = []
    for track in ("none", "in_curriculum", "heldout_params", "heldout_operators"):
        per_step = collections.defaultdict(lambda: collections.defaultdict(list))
        last = collections.defaultdict(list)
        for seed in range(args.seeds):
            path = cfg_dir / f"{track}-{seed:03d}.yaml"
            path.write_text(yaml.safe_dump({"name": path.stem, "track": track, "seed": seed, "rate": args.rate, "payloads_per_source": args.payloads}, sort_keys=False))
            cfg = timeline.load_config(path)
            steps, events = timeline.sample(cfg, keep_families=True)
            for ev in events:
                if ev["applied"]:
                    ev["regime"] = regime(ev, 400)
                    regimes[ev["op"]][ev["regime"]] += 1
                event_rows.append({"timeline": path.stem, **{k: v for k, v in ev.items() if k not in ("before", "after")},
                                   "trace_after": ev["after"]["trace"] if ev["applied"] else None})
            for t, fams in enumerate(steps):
                r = rates([row for _, row in timeline.render_step(cfg, fams, t)])
                for k in ("no_match", "silent_wrong", "yield"):
                    per_step[t][k].append(float(r[k].mean()))
                    if t == len(steps) - 1:
                        last[k].append(r[k])
        curve = [{"step": t, **{k: round(float(np.mean(v)), 5) for k, v in per_step[t].items()},
                  **{f"{k}_min": round(min(v), 5) for k, v in per_step[t].items()}, **{f"{k}_max": round(max(v), 5) for k, v in per_step[t].items()}}
                 for t in sorted(per_step)]
        final = {k: stats.bootstrap_ci(np.concatenate(v)).as_dict() for k, v in last.items()}
        summary["tracks"][track] = {"final_step": final, "curve": curve}
        print(f"{track:<18} final no-match {final['no_match']['point']:.3f}  silent-wrong {final['silent_wrong']['point']:.4f}  yield {final['yield']['point']:.3f}")
    summary["regime_by_operator"] = {op: dict(c) for op, c in sorted(regimes.items(), key=lambda kv: int(kv[0][1:]))}
    summary["note"] = ("Final-step CIs resample payloads pooled over timelines; payloads of one source share a family, so these "
                       "intervals understate between-timeline variance. The per-step min/max over timelines is reported next to them.")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in event_rows))
    print(json.dumps(summary["regime_by_operator"], indent=1))


if __name__ == "__main__":
    main()
