"""Gate G0: does a float student rescue what the frozen legacy parser rejects under drift, at bounded risk?

    .venv/bin/python scripts/w02_g0_rescue.py --model models/tel-stub-XS-fp [--seeds 5]     ->  results/w02_g0_<name>/

Legacy-first order. The rescue population is every payload the legacy parser rejects. The student accepts a rescue
line iff its score clears a threshold chosen on the in-distribution dev pool alone (largest coverage with empirical
selective risk <= 1%; the finite-sample certificate replaces this rule later). Reported per track at steps 9, 18, 26:
rescue rate = accepted and exact / rescue lines whose gold is a struct; selective risk on the rescue population;
hybrid yield against legacy yield on the same payloads (paired bootstrap). The max-probability score is the ablation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from static_student import legacy, stats
from static_student.drift import timeline
from static_student.student import data
from static_student.student.model import Student, StudentConfig
from static_student.student.train import device, predict, structs, wrong_if_accepted

REPO = Path(__file__).resolve().parent.parent
THRESHOLDS = {"far_no_match": 0.20, "rescue": 0.50, "risk": 0.02}


def pick_tau(score: np.ndarray, wrong: np.ndarray, emit: np.ndarray, alpha: float = 0.01) -> float:
    """Spans that do not decode to a struct are never emitted by the kernel, so they are deferrals, not accepted errors."""
    score, wrong = score[emit], wrong[emit]
    order = np.argsort(-score)
    risk = np.cumsum(wrong[order]) / np.arange(1, len(order) + 1)
    ok = np.nonzero(risk <= alpha)[0]
    return float(score[order][ok[-1]]) if len(ok) else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--families", default=None, help="families.jsonl the model was trained from (default: the path in its train report)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--payloads", type=int, default=500)
    args = ap.parse_args()
    mdir = REPO / args.model
    ck = torch.load(mdir / "student_fp.pt", map_location="cpu")
    dev = device()
    model = Student(StudentConfig(**ck["config"]))
    model.load_state_dict(ck["state"])
    model.to(dev)
    report = json.loads((mdir / "train_report.json").read_text())
    fams = data.load_families(REPO / (args.families or report["families_path"]))
    dev_rows = data.render_rows(data.split_families(fams, seed=report["seed"])["dev"], 20_000, "g0-dev")
    p = predict(model, data.encode(dev_rows)["ids"], dev)
    dev_got = structs(dev_rows, p)
    w, e = wrong_if_accepted(dev_rows, dev_got), np.array([g is not None for g in dev_got])
    scores = {"defer_head": lambda q: -q["defer"], "max_prob": lambda q: q["maxprob"]}
    tau = {k: pick_tau(f(p), w, e) for k, f in scores.items()}
    out = {"model": args.model, "params": report["params"], "thresholds": THRESHOLDS, "tau_rule": "dev pool, empirical selective risk <= 1%",
           "dev": {k: {"tau": tau[k], "coverage": float(((f(p) >= tau[k]) & e).mean()), "risk": float(w[(f(p) >= tau[k]) & e].mean()) if ((f(p) >= tau[k]) & e).any() else None} for k, f in scores.items()},
           "n_timelines_per_track": args.seeds, "tracks": {}}
    for track in ("in_curriculum", "heldout_params", "heldout_operators"):
        out["tracks"][track] = {}
        for step in (9, 18, 26):
            rows = []
            for seed in range(args.seeds):
                cfg = {**timeline.DEFAULTS, "name": f"{track}-{seed}", "track": track, "seed": seed, "payloads_per_source": args.payloads}
                steps, _ = timeline.sample(cfg)
                rows += [r for _, r in timeline.render_step(cfg, steps[step], step)]
            leg = legacy.parse_many([r.payload for r in rows])
            lo = legacy.outcomes(rows, leg)
            q = predict(model, data.encode(rows)["ids"], dev)
            got = structs(rows, q)
            wrong, emit = wrong_if_accepted(rows, got), np.array([g is not None for g in got])
            rescue = np.array([x is None for x in leg])
            struct = np.array([not r.defer for r in rows])
            cell = {"n": len(rows), "legacy_no_match_on_struct": float((rescue & struct).sum() / struct.sum()), "n_rescue": int(rescue.sum()),
                    "n_rescue_struct": int((rescue & struct).sum())}
            for name, f in scores.items():
                acc = (f(q) >= tau[name]) & rescue & emit
                hybrid_ok = np.where(rescue, np.where(acc, 1 - wrong, np.array([r.defer for r in rows], dtype=int)), np.array(lo["correct"]))
                cell[name] = {"rescue_rate": stats.bootstrap_ci(((acc & (wrong == 0))[rescue & struct]).astype(int)).as_dict(),
                              "risk_on_rescue": stats.selective_risk_ci(wrong[rescue], acc[rescue].astype(int)),
                              "yield_hybrid_minus_legacy": stats.paired_bootstrap_diff(hybrid_ok, np.array(lo["correct"])).as_dict()}
            cell["oracle_best_rescue_at_risk_2pct"] = None  # max-prob score, threshold chosen with test labels: an upper bound, not a result
            s, ww = q["maxprob"][rescue & emit], wrong[rescue & emit]
            order = np.argsort(-s)
            risk = np.cumsum(ww[order]) / np.arange(1, len(order) + 1)
            ok = np.nonzero(risk <= THRESHOLDS["risk"])[0]
            if len(ok):
                cell["oracle_best_rescue_at_risk_2pct"] = float((ok[-1] + 1 - ww[order][: ok[-1] + 1].sum()) / max(1, (rescue & struct).sum()))
            out["tracks"][track][str(step)] = cell
            for name in scores:
                d = cell[name]
                print(f"{track:<18} step {step:2d} {name:<10} no-match {cell['legacy_no_match_on_struct']:.3f}  rescue {d['rescue_rate']['point']:.3f}  risk {d['risk_on_rescue']['risk']:.4f} "
                      f"(cov {d['risk_on_rescue']['coverage']:.3f})  yield +{d['yield_hybrid_minus_legacy']['point']:.3f}  oracle@2% {cell['oracle_best_rescue_at_risk_2pct']}", flush=True)
    last = out["tracks"]["in_curriculum"]["26"]
    out["G0"] = {"far_no_match": last["legacy_no_match_on_struct"]}
    for name in scores:
        d = last[name]
        out["G0"][name] = {"rescue": d["rescue_rate"]["point"], "risk": d["risk_on_rescue"]["risk"],
                           "pass": bool(last["legacy_no_match_on_struct"] >= THRESHOLDS["far_no_match"] and d["rescue_rate"]["point"] >= THRESHOLDS["rescue"]
                                        and d["risk_on_rescue"]["risk"] <= THRESHOLDS["risk"])}
    res = REPO / f"results/w02_g0_{mdir.name}"
    res.mkdir(parents=True, exist_ok=True)
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out["G0"], indent=1), json.dumps(out["dev"], indent=1))


if __name__ == "__main__":
    main()
