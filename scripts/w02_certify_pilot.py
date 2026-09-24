"""Pilot certificate for a float student (the shipped-kernel version replaces the scores once the C kernel exists).

    .venv/bin/python scripts/w02_certify_pilot.py --model models/tel-stub12k-v2-S-fp     ->  results/w02_certify_pilot_<name>/

Grid placed on the dev families; certificate computed on a calibration pool rendered from families that neither
training nor model selection touched; then checked on a disjoint test pool from the same families (different draws).
Both the certificate and a deliberately broken student (pointer head shuffled) are reported: the second must fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from static_student import certify, stats
from static_student.curriculum.spec import parse_spec
from static_student.student import data
from static_student.student import quant
from static_student.student.qat import load_student
from static_student.student.train import predict, structs, wrong_if_accepted

REPO = Path(__file__).resolve().parent.parent


def score_pool(model, rows, dev):
    p = predict(model, data.encode(rows)["ids"], dev)
    got = structs(rows, p)
    return p["maxprob"], wrong_if_accepted(rows, got), np.array([g is not None for g in got])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-broken", action="store_true", help="skip the corrupted-student check")
    args = ap.parse_args()
    spec = parse_spec(REPO / "specs/telemetry.spec")
    mdir = REPO / args.model
    rep = json.loads((mdir / "train_report.json").read_text())
    dev = torch.device(args.device)
    model, _ = load_student(mdir)
    model.to(dev)
    split = data.split_families(data.load_families(REPO / rep["families_path"]), seed=rep["seed"])
    pools = {"dev": data.render_rows(split["dev"], args.n, "cert-dev"), "cal": data.render_rows(split["defer_fit"], args.n, "cert-cal"),
             "test": data.render_rows(split["defer_fit"], args.n, "cert-test")}
    sha = hashlib.sha256(b"".join(r.payload for r in pools["cal"])).hexdigest()
    out = {"model": args.model, "size": rep["size"], "bits": rep.get("bits"), "bytes": quant.weight_bytes(model) if rep.get("bits") else None, "spec": {k: spec[k] for k in ("alpha", "delta", "min_coverage")}, "score": "max_prob", "pools": {k: len(v) for k, v in pools.items()},
           "note": "pilot: float model, stub curriculum, calibration pool = rendered payloads of families unseen in training and selection (not the real-anchored pool)"}
    for name, m in (("student", model),) + (() if args.no_broken else (("broken_student", None),)):
        if m is None:  # a student whose pointer head was corrupted after training: the build must refuse it
            m, _ = load_student(mdir)
            with torch.no_grad():
                m.pointers.weight.copy_(m.pointers.weight[torch.tensor([2, 3, 0, 1])] * 0.2)
            m.to(dev)
        s_dev, _, e_dev = score_pool(m, pools["dev"], dev)
        grid = certify.coverage_grid(s_dev, e_dev)
        s, w, e = score_pool(m, pools["cal"], dev)
        try:
            cert = certify.certify(s, w, e, spec["alpha"], spec["delta"], spec["min_coverage"], grid, pool_sha256=sha)
        except certify.CertificationError as err:
            out[name] = {"certified": False, "build": "FAILED", "reason": str(err)}
            print(name, "-> build FAILED:", err)
            continue
        st, wt, et = score_pool(m, pools["test"], dev)
        acc = et & (st >= cert.tau)
        out[name] = {"certified": True, "certificate": cert.as_dict(), "test": stats.selective_risk_ci(wt, acc.astype(int))}
        print(name, json.dumps({"tau": round(cert.tau, 4), "cal_coverage": round(cert.coverage, 4), "cal_risk": round(cert.empirical_risk, 5), "n_accepted": cert.n_accepted,
                                "k_wrong": cert.k_wrong, "test_risk": round(out[name]["test"]["risk"], 5), "test_coverage": round(out[name]["test"]["coverage"], 4)}))
    res = REPO / f"results/w02_certify_pilot_{mdir.name}"
    res.mkdir(parents=True, exist_ok=True)
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
