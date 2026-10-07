"""Does a threshold certified on the framework model hold on the kernel that ships?

    .venv/bin/python scripts/w02_cert_transfer.py            ->  results/w02_cert_transfer/summary.json

For every shipped telemetry configuration (size x bits), a threshold is certified with Learn-then-Test on the scores
of each calibration path, exactly as the build does it (grid from that path's dev scores, certificate on its
calibration scores), and then applied to the compiled kernel's scores on a disjoint test pool. Paths:

  fp32_student : the unquantized student of the same size (certify before quantizing)
  qat_cpu      : the quantization-aware model in PyTorch on the CPU
  qat_mps      : the same model in PyTorch on the Apple GPU
  kernel       : the compiled kernel itself (what the build certifies)

Reported per path: the certified threshold, realized selective risk and coverage of the shipped kernel under that
threshold (bootstrap CIs), the coverage difference against the kernel-certified threshold (paired bootstrap), and
the share of test inputs whose accept decision differs from the kernel-certified rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from static_student import certify, stats
from static_student.build import outcomes
from static_student.codegen import run_kernel
from static_student.curriculum.spec import parse_spec
from static_student.student import data
from static_student.student.qat import load_student
from static_student.student.train import predict, structs, wrong_if_accepted

REPO = Path(__file__).resolve().parent.parent


def torch_scores(model, rows, dev):
    p = predict(model, data.encode(rows)["ids"], dev)
    got = structs(rows, p)
    return p["maxprob"].astype(np.float64), wrong_if_accepted(rows, got), np.array([g is not None for g in got])


def kernel_path(exe, rows, jobs):
    k = run_kernel.score(exe, [r.payload for r in rows], jobs)
    wrong, emits = outcomes(rows, k)
    return k["score"].astype(np.float64), wrong, emits


def certify_path(dev, cal, spec, pool_sha):
    grid = certify.coverage_grid(dev[0], dev[2])
    try:
        return certify.certify(cal[0], cal[1], cal[2], spec["alpha"], spec["delta"], spec["min_coverage"], grid, pool_sha256=pool_sha)
    except certify.CertificationError as e:
        return str(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="XS,S,M")
    ap.add_argument("--bits", default="4,3,2")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    spec = parse_spec(REPO / "specs/telemetry.spec")
    out = {"spec": {k: spec[k] for k in ("alpha", "delta", "min_coverage")}, "n_per_pool": args.n, "configs": {}}
    for size in args.sizes.split(","):
        fp_dir = REPO / f"models/tel-mlx-{size}-fp"
        for bits in args.bits.split(","):
            name = f"tel_{size}_w{bits}"
            mdir = REPO / f"models/tel-mlx-{size}-w{bits}"
            exe = REPO / f"build/{name}/ss_cli"
            rep = json.loads((mdir / "train_report.json").read_text())
            rep_fp = json.loads((fp_dir / "train_report.json").read_text())
            assert (rep["families_path"], rep["seed"]) == (rep_fp["families_path"], rep_fp["seed"]), "splits differ"
            split = data.split_families(data.load_families(REPO / rep["families_path"]), seed=rep["seed"])
            pools = {"dev": data.render_rows(split["dev"], args.n, "cert-dev"),
                     "cal": data.render_rows(split["defer_fit"], args.n, "cert-cal"),
                     "test": data.render_rows(split["defer_fit"], args.n, "cert-test")}
            pool_sha = hashlib.sha256(b"".join(r.payload for r in pools["cal"])).hexdigest()

            scored = {"kernel": {p: kernel_path(exe, rows, args.jobs) for p, rows in pools.items()}}
            qat, _ = load_student(mdir)
            fp, _ = load_student(fp_dir)
            for label, model, dev in (("fp32_student", fp, "cpu"), ("qat_cpu", qat, "cpu"), ("qat_mps", qat, "mps")):
                model.to(torch.device(dev))
                scored[label] = {p: torch_scores(model, rows, torch.device(dev)) for p, rows in pools.items() if p != "test"}
                model.to(torch.device("cpu"))

            ks, kw, ke = scored["kernel"]["test"]
            kcert = certify_path(scored["kernel"]["dev"], scored["kernel"]["cal"], spec, pool_sha)
            assert not isinstance(kcert, str), f"{name}: the shipped kernel no longer certifies: {kcert}"
            built = json.loads((REPO / f"build/{name}/build_report.json").read_text())["certificate"]
            if args.n == built["n_calibration"]:  # same pools as the build: the certificate must be reproduced exactly
                assert (kcert.tau, kcert.n_accepted, kcert.k_wrong) == (built["tau"], built["n_accepted"], built["k_wrong"]), name
            acc_k = ke & (ks >= kcert.tau)
            res = {}
            for label in ("kernel", "fp32_student", "qat_cpu", "qat_mps"):
                cert = kcert if label == "kernel" else certify_path(scored[label]["dev"], scored[label]["cal"], spec, pool_sha)
                if isinstance(cert, str):
                    res[label] = {"certified": False, "reason": cert}
                    continue
                acc = ke & (ks >= cert.tau)
                r = stats.selective_risk_ci(kw, acc.astype(int))
                d = stats.paired_bootstrap_diff(acc.astype(int), acc_k.astype(int))
                res[label] = {"certified": True, "tau": cert.tau, "cal_coverage": cert.coverage, "cal_risk": cert.empirical_risk,
                              "shipped_test": r, "violates_alpha": bool(r["risk_ci_low"] > spec["alpha"]),
                              "coverage_minus_kernel_rule": {"point": d.point, "ci_low": d.ci_low, "ci_high": d.ci_high},
                              "decisions_differing_from_kernel_rule": float(np.mean(acc != acc_k))}
                print(name, label, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in
                      {"tau": cert.tau, "risk": r["risk"], "risk_hi": r["risk_ci_high"], "cov": r["coverage"],
                       "dcov": d.point, "differ": res[label]["decisions_differing_from_kernel_rule"]}.items()}), flush=True)
            out["configs"][name] = {"model": str(mdir.relative_to(REPO)), "fp32_model": str(fp_dir.relative_to(REPO)),
                                    "kernel": str(exe.relative_to(REPO)), "pool_sha256": pool_sha, "paths": res}
    dst = REPO / "results/w02_cert_transfer"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("wrote", dst / "summary.json")


if __name__ == "__main__":
    main()
