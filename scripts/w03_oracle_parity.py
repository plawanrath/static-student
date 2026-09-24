"""Gate G1a, first half on the reference side: does the shipped integer model still decide what the float model decided,
and what does the contract cost when it is certified on the bits that ship instead of on the float weights?

    .venv/bin/python scripts/w03_oracle_parity.py --model models/tel-mlx-S-w4 [--n 5000] [--embed-bits 8 16]

For each embedding width: agreement of every pointer and unit argmax with PyTorch, the spread of the contract score,
and a certificate computed on the oracle's own outputs, next to the float model's certificate on the same pools.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from static_student import certify, stats
from static_student.codegen import oracle, pack
from static_student.curriculum.spec import parse_spec
from static_student.student import data
from static_student.student.qat import load_student
from static_student.student.train import predict, structs, wrong_if_accepted

REPO = Path(__file__).resolve().parent.parent


def repack(model, bits: int):
    p = pack.pack(model)
    if bits != 8:
        for name in ("tok", "pos"):
            src = getattr(model, name).weight.detach().numpy().astype(np.float32)
            lv, sc = pack.quantize_rows(src, bits)
            setattr(p, name, pack.RowLinear(lv, sc, None))
    return p


def oracle_scores(p, ids, batch=64):
    ptr, unit, mp = [], [], []
    order = np.argsort((ids != 256).sum(1))  # group payloads of similar length so trimming pays
    for i in range(0, len(ids), batch):
        idx = order[i:i + batch]
        d = oracle.decisions(oracle.forward(p, ids[idx]))
        ptr.append((idx, d["ptr"])), unit.append((idx, d["unit"])), mp.append((idx, d["maxprob"]))
    un = lambda parts, shape: np.concatenate([v for _, v in parts])[np.argsort(np.concatenate([i for i, _ in parts]))]  # noqa: E731
    return un(ptr, None), un(unit, None), un(mp, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--embed-bits", type=int, nargs="*", default=[8, 16])
    args = ap.parse_args()
    spec = parse_spec(REPO / "specs/telemetry.spec")
    mdir = REPO / args.model
    rep = json.loads((mdir / "train_report.json").read_text())
    model, _ = load_student(mdir)
    model.eval()
    split = data.split_families(data.load_families(REPO / rep["families_path"]), seed=rep["seed"])
    pools = {"dev": data.render_rows(split["dev"], args.n, "cert-dev"), "cal": data.render_rows(split["defer_fit"], args.n, "cert-cal"),
             "test": data.render_rows(split["defer_fit"], args.n, "cert-test")}
    enc = {k: data.encode(v)["ids"] for k, v in pools.items()}
    out = {"model": args.model, "bits": rep.get("bits"), "n_per_pool": args.n, "spec": {k: spec[k] for k in ("alpha", "delta", "min_coverage")}}

    tdev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(tdev)
    fl = {k: predict(model, v, tdev) for k, v in enc.items()}
    model.to("cpu")
    fl_w = {k: wrong_if_accepted(pools[k], structs(pools[k], fl[k])) for k in pools}
    fl_e = {k: np.array([g is not None for g in structs(pools[k], fl[k])]) for k in pools}
    grid = certify.coverage_grid(fl["dev"]["maxprob"], fl_e["dev"])
    c = certify.certify(fl["cal"]["maxprob"], fl_w["cal"], fl_e["cal"], spec["alpha"], spec["delta"], spec["min_coverage"], grid)
    acc = fl_e["test"] & (fl["test"]["maxprob"] >= c.tau)
    out["float_model"] = {"certificate": c.as_dict(), "test": stats.selective_risk_ci(fl_w["test"], acc.astype(int))}
    print("float  ", json.dumps({"cov": round(c.coverage, 4), "tau": round(c.tau, 4), "test_risk": round(out["float_model"]["test"]["risk"], 5)}))

    for bits in args.embed_bits:
        t0 = time.time()
        p = repack(model, bits)
        res = {}
        for k in ("dev", "cal", "test"):
            ptr, unit, mp = oracle_scores(p, enc[k])
            got = [data.decode(r.payload, ptr[i], int(unit[i])) for i, r in enumerate(pools[k])]
            res[k] = {"ptr": ptr, "unit": unit, "mp": mp, "wrong": wrong_if_accepted(pools[k], got), "emits": np.array([g is not None for g in got])}
        agree_ptr = float((res["test"]["ptr"] == fl["test"]["ptr"]).all(-1).mean())
        agree_unit = float((res["test"]["unit"] == fl["test"]["unit"]).mean())
        g2 = certify.coverage_grid(res["dev"]["mp"], res["dev"]["emits"])
        try:
            ck = certify.certify(res["cal"]["mp"], res["cal"]["wrong"], res["cal"]["emits"], spec["alpha"], spec["delta"], spec["min_coverage"], g2)
            acck = res["test"]["emits"] & (res["test"]["mp"] >= ck.tau)
            kernel = {"certificate": ck.as_dict(), "test": stats.selective_risk_ci(res["test"]["wrong"], acck.astype(int))}
        except certify.CertificationError as e:
            kernel = {"certified": False, "reason": str(e)}
        acc_float_tau = res["test"]["emits"] & (res["test"]["mp"] >= c.tau)
        out[f"oracle_embed{bits}"] = {
            "bytes": p.bytes(), "argmax_agreement_pointers": agree_ptr, "argmax_agreement_unit": agree_unit,
            "max_abs_score_delta": float(np.abs(res["test"]["mp"] - fl["test"]["maxprob"]).max()),
            "float_tau_on_kernel": stats.selective_risk_ci(res["test"]["wrong"], acc_float_tau.astype(int)), **kernel, "seconds": round(time.time() - t0, 1)}
        e = out[f"oracle_embed{bits}"]
        print(f"embed{bits}", json.dumps({"ptr_agree": round(agree_ptr, 4), "unit_agree": round(agree_unit, 4), "max_dscore": round(e["max_abs_score_delta"], 4),
                                          "kernel_cov": round(e.get("certificate", {}).get("coverage", 0), 4), "kernel_test_risk": round(e.get("test", {}).get("risk", 0), 5),
                                          "float_tau_risk_on_kernel": round(e["float_tau_on_kernel"]["risk"], 5), "bytes": p.bytes()["total"], "s": e["seconds"]}))
    res_dir = REPO / f"results/w03_oracle_parity_{mdir.name}"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
