"""Gate G1a: the emitted C kernel against the integer reference (bit-exact) and against the float model (decisions).

    .venv/bin/python scripts/w03_kernel_parity.py --model models/tel-mlx-S-w4 [--n 10000]   ->  results/w03_kernel_parity_<name>/

Generates C from the packed student, builds it, runs the same payloads through the C kernel, the NumPy reference and
PyTorch, and reports: every pointer and unit decision identical between C and the reference, the score identical to
the last bit, and the agreement of both with the float model.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from static_student.codegen import emit, oracle, pack, run_kernel
from static_student.student import data
from static_student.student.qat import load_student
from static_student.student.train import predict

REPO = Path(__file__).resolve().parent.parent
CC = ["cc", "-O3", "-std=c11", "-ffp-contract=off", "-Wall", "-Wextra"]


def build(gen: Path) -> Path:
    subprocess.run([*CC, "-I", str(gen), "-c", str(gen / "ss_weights.c"), "-o", str(gen / "ss_weights.o")], check=True)
    subprocess.run([*CC, "-I", str(gen), "-c", str(gen / "ss_kernel.c"), "-o", str(gen / "ss_kernel.o")], check=True)
    exe = gen / "ss_cli"
    subprocess.run([*CC, "-I", str(gen), str(REPO / "csrc/ss_cli.c"), str(gen / "ss_kernel.o"), str(gen / "ss_weights.o"), "-o", str(exe), "-lm"], check=True)
    return exe


def run_c(exe: Path, payloads: list[bytes]) -> dict[str, np.ndarray]:
    return run_kernel.score(exe, payloads)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    mdir = REPO / args.model
    rep = json.loads((mdir / "train_report.json").read_text())
    model, _ = load_student(mdir)
    model.eval()
    p = pack.pack(model)
    gen = REPO / "build" / f"gen_{mdir.name}"
    man = emit.emit(p, gen)
    t0 = time.time()
    exe = build(gen)
    build_s = time.time() - t0

    rows = data.render_rows(data.split_families(data.load_families(REPO / rep["families_path"]), seed=rep["seed"])["dev"], args.n, "parity")
    payloads = [r.payload for r in rows]
    ids = data.encode(rows)["ids"]

    t0 = time.time()
    c = run_c(exe, payloads)
    c_s = time.time() - t0

    t0 = time.time()
    order = np.argsort((ids != 256).sum(1))
    parts = {"ptr": [], "unit": [], "score": [], "defer": []}
    for i in range(0, len(ids), args.batch):
        idx = order[i:i + args.batch]
        d = oracle.decisions(oracle.forward(p, ids[idx]))
        parts["ptr"].append((idx, d["ptr"])), parts["unit"].append((idx, d["unit"]))
        parts["score"].append((idx, d["maxprob"])), parts["defer"].append((idx, d["defer"]))
    o = {k: np.concatenate([v for _, v in parts[k]])[np.argsort(np.concatenate([i for i, _ in parts[k]]))] for k in parts}
    o_s = time.time() - t0

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)
    fl = predict(model, ids, dev)

    bits_equal = lambda a, b: np.array_equal(a.astype("<f4").view(np.int32), b.astype("<f4").view(np.int32))  # noqa: E731
    out = {
        "model": args.model, "bits": rep.get("bits"), "n": len(payloads), "weight_bytes": man["weight_bytes"],
        "scratch_bytes": man["scratch_bytes"], "build_seconds": round(build_s, 1),
        "seconds": {"c_kernel": round(c_s, 2), "numpy_reference": round(o_s, 1)},
        "c_vs_reference": {
            "pointers_identical": float((c["ptr"] == o["ptr"]).all(-1).mean()),
            "unit_identical": float((c["unit"] == o["unit"]).mean()),
            "score_bit_identical": bool(bits_equal(c["score"], o["score"])),
            "score_max_abs_delta": float(np.abs(c["score"] - o["score"]).max()),
            "score_ulp_max": int(np.abs(c["score"].view(np.int32).astype(np.int64) - o["score"].view(np.int32).astype(np.int64)).max()),
            "defer_bit_identical": bool(bits_equal(c["defer"], o["defer"])),
            "defer_max_abs_delta": float(np.abs(c["defer"] - o["defer"]).max()),
        },
        "c_vs_float_model": {
            "pointers_identical": float((c["ptr"] == fl["ptr"]).all(-1).mean()),
            "unit_identical": float((c["unit"] == fl["unit"]).mean()),
            "score_max_abs_delta": float(np.abs(c["score"] - fl["maxprob"]).max()),
            "defer_max_abs_delta": float(np.abs(c["defer"] - fl["defer"]).max()),
        },
    }
    out["G1a"] = {"bit_exact_vs_reference": out["c_vs_reference"]["score_bit_identical"] and out["c_vs_reference"]["pointers_identical"] == 1.0
                                            and out["c_vs_reference"]["unit_identical"] == 1.0,
                  "argmax_vs_float_model": out["c_vs_float_model"]["pointers_identical"] == 1.0 and out["c_vs_float_model"]["unit_identical"] == 1.0}
    res = REPO / f"results/w03_kernel_parity_{mdir.name}"
    res.mkdir(parents=True, exist_ok=True)
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
