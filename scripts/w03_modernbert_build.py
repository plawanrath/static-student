"""Compile an imported ModernBERT classifier, check it against the integer reference, and calibrate on its own outputs.

    .venv/bin/python scripts/w03_modernbert_build.py --model toxicity_large --bits 8   ->  build/mb_<model>_w<bits>/
                                                                                       results/w03_modernbert_<model>_w<bits>/
Steps: pack (post-training quantization) -> emit C + weights as a linked constant -> compile -> run the kernel on the
locked dev / cal / test pools (10,000 inputs) -> bit-exactness against the NumPy reference on the first --parity test
inputs -> accuracy of the kernel against PyTorch fp32 on the test pool -> Learn-then-Test on the kernel's own scores ->
guarantee record compiled into the binary. The GC gate (ADR-0016) is the parity and accuracy clauses at 8 bits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w02_cert_gap_pilot as P  # noqa: E402

from static_student import certify, stats  # noqa: E402
from static_student.codegen import emit_modernbert as E  # noqa: E402
from static_student.codegen import modernbert as MB  # noqa: E402

REPO = P.REPO
_PK = None


def _oracle(x):
    logits = MB.forward(_PK, x)
    lab, prob = MB.decision(logits)
    return lab, np.float32(prob), logits


def parse(line: str):
    f = line.split()
    return int(f[0]), np.float32(float.fromhex(f[1])), np.array([float.fromhex(v) for v in f[2:]], dtype=np.float32)


def main() -> None:
    global _PK
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="toxicity_large")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--parity", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    tag = f"{args.model}_w{args.bits}"
    out, res = REPO / f"build/mb_{tag}", REPO / f"results/w03_modernbert_{tag}"
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model, pools, rev, sha = P.build_pools(args.model)
    _PK = MB.pack(model, args.bits)
    man = E.emit(_PK, out)
    exe = E.compile_cli(out)
    print(f"[mb] packed + emitted + compiled {man['weight_bytes'] / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)

    lines = {}
    for pool in ("test", "dev", "cal"):
        t1 = time.time()
        lines[pool] = E.run_cli(exe, pools[pool]["ids"], jobs=args.jobs)
        (res / f"kernel_{pool}.txt").write_text("\n".join(lines[pool]) + "\n")
        print(f"[mb] kernel on {pool}: {len(lines[pool])} inputs in {time.time() - t1:.0f}s", flush=True)
    per_input_s = None

    # parity against the integer reference (the GC gate's first clause)
    t1 = time.time()
    n = min(args.parity, len(pools["test"]["ids"]))
    with mp.get_context("fork").Pool(args.jobs) as pool:
        ref = pool.map(_oracle, pools["test"]["ids"][:n], chunksize=4)
    exact = 0
    for (rl, rp, rlog), line in zip(ref, lines["test"][:n]):
        kl, kp, klog = parse(line)
        exact += int(kl == rl and kp.view(np.uint32) == rp.view(np.uint32) and np.array_equal(klog.view(np.uint32), rlog.view(np.uint32)))
    print(f"[mb] parity {exact}/{n} bit-exact ({time.time() - t1:.0f}s)", flush=True)

    # accuracy of the shipped kernel vs PyTorch fp32 (the second clause)
    t1 = time.time()
    torch.set_num_threads(8)
    with torch.no_grad():
        fl = np.concatenate([model(input_ids=torch.from_numpy(x)[None], attention_mask=torch.ones(1, len(x), dtype=torch.long)).logits.numpy()
                             for x in pools["test"]["ids"]])
    y = pools["test"]["label"]
    k_pred = np.array([parse(l)[0] for l in lines["test"]])
    acc_float, acc_kernel = float(np.mean(fl.argmax(1) == y)), float(np.mean(k_pred == y))
    agree = float(np.mean(fl.argmax(1) == k_pred))
    print(f"[mb] accuracy float {100 * acc_float:.2f}% kernel {100 * acc_kernel:.2f}% agreement {100 * agree:.2f}% ({time.time() - t1:.0f}s)", flush=True)

    # Learn-then-Test on the kernel's own scores
    sc = {p: (np.array([parse(l)[0] for l in lines[p]]), np.array([parse(l)[1] for l in lines[p]], dtype=np.float64), pools[p]["label"]) for p in lines}
    records = {}
    for alpha in (0.05, 0.02):
        c = P.calibrate(sc, alpha)
        if isinstance(c, str):
            records[str(alpha)] = {"passes": False, "reason": c}
            continue
        acc = sc["test"][1] >= c.tau
        r = stats.selective_risk_ci((sc["test"][0] != sc["test"][2]).astype(int), acc.astype(int))
        records[str(alpha)] = {"passes": True, "record": c.as_dict(), "test": r}
    rec_line = json.dumps({"model": args.model, "bits": args.bits, "weights_sha256": man["weights_sha256"], "kernel_sha256": man["kernel_sha256"],
                           "token_ids_sha256": sha, "records": {a: v.get("record") for a, v in records.items()}}, sort_keys=True, default=float)
    (out / "mb_record.c").write_text('/* GENERATED guarantee record */\nconst char mb_record[] = ' + json.dumps("MBREC01 " + rec_line) + ';\n')
    subprocess.run(["cc", *E.cflags(), "-c", "mb_record.c", "-o", "mb_record.o"], cwd=out, check=True)
    subprocess.run(["cc", *E.cflags(), "mb_cli.c", "mb_kernel.o", "mb_weights.o", "mb_record.o", "-o", "mb_cli", "-lm", "-lpthread"], cwd=out, check=True)
    summ = {"model": args.model, "source": P.MODELS[args.model][0], "revision": rev, "bits": args.bits, "weight_bytes": man["weight_bytes"],
            "weights_sha256": man["weights_sha256"], "kernel_sha256": man["kernel_sha256"], "binary_bytes": (out / "mb_cli").stat().st_size,
            "pools": {p: len(v["ids"]) for p, v in pools.items()}, "token_ids_sha256": sha,
            "parity": {"n": n, "bit_exact": exact}, "accuracy": {"float_fp32": acc_float, "kernel": acc_kernel, "argmax_agreement": agree},
            "GC": {"bit_exact_all": exact == n and n >= 1000, "accuracy_within_2pp": abs(acc_float - acc_kernel) <= 0.02},
            "guarantee": records, "seconds": round(time.time() - t0, 1)}
    summ["GC"]["pass"] = summ["GC"]["bit_exact_all"] and summ["GC"]["accuracy_within_2pp"]
    (res / "summary.json").write_text(json.dumps(summ, indent=1, default=float) + "\n")
    print("[mb] GC:", json.dumps(summ["GC"]), flush=True)
    print("[mb] guarantee:", json.dumps({a: (v["record"]["tau"], v["record"]["coverage"]) if v.get("passes") else v for a, v in records.items()}, default=float))


if __name__ == "__main__":
    main()
