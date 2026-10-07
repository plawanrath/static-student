"""The price of reproducibility: latency and bytes of the compiled kernel against engines serving the same model.

    .venv/bin/python scripts/w03_price.py --model toxicity_large     ->  results/w03_price_<model>_<machine>/summary.json

Same 200 held-out probe inputs (the first 200 of the dev pool, never in any calibration or test split), batch 1, idle
machine. Per-input latency is measured inside one warm process; start-to-first-decision is measured from process
spawn to the first output line, 20 launches, file cache warm. Our kernel is single-threaded; engines are measured at 1
thread (same resources) and 8 threads (their best on this machine).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w02_cert_gap_pilot as P  # noqa: E402

REPO = P.REPO
N_PROBE = 200


def cpu() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    except Exception:
        return platform.processor()


def ours(build: Path, ids, threads: int = 1) -> dict:
    import os
    exe = build / "mb_cli"
    env = {**os.environ, "MB_THREADS": str(threads)}
    text = ("\n".join(" ".join(str(int(v)) for v in x) for x in ids) + "\n").encode()
    t0 = time.perf_counter()
    subprocess.run([str(exe)], input=text, capture_output=True, check=True, env=env)
    per = (time.perf_counter() - t0) / len(ids)  # includes one start-up, amortized over 200 inputs
    first = []
    one = (" ".join(str(int(v)) for v in ids[0]) + "\n").encode()
    for _ in range(20):
        t0 = time.perf_counter()
        pr = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        pr.stdin.write(one)
        pr.stdin.close()
        pr.stdout.readline()
        first.append(time.perf_counter() - t0)
        pr.wait()
    return {"mean_ms_per_input": 1000 * per, "start_to_first_decision_ms_median": 1000 * float(np.median(first)),
            "binary_bytes": exe.stat().st_size}


def ort(path: Path, ids, threads: int) -> dict:
    import onnxruntime as rt
    so = rt.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    t0 = time.perf_counter()
    s = rt.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    s.run(["logits"], {"input_ids": ids[0][None], "attention_mask": np.ones_like(ids[0])[None]})
    first = time.perf_counter() - t0
    t = []
    for x in ids:
        t0 = time.perf_counter()
        s.run(["logits"], {"input_ids": x[None], "attention_mask": np.ones_like(x)[None]})
        t.append(time.perf_counter() - t0)
    return {"mean_ms_per_input": 1000 * float(np.mean(t)), "median_ms_per_input": 1000 * float(np.median(t)),
            "session_create_plus_first_ms": 1000 * first, "model_bytes": path.stat().st_size, "threads": threads}


def torch_fp32(model, ids, threads: int) -> dict:
    import torch
    torch.set_num_threads(threads)
    t = []
    with torch.no_grad():
        for x in ids:
            t0 = time.perf_counter()
            model(input_ids=torch.from_numpy(x)[None], attention_mask=torch.ones(1, len(x), dtype=torch.long))
            t.append(time.perf_counter() - t0)
    return {"mean_ms_per_input": 1000 * float(np.mean(t)), "median_ms_per_input": 1000 * float(np.median(t)), "threads": threads}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="toxicity_large")
    a = ap.parse_args()
    tok, model, pools, rev, sha = P.build_pools(a.model)
    ids = pools["dev"]["ids"][:N_PROBE]
    b = REPO / f"build/price_{a.model}"
    b.mkdir(parents=True, exist_ok=True)
    fp32 = P.export_onnx(model, b)
    q8 = P.quantized(model, b)
    res = {"model": a.model, "machine": {"cpu": cpu(), "platform": platform.platform()}, "probe_inputs": len(ids),
           "mean_tokens": float(np.mean([len(x) for x in ids])), "paths": {}}
    for bits in (8, 4):
        d = REPO / f"build/mb_{a.model}_w{bits}"
        if (d / "mb_cli").exists():
            for th in (1, 8):
                res["paths"][f"ours_w{bits}_t{th}"] = ours(d, ids, th) | {"weight_bytes": (d / "mb_weights.bin").stat().st_size, "threads": th}
                print(f"ours_w{bits}_t{th}", res["paths"][f"ours_w{bits}_t{th}"], flush=True)
    for name, path in (("ort_fp32", fp32), ("ort_int8_dynamic", q8)):
        for th in (1, 8):
            res["paths"][f"{name}_t{th}"] = ort(path, ids, th)
            print(f"{name}_t{th}", res["paths"][f"{name}_t{th}"], flush=True)
    for th in (1, 8):
        res["paths"][f"torch_fp32_t{th}"] = torch_fp32(model, ids, th)
        print(f"torch_fp32_t{th}", res["paths"][f"torch_fp32_t{th}"], flush=True)
    tagm = "".join(ch for ch in res["machine"]["cpu"].replace(" ", "_") if ch.isalnum() or ch in "_-")[:40]
    out = REPO / f"results/w03_price_{a.model}_{tagm}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
