"""Re-derive the compiled ModernBERT's outputs and guarantee records on a second machine from identical bytes.

    .venv/bin/python scripts/w03_x86_modernbert.py export                         # reference: build/x86_bundle_mb/
    python scripts/w03_x86_modernbert.py run --tag <machine>                      # any machine: results/w03_x86_mb/<machine>/
    .venv/bin/python scripts/w03_x86_modernbert.py compare --other <machine>      # -> results/w03_x86_mb/compare_<machine>.json

The bundle carries the generated kernel source, the weights constant, the guarantee record, and the token ids of the
three pools, each checked by SHA-256 on arrival. The target compiles with its own native flags (-ffp-contract=off
always), so integer dot products may be vectorized differently on each machine and float work may not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # the package is not installed on a fresh machine
BUNDLE = REPO / "build/x86_bundle_mb"
OUT = REPO / "results/w03_x86_mb"
TAGS = ("toxicity_large_w8", "toxicity_large_w4")
FILES = ("mb_weights.bin", "mb_weights.S", "mb_model.h", "mb_kernel.c", "mb_cli.c", "mb_record.c", "manifest.json")
POOLS = ("dev", "cal", "test")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def export() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    import w02_cert_gap_pilot as P
    _, _, pools, _, sha = P.build_pools("toxicity_large")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    np.savez(BUNDLE / "inputs.npz", **{f"{p}_ids": np.concatenate(pools[p]["ids"]) for p in POOLS},
             **{f"{p}_len": np.array([len(x) for x in pools[p]["ids"]]) for p in POOLS})
    man = {"token_ids_sha256": sha, "inputs.npz": sha256(BUNDLE / "inputs.npz"), "builds": {}}
    for tag in TAGS:
        src, dst = REPO / f"build/mb_{tag}", BUNDLE / tag
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            shutil.copy2(src / f, dst / f)
        man["builds"][tag] = {f: sha256(dst / f) for f in FILES}
    (BUNDLE / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print("bundle", BUNDLE)


def run(tag_machine: str, jobs: int) -> None:
    from static_student.codegen import emit_modernbert as E
    man = json.loads((BUNDLE / "manifest.json").read_text())
    assert sha256(BUNDLE / "inputs.npz") == man["inputs.npz"], "inputs damaged"
    z = np.load(BUNDLE / "inputs.npz")
    pools = {p: np.split(z[f"{p}_ids"], np.cumsum(z[f"{p}_len"])[:-1]) for p in POOLS}
    dst = OUT / tag_machine
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "machine.json").write_text(json.dumps({"platform": platform.platform(), "machine": platform.machine(), "cflags": E.cflags()}, indent=1) + "\n")
    for tag in TAGS:
        b = BUNDLE / tag
        bad = [f for f, h in man["builds"][tag].items() if sha256(b / f) != h]
        assert not bad, f"{tag}: damaged {bad}"
        f = E.cflags()
        import subprocess
        for step in (["cc", *f, "-c", "mb_weights.S", "-o", "mb_weights.o"], ["cc", *f, "-c", "mb_kernel.c", "-o", "mb_kernel.o"],
                     ["cc", *f, "-c", "mb_record.c", "-o", "mb_record.o"],
                     ["cc", *f, "mb_cli.c", "mb_kernel.o", "mb_weights.o", "mb_record.o", "-o", "mb_cli", "-lm", "-lpthread"]):
            subprocess.run(step, cwd=b, check=True)
        import os
        probe = "\n".join(" ".join(str(int(v)) for v in x) for x in pools["dev"][:200]) + "\n"
        lat = {}
        for th in (1, 8):  # per-input latency in one warm process, same 200 probes as scripts/w03_price.py
            t0 = time.perf_counter()
            subprocess.run([str(b / "mb_cli")], input=probe.encode(), capture_output=True, check=True, env={**os.environ, "MB_THREADS": str(th)})
            lat[f"threads_{th}_mean_ms_per_input"] = 1000 * (time.perf_counter() - t0) / 200
        (dst / f"{tag}_latency.json").write_text(json.dumps(lat, indent=1) + "\n")
        print(tag, "latency", lat, flush=True)
        for p in POOLS:
            t0 = time.time()
            lines = E.run_cli(b / "mb_cli", pools[p], jobs=jobs)
            (dst / f"{tag}_kernel_{p}.txt").write_text("\n".join(lines) + "\n")
            print(tag, p, len(lines), f"{time.time() - t0:.0f}s", flush=True)
    print("done", dst)


def compare(other: str) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    import w02_cert_gap_pilot as P
    from w03_modernbert_build import parse
    _, _, pools, _, _ = P.build_pools("toxicity_large")
    res = {"other": json.loads((OUT / other / "machine.json").read_text()), "builds": {}}
    for tag in TAGS:
        ref_dir = REPO / f"results/w03_modernbert_{tag}"
        row = {"outputs_byte_identical": {}, "records": {}}
        sc = {}
        for p in POOLS:
            a, b = (ref_dir / f"kernel_{p}.txt").read_bytes(), (OUT / other / f"{tag}_kernel_{p}.txt").read_bytes()
            row["outputs_byte_identical"][p] = a == b
            ls = b.decode().splitlines()
            sc[p] = (np.array([parse(l)[0] for l in ls]), np.array([parse(l)[1] for l in ls], dtype=np.float64), pools[p]["label"])
        shipped = json.loads((ref_dir / "summary.json").read_text())["guarantee"]
        for alpha in ("0.05", "0.02"):
            c = P.calibrate(sc, float(alpha))
            mine = None if isinstance(c, str) else c.as_dict()
            ref = shipped[alpha].get("record")
            row["records"][alpha] = {"other": mine, "identical_to_shipped": mine == ref}
        res["builds"][tag] = row
    res["H-X3_modernbert"] = all(all(v["outputs_byte_identical"].values()) and all(r["identical_to_shipped"] for r in v["records"].values())
                                 for v in res["builds"].values())
    (OUT / f"compare_{other}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({t: {"bytes": v["outputs_byte_identical"], "records": {a: r["identical_to_shipped"] for a, r in v["records"].items()}}
                      for t, v in res["builds"].items()}), "H-X3_modernbert:", res["H-X3_modernbert"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("export", "run", "compare"))
    ap.add_argument("--tag", default=platform.machine())
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--other")
    a = ap.parse_args()
    {"export": export, "run": lambda: run(a.tag, a.jobs), "compare": lambda: compare(a.other)}[a.mode]()


if __name__ == "__main__":
    main()
