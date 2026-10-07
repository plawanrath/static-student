"""Outputs of the compiled ModernBERT do not depend on the thread count or on the instruction path.

    .venv/bin/python scripts/w03_thread_invariance.py      ->  results/w03_thread_invariance/summary.json

For each shipped width, the test pool (4,000 inputs) is scored at 1 and 8 threads and by the same source compiled with
-DMB_PORTABLE (plain C integer dot products instead of the ARM dot-product path), and every output line is compared
byte for byte with the shipped kernel's single-threaded output in results/w03_modernbert_<model>_w<bits>/.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w02_cert_gap_pilot as P  # noqa: E402

from static_student.codegen import emit_modernbert as E  # noqa: E402

REPO = P.REPO
MODEL = "toxicity_large"


def portable_cli(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("mb_weights.bin", "mb_weights.S", "mb_model.h", "mb_kernel.c", "mb_cli.c", "mb_record.c"):
        shutil.copy2(src / f, dst / f)
    f = E.cflags()
    for step in (["cc", *f, "-c", "mb_weights.S", "-o", "mb_weights.o"], ["cc", *f, "-DMB_PORTABLE", "-c", "mb_kernel.c", "-o", "mb_kernel.o"],
                 ["cc", *f, "-c", "mb_record.c", "-o", "mb_record.o"],
                 ["cc", *f, "mb_cli.c", "mb_kernel.o", "mb_weights.o", "mb_record.o", "-o", "mb_cli", "-lm", "-lpthread"]):
        subprocess.run(step, cwd=dst, check=True)
    return dst / "mb_cli"


def main() -> None:
    _, _, pools, _, _ = P.build_pools(MODEL)
    ids = pools["test"]["ids"]
    res = {"model": MODEL, "machine": platform.platform(), "inputs": len(ids), "builds": {}}
    for bits in (8, 4):
        src = REPO / f"build/mb_{MODEL}_w{bits}"
        ref = (REPO / f"results/w03_modernbert_{MODEL}_w{bits}/kernel_test.txt").read_text().splitlines()
        runs = {"threads_1": (src / "mb_cli", 1, 16), "threads_8": (src / "mb_cli", 8, 3)}
        runs["portable_threads_1"] = (portable_cli(src, REPO / f"build/mb_{MODEL}_w{bits}_portable"), 1, 16)
        row = {}
        for name, (exe, th, jobs) in runs.items():
            out = E.run_cli(exe, ids, jobs=jobs, threads=th)
            row[name] = {"identical_lines": sum(a == b for a, b in zip(out, ref)), "of": len(ref)}
            print(f"w{bits} {name}: {row[name]['identical_lines']}/{len(ref)} identical", flush=True)
        res["builds"][f"w{bits}"] = row
    res["all_identical"] = all(v["identical_lines"] == v["of"] for b in res["builds"].values() for v in b.values())
    out = REPO / "results/w03_thread_invariance"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print("all identical:", res["all_identical"])


if __name__ == "__main__":
    main()
