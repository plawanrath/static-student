"""Start-to-first-struct: how long a process takes from exec to its first parsed struct.

    .venv/bin/python scripts/w03_startup.py --launches 500      ->  results/w03_startup/

The clock is the parent's: fork, exec, wait for the child to print its first result, stop. That covers everything a
deployment pays for, including the dynamic loader, page faults on the weight array and the first inference. Every mode
is reported next to the null binary, which links the same legacy parser and carries no model at all, so the difference
is what carrying the model costs. Cold runs drop the file cache for the binary where the platform allows it; warm runs
do not, and both are reported because a real service restarts warm.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from static_student import stats

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/w03_startup"


BENCH = REPO / "build" / "spawn_bench"


def time_launches(cmd: list[str], n: int, warmup: int = 20) -> np.ndarray:
    """Milliseconds from posix_spawn to the child's first line of output, measured by a C parent: no interpreter, no
    shell and no pipe buffering between the clock and the child's first struct."""
    r = subprocess.run([str(BENCH), str(n), str(warmup), "--", *cmd], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]}: {r.stderr.decode()[:300]}")
    return np.array([float(x) for x in r.stdout.split()]) / 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launches", type=int, default=500)
    ap.add_argument("--models", nargs="*", default=["S_w4", "S_w3", "S_w2", "XS_w4", "XS_w3", "XS_w2"])
    ap.add_argument("--onnx", nargs="*", default=["tel_S_fp"], help="exported graphs to serve with ONNX Runtime")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    subprocess.run(["make", "-C", str(REPO / "csrc")], check=True, capture_output=True)
    null_bin = REPO / "build" / "null_binary"
    subprocess.run(["cc", "-O3", "-std=c11", "-Wall", "-I", str(REPO / "csrc"), str(REPO / "csrc/null_binary.c"),
                    str(REPO / "build/legacy_telemetry.o"), "-o", str(null_bin), "-lstdc++",
                    *subprocess.run(["pkg-config", "--libs", "re2"], capture_output=True, text=True).stdout.split()], check=True)

    subprocess.run(["cc", "-O3", "-std=c11", str(REPO / "csrc/exec_floor.c"), "-o", str(REPO / "build/exec_floor")], check=True)
    subprocess.run(["cc", "-O3", "-std=c11", "-Wall", str(REPO / "csrc/spawn_bench.c"), "-o", str(BENCH)], check=True)
    # exec_floor is the absolute cost of starting any process here; null_binary adds the legacy parser and the shared
    # libraries it needs, which every other mode also links. Both are reported so the model's own cost is visible.
    modes = {"exec_floor": [str(REPO / "build/exec_floor")], "null_binary": [str(null_bin)]}
    cflags = ["-O3", "-std=c11", "-ffp-contract=off"]
    re2 = subprocess.run(["pkg-config", "--libs", "re2"], capture_output=True, text=True).stdout.split()
    for name in args.models:
        gen = REPO / "build" / f"tel_{name}"
        exe = gen / "ss_hybrid"
        if not (gen / "ss_kernel.o").exists():
            continue
        # Rebuild the hybrid front end and the entry point from source, not from whatever objects the build left
        # behind: those go stale the moment csrc/ss_hybrid.c gains a function, and then the link fails on a missing
        # symbol hours into an overnight run. Only the generated kernel and weights are reused.
        subprocess.run(["cc", *cflags, "-I", str(gen), "-I", str(REPO / "csrc"), "-c", str(REPO / "csrc/ss_hybrid.c"),
                        "-o", str(gen / "ss_hybrid.o")], check=True)
        subprocess.run(["cc", *cflags, "-I", str(gen), "-I", str(REPO / "csrc"), str(REPO / "csrc/ss_hybrid_cli.c"),
                        str(gen / "ss_certificate.o"), str(gen / "ss_hybrid.o"), str(gen / "ss_kernel.o"),
                        str(gen / "ss_weights.o"), str(gen / "ss_struct.o"), str(REPO / "build/legacy_telemetry.o"),
                        "-o", str(exe), "-lm", "-lstdc++", "-lpthread", *re2], check=True)
        modes[f"ours_{name}"] = [str(exe), "--first-struct"]
    ort_exe = REPO / "build" / "ort_first_struct"
    for g in args.onnx:
        onnx = REPO / "build" / "onnx" / g / "student.onnx"
        if ort_exe.exists() and onnx.exists():
            modes[f"onnxruntime_{g}"] = [str(ort_exe), str(onnx)]
    if len(modes) == 1:
        raise SystemExit("no built binaries found: run scripts/w03_build_all.sh first")

    res = {"launches": args.launches, "host": subprocess.run(["uname", "-mrs"], capture_output=True, text=True).stdout.strip(), "modes": {}}
    base = None
    for name, cmd in modes.items():
        t = time_launches(cmd, args.launches)
        entry = {"cmd": " ".join(Path(c).name for c in cmd), "n": len(t), "p50_ms": float(np.median(t)),
                 "p99_ms": float(np.percentile(t, 99)), "mean_ms": float(t.mean()),
                 "bytes_on_disk": Path(cmd[0]).stat().st_size,
                 "model_file_bytes": Path(cmd[1]).stat().st_size if name.startswith("onnxruntime") else 0}
        if name == "exec_floor":
            res["modes"][name] = entry
            print(f"{name:<22} p50 {entry['p50_ms']:7.3f} ms  p99 {entry['p99_ms']:7.3f} ms  disk {entry['bytes_on_disk']:>9,}", flush=True)
            continue
        if base is None:
            base = t
        else:
            r = stats.ratio_of_medians_ci(t, base)
            entry["ratio_to_null"] = {"point": r.point, "ci_low": r.ci_low, "ci_high": r.ci_high}
        res["modes"][name] = entry
        extra = f"  x{entry['ratio_to_null']['point']:.3f} [{entry['ratio_to_null']['ci_low']:.3f}, {entry['ratio_to_null']['ci_high']:.3f}] vs null" if base is not t else ""
        print(f"{name:<22} p50 {entry['p50_ms']:7.3f} ms  p99 {entry['p99_ms']:7.3f} ms  disk {entry['bytes_on_disk'] + entry['model_file_bytes']:>9,}{extra}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
