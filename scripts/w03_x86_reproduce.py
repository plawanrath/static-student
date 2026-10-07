"""Re-derive every guarantee on a second machine from identical artifact bytes, and compare.

    .venv/bin/python scripts/w03_x86_reproduce.py export              # reference machine: write build/x86_bundle/
    python scripts/w03_x86_reproduce.py run --tag <machine>           # any machine: results/w03_x86/<machine>/
    .venv/bin/python scripts/w03_x86_reproduce.py compare --ref <a> --other <b>   ->  results/w03_x86/compare_<a>_<b>.json

Two kinds of artifact are carried over byte for byte and checked by SHA-256 on arrival:
  * our telemetry kernels: the generated C (weights as constants), compiled on the target with the build's flags, then
    run on the exact calibration and dev payloads; the guarantee record is re-derived from the target's own scores;
  * vendor artifacts: ONNX files (fp32, int8 dynamic, int8 per-channel, int8 static QDQ) for three public or
    fine-tuned encoders, fed the exact token ids the reference machine used; PyTorch fp32 from pinned weights.
`run` writes raw kernel output lines and logits; `compare` does all statistics on the reference machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "build/x86_bundle"
OUT = REPO / "results/w03_x86"
TEL = [f"tel_{s}_w{b}" for s in ("XS", "S", "M") for b in (4, 3, 2)]
CFLAGS = ["-O3", "-std=c11", "-ffp-contract=off", "-D_POSIX_C_SOURCE=200809L"]
VENDOR = {  # model -> {path name -> artifact source on the reference machine}
    "sentiment": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "injection": "protectai/deberta-v3-base-prompt-injection-v2",
    "toxicity": "models/modernbert-base-civil",
}
ONNX = {"ort_fp32": "build/cert_gap_pilot/{m}/model_fp32.onnx", "ort_int8_dynamic": "build/cert_gap_pilot/{m}/model_int8_dynamic.onnx",
        "ort_int8_perchannel": "build/cert_gap_pilot/{m}/model_int8_dynamic_perchannel.onnx", "ort_int8_qdq": "build/contract_search_spike/{m}/spike_qdq.onnx"}
REF_LOGITS = {"ort_fp32": "results/w02_cert_gap_pilot/{m}/logits_ort_fp32_t8.npz", "ort_int8_dynamic": "results/w02_cert_gap_pilot/{m}/logits_ort_int8_dynamic.npz",
              "ort_int8_perchannel": "results/w02_cert_gap_pilot/{m}/logits_ort_int8_perchannel.npz",
              "ort_int8_qdq": "results/w02_contract_search_spike/{m}/logits_qdq.npz", "torch_fp32_b1": "results/w02_cert_gap_pilot/{m}/logits_torch_cpu_fp32_b1.npz"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def machine() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or \
            next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        info["cpu"] = "unknown"
    for mod in ("numpy", "onnxruntime", "torch", "transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    info["cc"] = subprocess.run(["cc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return info


# ---------------------------------------------------------------- export (reference machine)
def export() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from static_student.student import data
    import w02_cert_gap_pilot as P
    BUNDLE.mkdir(parents=True, exist_ok=True)
    manifest = {"telemetry": {}, "vendor": {}}
    for cfg in TEL:
        src, dst = REPO / "build" / cfg, BUNDLE / "telemetry" / cfg
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("ss_kernel.c", "ss_kernel.h", "ss_weights.c", "build_report.json"):
            shutil.copy2(src / f, dst / f)
        rep = json.loads((src / "build_report.json").read_text())
        trep = json.loads((REPO / rep["model"] / "train_report.json").read_text())
        split = data.split_families(data.load_families(REPO / trep["families_path"]), seed=trep["seed"])
        n = rep["certificate"]["n_calibration"]
        for pool, fam, tag in (("dev", "dev", "cert-dev"), ("cal", "defer_fit", "cert-cal")):
            rows = data.render_rows(split[fam], n, tag)
            assert all(b"\n" not in r.payload for r in rows)
            (dst / f"payloads_{pool}.bin").write_bytes(b"".join(r.payload + b"\n" for r in rows))
            with open(dst / f"gold_{pool}.jsonl", "w") as g:
                for r in rows:
                    g.write(json.dumps({"latency_us": r.latency_us, "user_id": r.user_id, "defer": bool(r.defer)}) + "\n")
        manifest["telemetry"][cfg] = {f.name: sha256(f) for f in sorted(dst.iterdir())}
        print("exported", cfg, flush=True)
    for m in VENDOR:
        _, _, pools, rev, sha = P.build_pools(m)
        dst = BUNDLE / "vendor" / m
        dst.mkdir(parents=True, exist_ok=True)
        np.savez(dst / "inputs.npz", **{f"{p}_ids": np.concatenate(v["ids"]) for p, v in pools.items()},
                 **{f"{p}_len": np.array([len(x) for x in v["ids"]]) for p, v in pools.items()})
        for path, pat in ONNX.items():
            shutil.copy2(REPO / pat.format(m=m), dst / f"{path}.onnx")
        manifest["vendor"][m] = {"model": VENDOR[m], "revision": rev, "token_ids_sha256": sha,
                                 "files": {f.name: sha256(f) for f in sorted(dst.iterdir())}}
        if m == "toxicity":
            shutil.copytree(REPO / VENDOR[m], BUNDLE / VENDOR[m], dirs_exist_ok=True)
        print("exported", m, flush=True)
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("bundle:", BUNDLE, subprocess.run(["du", "-sh", str(BUNDLE)], capture_output=True, text=True).stdout.split()[0])


# ---------------------------------------------------------------- run (any machine)
def _score_telemetry(cfg: str, dst: Path, jobs: int) -> None:
    src = BUNDLE / "telemetry" / cfg
    work = dst / "telemetry" / cfg
    work.mkdir(parents=True, exist_ok=True)
    for f in ("ss_weights.c", "ss_kernel.c"):
        subprocess.run(["cc", *CFLAGS, "-I", str(src), "-c", str(src / f), "-o", str(work / f.replace(".c", ".o"))], check=True)
    exe = work / "ss_cli"
    subprocess.run(["cc", *CFLAGS, "-I", str(src), str(REPO / "csrc/ss_cli.c"), str(work / "ss_kernel.o"), str(work / "ss_weights.o"),
                    "-o", str(exe), "-lm"], check=True)
    for pool in ("dev", "cal"):
        lines = (src / f"payloads_{pool}.bin").read_bytes().split(b"\n")[:-1]
        k = max(1, jobs)
        bounds = [round(i * len(lines) / k) for i in range(k + 1)]
        procs = [subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(k)]
        outs = []
        for p, a, b in zip(procs, bounds, bounds[1:]):  # feed all shards first, then collect, so they run in parallel
            p.stdin.write(b"".join(x + b"\n" for x in lines[a:b]))
            p.stdin.close()
        for p in procs:
            outs.append(p.stdout.read())
            assert p.wait() == 0
        (work / f"out_{pool}.txt").write_bytes(b"".join(outs))
    shutil.copy2(src / "build_report.json", work / "reference_build_report.json")


def _score_vendor(m: str, dst: Path, threads: int) -> None:
    import onnxruntime as ort
    src = BUNDLE / "vendor" / m
    z = np.load(src / "inputs.npz")
    pools = {}
    for p in ("dev", "cal", "test"):
        ids, lens = z[f"{p}_ids"], z[f"{p}_len"]
        pools[p] = np.split(ids, np.cumsum(lens)[:-1])
    work = dst / "vendor" / m
    work.mkdir(parents=True, exist_ok=True)
    for path in ONNX:
        f = work / f"logits_{path}.npz"
        if f.exists():
            continue
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        s = ort.InferenceSession(str(src / f"{path}.onnx"), so, providers=["CPUExecutionProvider"])
        t0 = time.time()
        np.savez(f, **{p: np.concatenate([s.run(["logits"], {"input_ids": x[None], "attention_mask": np.ones_like(x)[None]})[0] for x in v])
                       for p, v in pools.items()})
        print(m, path, f"{time.time() - t0:.0f}s", flush=True)
    f = work / "logits_torch_fp32_b1.npz"
    if not f.exists():
        import torch
        from transformers import AutoModelForSequenceClassification
        torch.set_num_threads(threads)
        man = json.loads((BUNDLE / "manifest.json").read_text())["vendor"][m]
        repo = VENDOR[m]
        if repo.startswith("models/"):
            model = AutoModelForSequenceClassification.from_pretrained(str(BUNDLE / repo), torch_dtype=torch.float32).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(repo, revision=man["revision"], torch_dtype=torch.float32).eval()
        t0 = time.time()
        with torch.no_grad():
            np.savez(f, **{p: np.concatenate([model(input_ids=torch.from_numpy(x)[None], attention_mask=torch.ones(1, len(x), dtype=torch.long))
                                              .logits.float().numpy() for x in v]) for p, v in pools.items()})
        print(m, "torch_fp32_b1", f"{time.time() - t0:.0f}s", flush=True)


def run(tag: str, jobs: int, threads: int, only: str | None) -> None:
    man = json.loads((BUNDLE / "manifest.json").read_text())
    bad = [f"{g}/{k}/{n}" for g, d in (("telemetry", man["telemetry"]),) for k, fs in d.items() for n, h in fs.items() if sha256(BUNDLE / g / k / n) != h]
    bad += [f"vendor/{m}/{n}" for m, v in man["vendor"].items() for n, h in v["files"].items() if sha256(BUNDLE / "vendor" / m / n) != h]
    assert not bad, f"bundle damaged in transit: {bad}"
    dst = OUT / tag
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "machine.json").write_text(json.dumps(machine(), indent=1) + "\n")
    if only in (None, "telemetry"):
        for cfg in TEL:
            t0 = time.time()
            _score_telemetry(cfg, dst, jobs)
            print(cfg, f"{time.time() - t0:.0f}s", flush=True)
    if only in (None, "vendor"):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=len(VENDOR)) as ex:
            list(ex.map(_score_vendor, list(VENDOR), [dst] * len(VENDOR), [threads] * len(VENDOR)))
    print("done", dst)


# ---------------------------------------------------------------- compare (reference machine)
def _certify_telemetry(outdir: Path, src: Path, spec: dict):
    from static_student import certify
    from static_student.student import data

    def load(pool):
        rows = (outdir / f"out_{pool}.txt").read_bytes().decode().splitlines()
        pays = (src / f"payloads_{pool}.bin").read_bytes().split(b"\n")[:-1]
        gold = [json.loads(l) for l in open(src / f"gold_{pool}.jsonl")]
        score = np.array([float(r.split()[5]) for r in rows], dtype=np.float32).astype(np.float64)
        got = [data.decode(p, np.array([int(x) for x in r.split()[:4]]), int(r.split()[4])) for p, r in zip(pays, rows)]
        emits = np.array([g is not None for g in got])
        wrong = np.array([int(g is None or gd["defer"] or g != (gd["latency_us"], gd["user_id"])) for g, gd in zip(got, gold)])
        return score, wrong, emits

    sd, _, ed = load("dev")
    sc, wc, ec = load("cal")
    grid = certify.coverage_grid(sd, ed)
    return certify.certify(sc, wc, ec, spec["alpha"], spec["delta"], spec["min_coverage"], grid)


def compare(ref: str, other: str) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from static_student import certify
    from static_student.curriculum.spec import parse_spec
    import w02_cert_gap_pilot as P
    spec = parse_spec(REPO / "specs/telemetry.spec")
    res = {"ref": json.loads((OUT / ref / "machine.json").read_text()), "other": json.loads((OUT / other / "machine.json").read_text()),
           "telemetry": {}, "vendor": {}}
    for cfg in TEL:
        a, b = OUT / ref / "telemetry" / cfg, OUT / other / "telemetry" / cfg
        if not ((a / "out_cal.txt").exists() and (b / "out_cal.txt").exists()):
            continue
        same = {p: (a / f"out_{p}.txt").read_bytes() == (b / f"out_{p}.txt").read_bytes() for p in ("dev", "cal")}
        ca, cb = (_certify_telemetry(x, BUNDLE / "telemetry" / cfg, spec) for x in (a, b))
        built = json.loads((a / "reference_build_report.json").read_text())["certificate"]
        res["telemetry"][cfg] = {"outputs_byte_identical": same, "certificate_ref": ca.as_dict(), "certificate_other": cb.as_dict(),
                                 "certificates_identical": ca.as_dict() == cb.as_dict(),
                                 "matches_shipped_record": (ca.tau, ca.n_accepted, ca.k_wrong) == (built["tau"], built["n_accepted"], built["k_wrong"])}
    for m in VENDOR:
        lab = dict(np.load(P.OUT / m / "labels.npz"))
        for path in REF_LOGITS:
            fa, fb = REPO / REF_LOGITS[path].format(m=m), OUT / other / "vendor" / m / f"logits_{path}.npz"
            if not fb.exists():
                continue
            A, B = dict(np.load(fa)), dict(np.load(fb))
            row = {"logits_byte_identical": all(np.array_equal(A[p], B[p]) for p in lab), "max_abs_logit_diff": float(max(np.abs(A[p] - B[p]).max() for p in lab)),
                   "argmax_flip_test": float(np.mean(A["test"].argmax(1) != B["test"].argmax(1))), "alphas": {}}
            for alpha in (0.05, 0.02):
                ra, rb = ({p: (*P.decisions(X[p]), lab[p]) for p in lab} for X in (A, B))
                ca, cb = P.calibrate(ra, alpha), P.calibrate(rb, alpha)
                if isinstance(ca, str):
                    row["alphas"][str(alpha)] = {"reference_certifies": False}
                    continue
                acc_a, acc_b = ra["test"][1] >= ca.tau, rb["test"][1] >= ca.tau
                row["alphas"][str(alpha)] = {
                    "accept_flip_at_ref_tau": float(np.mean(acc_a != acc_b)),
                    "certificate_ref": {"tau": ca.tau, "n": ca.n_accepted, "k": ca.k_wrong},
                    "certificate_other": None if isinstance(cb, str) else {"tau": cb.tau, "n": cb.n_accepted, "k": cb.k_wrong},
                    "certificate_differs": isinstance(cb, str) or (ca.tau, ca.n_accepted, ca.k_wrong) != (cb.tau, cb.n_accepted, cb.k_wrong),
                    "certificate_counts_differ": isinstance(cb, str) or (ca.n_accepted, ca.k_wrong) != (cb.n_accepted, cb.k_wrong)}  # H-X2 as registered
            res["vendor"].setdefault(m, {})[path] = row
    tel = res["telemetry"]
    vend = res["vendor"]
    int8 = ("ort_int8_dynamic", "ort_int8_perchannel", "ort_int8_qdq")
    hx1 = sum(any(vend.get(m, {}).get(p, {}).get("alphas", {}).get(a, {}).get("accept_flip_at_ref_tau", 0) >= 0.001 for p in int8 for a in ("0.05", "0.02")) for m in VENDOR)
    hx2 = sum(any(vend.get(m, {}).get(p, {}).get("alphas", {}).get(a, {}).get("certificate_counts_differ", False) for p in REF_LOGITS for a in ("0.05", "0.02")) for m in VENDOR)
    res["hypotheses"] = {"H-X1_models": hx1, "H-X1": hx1 >= 2, "H-X2_models": hx2, "H-X2": hx2 >= 2,
                         "H-X3_telemetry": bool(tel) and len(tel) == len(TEL) and all(v["certificates_identical"] and all(v["outputs_byte_identical"].values()) for v in tel.values())}
    (OUT / f"compare_{ref}_{other}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res["hypotheses"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("export", "run", "compare"))
    ap.add_argument("--tag", default=platform.machine())
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", choices=("telemetry", "vendor"), default=None)
    ap.add_argument("--ref")
    ap.add_argument("--other")
    a = ap.parse_args()
    if a.mode == "export":
        export()
    elif a.mode == "run":
        run(a.tag, a.jobs, a.threads, a.only)
    else:
        compare(a.ref, a.other)


if __name__ == "__main__":
    main()
