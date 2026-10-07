"""Spike: choose the cheapest compiled artifact that meets a selective-risk contract, versus practitioner routes.

    .venv/bin/python scripts/w02_contract_search_spike.py                 ->  results/w02_contract_search_spike/
    .venv/bin/python scripts/w02_contract_search_spike.py --analyze-only

For each model, eight ONNX Runtime artifacts (fp32 and seven int8 recipes) score the pilot's locked 10,000 inputs once.
The first 200 inputs of the pilot's dev pool set static activation ranges and serve as latency probes, and are held out
of every split. The remaining 9,800 are re-partitioned 200 times into opt / cal / test, and four routes choose an
artifact and a threshold on opt + cal:

  B0 safe            : Learn-then-Test on the fp32 artifact, ship fp32
  B1 transplant      : fp32 threshold; artifact = cheapest whose cal accuracy >= 0.99 x fp32 (1% relative tolerance)
  B2 tune-then-cal   : same artifact choice; threshold re-derived by Learn-then-Test on the same cal pool
  ours               : Learn-then-Test at delta/K on every artifact's own scores; ship the cheapest that passes

Each route's shipped (artifact, threshold) is scored on the split's test inputs. The analysis plan and the GO rule
were fixed before any spike data existed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w02_cert_gap_pilot as P  # noqa: E402

from static_student import certify, stats  # noqa: E402

REPO = P.REPO
OUT = REPO / "results/w02_contract_search_spike"
BUILD = REPO / "build/contract_search_spike"
MODELS = ("sentiment", "injection", "toxicity")
HOLD = 200
SPLITS = 200
SIZES = (1800, 4000, 4000)
ALPHAS = (0.05, 0.02)
DELTA = 0.05
RECIPES = ("fp32", "dyn", "dyn_pc", "dyn_mm", "dyn_mm_pc", "dyn_rr", "qdq", "qdq_pc")
FROM_PILOT = {"fp32": "ort_fp32_t8", "dyn": "ort_int8_dynamic", "dyn_pc": "ort_int8_perchannel", "dyn_mm": "ort_int8_matmul_only"}


class Reader:
    """Calibration inputs for static quantization: the held-out probe inputs, one at a time."""

    def __init__(self, ids):
        self.it = iter([{"input_ids": x[None], "attention_mask": np.ones_like(x)[None]} for x in ids])

    def get_next(self):
        return next(self.it, None)


def build_recipe(recipe, model, b: Path, probe_ids) -> Path:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static
    fp32 = P.export_onnx(model, b)
    out = b / f"spike_{recipe}.onnx"
    if recipe == "fp32":
        return fp32
    if out.exists():
        return out
    if recipe.startswith("dyn"):
        quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8, per_channel=recipe.endswith("_pc"),
                         reduce_range=recipe == "dyn_rr", op_types_to_quantize=["MatMul"] if "_mm" in recipe else None)
    else:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = b / "model_fp32_pre.onnx"
        if not pre.exists():
            quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)  # symbolic inference fails on these graphs
        quantize_static(str(pre), str(out), Reader(probe_ids), quant_format=QuantFormat.QDQ, per_channel=recipe == "qdq_pc",
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)
    return out


def latency_ms(path: Path, ids, passes=3) -> float:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 8, 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feed = [{"input_ids": x[None], "attention_mask": np.ones_like(x)[None]} for x in ids]
    for f in feed[:20]:
        s.run(["logits"], f)
    t = []
    for _ in range(passes):
        for f in feed:
            t0 = time.perf_counter()
            s.run(["logits"], f)
            t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def score_model(name):
    d, b = OUT / name, BUILD / name
    d.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    tok, model, pools, rev, sha = P.build_pools(name)
    pilot_meta = json.loads((P.OUT / name / "meta.json").read_text())
    assert pilot_meta["token_ids_sha256"] == sha, "pilot pools changed"
    probe = pools["dev"]["ids"][:HOLD]
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {"model": P.MODELS[name][0], "revision": rev,
                                                                                         "token_ids_sha256": sha, "recipes": {}}
    for r in RECIPES:
        f = d / f"logits_{r}.npz"
        if r in meta["recipes"] and meta["recipes"][r].get("ok") is False:
            continue
        try:
            path = build_recipe(r, model, P.BUILD / name if r == "fp32" else b, probe)
            if not f.exists():
                src = P.OUT / name / f"logits_{FROM_PILOT[r]}.npz" if r in FROM_PILOT else None
                if src is not None and src.exists():
                    np.savez(f, **dict(np.load(src)))
                else:
                    t0 = time.time()
                    np.savez(f, **{p: P.run_ort(path, v["ids"], 8) for p, v in pools.items()})
                    print(name, r, f"scored in {time.time() - t0:.0f}s", flush=True)
            if "latency_ms" not in meta["recipes"].get(r, {}):
                meta["recipes"][r] = {"ok": True, "bytes": path.stat().st_size, "latency_ms": latency_ms(path, probe)}
                print(name, r, f"median {meta['recipes'][r]['latency_ms']:.2f} ms, {meta['recipes'][r]['bytes'] / 1e6:.1f} MB", flush=True)
        except Exception as e:  # a recipe that cannot be built is recorded and excluded
            meta["recipes"][r] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(name, r, "FAILED:", meta["recipes"][r]["error"], flush=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def ltt(score, wrong, opt, cal, alpha, delta):
    grid = certify.coverage_grid(score[opt], np.ones(len(opt), bool))
    try:
        return certify.certify(score[cal], wrong[cal], np.ones(len(cal), bool), alpha, delta, 0.0, grid).tau
    except certify.CertificationError:
        return None


def analyze(name):
    d = OUT / name
    meta = json.loads((d / "meta.json").read_text())
    lab = dict(np.load(P.OUT / name / "labels.npz"))
    y = np.concatenate([lab["dev"], lab["cal"], lab["test"]])
    ok = [r for r in RECIPES if meta["recipes"].get(r, {}).get("ok")]
    assert "fp32" in ok
    sc, wr = {}, {}
    for r in ok:
        z = dict(np.load(d / f"logits_{r}.npz"))
        pred, s = P.decisions(np.concatenate([z["dev"], z["cal"], z["test"]]))
        sc[r], wr[r] = s, (pred != y).astype(int)
    lat = {r: meta["recipes"][r]["latency_ms"] for r in ok}
    by_cost = sorted(ok, key=lambda r: lat[r])
    K = len(ok)
    pool = np.arange(HOLD, len(y))
    res = {"model": meta["model"], "recipes": {r: meta["recipes"][r] for r in RECIPES if r in meta["recipes"]}, "K": K, "splits": SPLITS, "alphas": {}}
    for alpha in ALPHAS:
        rows = {k: [] for k in ("B0", "B1", "B2", "ours")}
        for seed in range(SPLITS):
            perm = np.random.default_rng(seed).permutation(pool)
            opt, cal, test = perm[:SIZES[0]], perm[SIZES[0]:SIZES[0] + SIZES[1]], perm[SIZES[0] + SIZES[1]:]
            tau0 = ltt(sc["fp32"], wr["fp32"], opt, cal, alpha, DELTA)
            acc_fp = 1 - wr["fp32"][cal].mean()
            r1 = next(r for r in by_cost if 1 - wr[r][cal].mean() >= 0.99 * acc_fp)
            tau2 = ltt(sc[r1], wr[r1], opt, cal, alpha, DELTA)
            passing = [r for r in by_cost if ltt(sc[r], wr[r], opt, cal, alpha, DELTA / K) is not None]
            r3 = passing[0] if passing else None
            tau3 = ltt(sc[r3], wr[r3], opt, cal, alpha, DELTA / K) if r3 else None
            for route, (r, tau) in {"B0": ("fp32", tau0), "B1": (r1, tau0), "B2": (r1, tau2), "ours": (r3, tau3)}.items():
                if r is None or tau is None:
                    rows[route].append({"shipped": False})
                    continue
                acc = sc[r][test] >= tau
                n = int(acc.sum())
                risk = float(wr[r][test][acc].sum() / n) if n else 0.0
                rows[route].append({"shipped": True, "artifact": r, "latency_ms": lat[r], "coverage": float(acc.mean()), "risk": risk,
                                    "violation": bool(n and risk > alpha)})
        summ = {}
        for route, rr in rows.items():
            sh = [x for x in rr if x["shipped"]]
            v = np.array([x["violation"] for x in sh], dtype=float)
            summ[route] = {"ship_rate": len(sh) / SPLITS, "violation_rate": float(v.mean()) if len(sh) else None,
                           "violation_ci": (lambda c: [c.ci_low, c.ci_high])(stats.bootstrap_ci(v.astype(int))) if len(sh) > 1 else None,
                           "mean_coverage": float(np.mean([x["coverage"] for x in sh])) if sh else None,
                           "mean_latency_ms": float(np.mean([x["latency_ms"] for x in sh])) if sh else None,
                           "artifacts": {a: sum(x.get("artifact") == a for x in sh) for a in ok}}
        res["alphas"][str(alpha)] = {"routes": summ, "per_split": rows}
    (d / "analysis.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


def verdict(results):
    """GO rule, as pre-registered in ADR-0015."""
    hits = {}
    for name, a in results.items():
        why = []
        for alpha, v in a["alphas"].items():
            R = v["routes"]
            ours_ok = R["ours"]["violation_rate"] is not None and R["ours"]["violation_rate"] <= 0.07
            for route in ("B1", "B2"):
                b = R[route]
                if b["violation_rate"] is not None and b["violation_rate"] > 2 * DELTA and ours_ok:
                    why.append(f"{route} violates alpha={alpha} in {100 * b['violation_rate']:.1f}% of splits (ours {100 * R['ours']['violation_rate']:.1f}%)")
                if (b["mean_coverage"] is not None and R["ours"]["mean_coverage"] is not None and R["ours"]["mean_latency_ms"] is not None
                        and R["ours"]["mean_coverage"] - b["mean_coverage"] >= 0.02 and b["mean_latency_ms"] >= R["ours"]["mean_latency_ms"]):
                    why.append(f"{route} coverage {100 * b['mean_coverage']:.1f}% vs ours {100 * R['ours']['mean_coverage']:.1f}% at alpha={alpha}")
        hits[name] = why
    go = sum(bool(w) for w in hits.values()) >= 2
    return {"GO": go, "models_meeting_rule": {k: v for k, v in hits.items() if v}, "rule": "ADR-0015 spike, pre-registered 2026-09-26"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=",".join(MODELS))
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    names = args.model.split(",")
    if not args.analyze_only:
        for n in names:
            score_model(n)
    results = {}
    for n in names:
        a = analyze(n)
        results[n] = a
        for alpha, v in a["alphas"].items():
            for route, s in v["routes"].items():
                fmt = lambda x, k, u: "-" if x is None else f"{k * x:.{1 if u == '%' else 2}f}{u}"  # noqa: E731
                used = {k: c for k, c in s["artifacts"].items() if c}
                print(f"{n:9} a={alpha} {route:4} ship={100 * s['ship_rate']:.0f}% viol={fmt(s['violation_rate'], 100, '%')} "
                      f"cov={fmt(s['mean_coverage'], 100, '%')} lat={fmt(s['mean_latency_ms'], 1, 'ms')} artifacts={used}", flush=True)
    v = verdict(results)
    (OUT / "verdict.json").write_text(json.dumps(v, indent=1) + "\n")
    print("VERDICT:", json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
