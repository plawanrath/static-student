"""Pilot: does a threshold calibrated on the framework model hold on the path that executes, for public classifiers?

    .venv/bin/python scripts/w02_cert_gap_pilot.py --model injection     ->  results/w02_cert_gap_pilot/<model>/
    .venv/bin/python scripts/w02_cert_gap_pilot.py --model all --analyze-only

Each model is tokenized once, so every path sees identical token ids and only the execution of the model differs.
Paths: PyTorch fp32 on the CPU at batch 1 (the reference, where the threshold is calibrated) and at batch 32 with
padding; PyTorch fp16 on the Apple GPU at batch 32; ONNX Runtime fp32 at 1 and 8 threads; ONNX Runtime with dynamic
int8 quantization; Core ML fp16. The decision is the argmax class, its score is the softmax probability, and a threshold
is chosen by Learn-then-Test exactly as the compiler does it (grid from the dev pool, test on the calibration pool).
Scores are cached per path, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from static_student import certify, stats

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/w02_cert_gap_pilot"
BUILD = REPO / "build/cert_gap_pilot"  # exported and converted models (large, not tracked)
SEED = 20260926
POOLS = (("dev", 2000), ("cal", 4000), ("test", 4000))
MAX_LEN = 256
ALPHAS = (0.05, 0.02)
DELTA = 0.05
DEFAULT_PATHS = ("ort_int8_dynamic", "torch_mps_fp16_b32", "coreml_fp16")


def _csv(repo, name, rev):
    return pd.read_csv(hf_hub_download(repo, name, repo_type="dataset", revision=rev))


def _parquet(repo, names, rev):
    return pd.concat([pd.read_parquet(hf_hub_download(repo, n, repo_type="dataset", revision=rev)) for n in names], ignore_index=True)


def data_injection(cfg):
    df = _csv("reshabhs/SPML_Chatbot_Prompt_Injection", "spml_prompt_injection.csv", "02ce8084")
    df = df.dropna(subset=["User Prompt"]).drop_duplicates(subset=["User Prompt"])
    pos = cfg.label2id["INJECTION"]
    return list(df["User Prompt"].astype(str)), None, np.where(df["Prompt injection"].to_numpy() == 1, pos, 1 - pos)


def data_sentiment(cfg):
    df = _parquet("cornell-movie-review-data/rotten_tomatoes", ["train.parquet", "validation.parquet", "test.parquet"], "aa13bc28")
    pos = cfg.label2id["POSITIVE"]
    return list(df["text"].astype(str)), None, np.where(df["label"].to_numpy() == 1, pos, 1 - pos)


def data_zeroshot(cfg):
    df = _parquet("fancyzhx/ag_news", ["data/test-00000-of-00001.parquet", "data/train-00000-of-00001.parquet"], "eb185aad")
    df = df.sample(n=20_000, random_state=SEED).reset_index(drop=True)
    ent = cfg.label2id["entailment"]
    return list(df["text"].astype(str)), "This example is about sports.", np.where(df["label"].to_numpy() == 1, ent, 1 - ent)


def data_toxicity(cfg):
    files = ["data/validation-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]  # never used in fine-tuning
    df = _parquet("google/civil_comments", files, "f2970eb3")
    df = df[df["text"].str.len() > 0].sample(n=20_000, random_state=SEED).reset_index(drop=True)
    return list(df["text"].astype(str)), None, (df["toxicity"].to_numpy() >= 0.5).astype(int) * cfg.label2id["toxic"] + \
        (df["toxicity"].to_numpy() < 0.5).astype(int) * cfg.label2id["non_toxic"]


MODELS = {
    "injection": ("protectai/deberta-v3-base-prompt-injection-v2", "90c9989b", data_injection),
    "sentiment": ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "714eb0fa", data_sentiment),
    "zeroshot": ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "cf44676c", data_zeroshot),
    "toxicity": ("models/modernbert-base-civil", None, data_toxicity),  # control: fine-tuned by us, scripts/w02_finetune_modernbert.py
    "toxicity_large": ("models/modernbert-large-civil", None, data_toxicity),  # the compiled 395M model (ADR-0016)
}


def full_rev(repo, short):
    for m in OUT.glob("*/meta.json"):  # resolved once already: no network needed
        meta = json.loads(m.read_text())
        if meta.get("model") == repo and str(meta.get("revision", "")).startswith(short):
            return meta["revision"]
    from huggingface_hub import HfApi
    for attempt in range(5):
        try:
            for c in HfApi().list_repo_commits(repo):
                if c.commit_id.startswith(short):
                    return c.commit_id
            raise ValueError(f"{repo}: no commit {short}")
        except ValueError:
            raise
        except Exception:
            if attempt == 4:
                raise
            time.sleep(30 * (attempt + 1))


def build_pools(name):
    repo, rev, loader = MODELS[name]
    if rev is None:  # a model we trained: load from disk, identify by the hash of its weights
        repo = str(REPO / repo)
        rev = "local:" + hashlib.sha256((Path(repo) / "model.safetensors").read_bytes()).hexdigest()[:16]
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForSequenceClassification.from_pretrained(repo, torch_dtype=torch.float32).eval()
    else:
        rev = full_rev(repo, rev)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev, torch_dtype=torch.float32).eval()
    texts, hyp, labels = loader(model.config)
    need = sum(n for _, n in POOLS)
    assert len(texts) >= need, f"{name}: {len(texts)} rows < {need}"
    order = np.random.default_rng(SEED).permutation(len(texts))
    pools, start = {}, 0
    for pool, n in POOLS:
        idx = order[start:start + n]
        start += n
        enc = tok([texts[i] for i in idx], [hyp] * n if hyp else None, truncation=True, max_length=MAX_LEN)
        pools[pool] = {"ids": [np.array(x, dtype=np.int64) for x in enc["input_ids"]], "label": labels[idx], "rows": idx}
    sha = hashlib.sha256(b"".join(np.concatenate(pools[p]["ids"]).tobytes() for p, _ in POOLS)).hexdigest()
    return tok, model, pools, rev, sha


class Logits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def run_torch(model, ids, pad_id, batch, device, dtype):
    m = model.to(device=device, dtype=dtype)
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            L = max(len(x) for x in chunk)
            x = np.full((len(chunk), L), pad_id, dtype=np.int64)
            a = np.zeros((len(chunk), L), dtype=np.int64)
            for j, s in enumerate(chunk):
                x[j, :len(s)], a[j, :len(s)] = s, 1
            out.append(m(input_ids=torch.from_numpy(x).to(device), attention_mask=torch.from_numpy(a).to(device)).logits.float().cpu().numpy())
    model.to(device="cpu", dtype=torch.float32)
    return np.concatenate(out)


def export_onnx(model, d: Path) -> Path:
    fp32 = d / "model_fp32.onnx"
    if not fp32.exists():
        x = torch.ones(1, 16, dtype=torch.int64)
        torch.onnx.export(Logits(model).eval(), (x, torch.ones_like(x)), str(fp32), input_names=["input_ids", "attention_mask"],
                          output_names=["logits"], dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}},
                          opset_version=17, dynamo=False)
    return fp32


def run_ort(path: Path, ids, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(["logits"], {"input_ids": x[None], "attention_mask": np.ones_like(x)[None]})[0] for x in ids])


def _register_coreml_ops():
    """transformers builds its attention mask with `new_ones`, which the Core ML PyTorch frontend does not convert."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
    if "new_ones" in _TORCH_OPS_REGISTRY:
        return

    @register_torch_op(torch_alias=["int"], override=True)
    def _int(context, node):  # ModernBERT calls int() on a one-element tensor; the stock converter needs a 0-d value
        x = _get_inputs(context, node, expected=1)[0]
        if getattr(x, "val", None) is not None:
            context.add(mb.const(val=np.int32(np.asarray(x.val).reshape(-1)[0]), name=node.name))
        else:
            context.add(mb.cast(x=mb.reshape(x=x, shape=[]) if x.rank > 0 else x, dtype="int32", name=node.name))

    @register_torch_op
    def new_ones(context, node):
        inputs = _get_inputs(context, node)
        shape = inputs[1]
        if isinstance(shape, (list, tuple)):
            shape = mb.concat(values=[mb.reshape(x=mb.cast(x=v, dtype="int32") if hasattr(v, "op") else mb.const(val=[int(v)]), shape=[1])
                                      for v in shape], axis=0)
        elif getattr(shape, "val", None) is not None:
            static = [int(v) for v in np.asarray(shape.val).reshape(-1)]
            if not static:  # new_ones(()) is a scalar one
                shape = None
            else:
                shape = mb.const(val=np.array(static, dtype=np.int32))
        else:
            shape = mb.cast(x=shape, dtype="int32")
        code = getattr(inputs[2], "val", None) if len(inputs) > 2 and inputs[2] is not None else None
        target = {11: "bool", 3: "int32", 4: "int32"}.get(int(code) if code is not None else -1)  # torch ScalarType codes
        ones = mb.const(val=np.float32(1.0)) if shape is None else mb.fill(shape=shape, value=1.0)
        context.add(mb.cast(x=ones, dtype=target, name=node.name) if target else mb.identity(x=ones, name=node.name))


def run_coreml(model, d: Path, ids):
    import coremltools as ct
    _register_coreml_ops()
    pkg = d / "model_fp16.mlpackage"
    if not pkg.exists():
        x = torch.ones(1, 16, dtype=torch.int64)
        traced = torch.jit.trace(Logits(model).eval(), (x, torch.ones_like(x)))
        shape = ct.Shape(shape=(1, ct.RangeDim(1, MAX_LEN, default=16)))
        ml = ct.convert(traced, inputs=[ct.TensorType(name="input_ids", shape=shape, dtype=np.int32),
                                        ct.TensorType(name="attention_mask", shape=shape, dtype=np.int32)],
                        compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.ALL, minimum_deployment_target=ct.target.macOS15)
        ml.save(str(pkg))
    ml = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.ALL)
    key = list(ml.get_spec().description.output)[0].name
    return np.concatenate([np.asarray(ml.predict({"input_ids": x[None].astype(np.int32), "attention_mask": np.ones((1, len(x)), np.int32)})[key], dtype=np.float32).reshape(1, -1) for x in ids])


def score_paths(name, only=None):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    tok, model, pools, rev, sha = build_pools(name)
    meta = {"model": MODELS[name][0], "revision": rev, "pools": {p: len(v["label"]) for p, v in pools.items()}, "token_ids_sha256": sha,
            "seed": SEED, "max_len": MAX_LEN, "label_rate": {p: float(np.bincount(v["label"], minlength=2).max() / len(v["label"])) for p, v in pools.items()},
            "paths": {}}
    old = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    assert not old or old["token_ids_sha256"] == sha, "pools changed since scores were cached"
    meta["paths"] = old.get("paths", {})
    np.savez(d / "labels.npz", **{p: v["label"] for p, v in pools.items()})
    pad = tok.pad_token_id
    b = BUILD / name
    b.mkdir(parents=True, exist_ok=True)
    runners = {
        "torch_cpu_fp32_b1": lambda ids: run_torch(model, ids, pad, 1, "cpu", torch.float32),
        "torch_cpu_fp32_b32": lambda ids: run_torch(model, ids, pad, 32, "cpu", torch.float32),
        "torch_mps_fp16_b32": lambda ids: run_torch(model, ids, pad, 32, "mps", torch.float16),
        "ort_fp32_t1": lambda ids: run_ort(export_onnx(model, b), ids, 1),
        "ort_fp32_t8": lambda ids: run_ort(export_onnx(model, b), ids, 8),
        "ort_int8_dynamic": lambda ids: run_ort(quantized(model, b), ids, 8),
        "coreml_fp16": lambda ids: run_coreml(model, b, ids),
        # post-hoc (added 2026-09-26 after the default recipe collapsed one model; not part of the GO rule):
        "ort_int8_perchannel": lambda ids: run_ort(quantized(model, b, per_channel=True), ids, 8),
        "ort_int8_matmul_only": lambda ids: run_ort(quantized(model, b, ops=["MatMul"]), ids, 8),
    }
    for path, fn in runners.items():
        if only and path not in only:
            continue
        f = d / f"logits_{path}.npz"
        if f.exists():
            continue
        t0 = time.time()
        try:
            res = {p: fn(v["ids"]) for p, v in pools.items()}
        except Exception as e:  # a path that cannot be built is recorded, not hidden
            meta["paths"][path] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}
            print(name, path, "FAILED:", meta["paths"][path]["error"], flush=True)
            (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
            continue
        np.savez(f, **res)
        meta["paths"][path] = {"ok": True, "seconds": round(time.time() - t0, 1)}
        print(name, path, f"{time.time() - t0:.0f}s", flush=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def quantized(model, d: Path, per_channel: bool = False, ops=None) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = d / ("model_int8_dynamic" + ("_perchannel" if per_channel else "") + ("_" + "_".join(ops) if ops else "") + ".onnx")
    if not q.exists():
        quantize_dynamic(str(export_onnx(model, d)), str(q), weight_type=QuantType.QInt8, per_channel=per_channel, op_types_to_quantize=ops)
    return q


def decisions(logits):
    z = logits.astype(np.float64)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return p.argmax(1), p.max(1)


def calibrate(sc, alpha):
    (pd_, sd, yd), (pc, s_c, yc) = sc["dev"], sc["cal"]
    grid = certify.coverage_grid(sd, np.ones_like(sd, bool))
    try:
        return certify.certify(s_c, (pc != yc).astype(int), np.ones_like(s_c, bool), alpha, DELTA, 0.0, grid)
    except certify.CertificationError as e:
        return str(e)


def analyze(name):
    d = OUT / name
    meta = json.loads((d / "meta.json").read_text())
    lab = dict(np.load(d / "labels.npz"))
    paths = {f.stem[len("logits_"):]: dict(np.load(f)) for f in sorted(d.glob("logits_*.npz"))}
    ref = "torch_cpu_fp32_b1"
    assert ref in paths, f"{name}: reference path missing"
    sc = {p: {pool: (*decisions(v[pool]), lab[pool]) for pool in lab} for p, v in paths.items()}
    res = {"model": meta["model"], "revision": meta["revision"], "pools": meta["pools"], "label_rate": meta["label_rate"],
           "failed_paths": {p: v["error"] for p, v in meta["paths"].items() if not v.get("ok")}, "alphas": {}}
    for alpha in ALPHAS:
        rc = calibrate(sc[ref], alpha)
        if isinstance(rc, str):
            res["alphas"][str(alpha)] = {"reference_calibrated": False, "reason": rc}
            continue
        pr, sr, yr = sc[ref]["test"]
        acc_ref = sr >= rc.tau
        out = {"reference_calibrated": True, "tau_ref": rc.tau, "ref_cal_coverage": rc.coverage, "paths": {}}
        for p in paths:
            pp, sp, yp = sc[p]["test"]
            acc = sp >= rc.tau
            wrong = (pp != yp).astype(int)
            r = stats.selective_risk_ci(wrong, acc.astype(int))
            own = calibrate(sc[p], alpha)
            row = {"accept_flip": float(np.mean(acc != acc_ref)), "argmax_flip": float(np.mean(pp != pr)),
                   "max_abs_prob_diff": float(np.max(np.abs(sp - sr))), "risk_at_tau_ref": r, "risk_ci_above_alpha": bool(r["risk_ci_low"] > alpha)}
            if isinstance(own, str):
                row["recalibrated"] = {"ok": False, "reason": own}
            else:
                acc_own = sp >= own.tau
                dc = stats.paired_bootstrap_diff(acc.astype(int), acc_own.astype(int))
                row["recalibrated"] = {"ok": True, "tau": own.tau, "coverage": float(acc_own.mean()),
                                       "delta_cov_tau_ref_minus_own": {"point": dc.point, "ci_low": dc.ci_low, "ci_high": dc.ci_high}}
            row["gap"] = bool(row["risk_ci_above_alpha"] or row["accept_flip"] >= 0.01 or
                              (row["recalibrated"].get("ok") and abs(row["recalibrated"]["delta_cov_tau_ref_minus_own"]["point"]) >= 0.02))
            out["paths"][p] = row
        res["alphas"][str(alpha)] = out
    (d / "analysis.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="injection, sentiment, zeroshot, or all")
    ap.add_argument("--paths", default=None, help="comma-separated subset of paths to score")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    names = list(MODELS) if args.model == "all" else args.model.split(",")
    torch.set_num_threads(8)
    for n in names:
        if not args.analyze_only:
            score_paths(n, set(args.paths.split(",")) if args.paths else None)
        if (OUT / n / "logits_torch_cpu_fp32_b1.npz").exists():
            a = analyze(n)
            for alpha, v in a["alphas"].items():
                if not v.get("reference_calibrated"):
                    print(n, alpha, "reference does not calibrate:", v["reason"])
                    continue
                for p, r in v["paths"].items():
                    rr = r["risk_at_tau_ref"]
                    dc = r["recalibrated"].get("delta_cov_tau_ref_minus_own", {}).get("point", float("nan"))
                    print(f"{n:9} a={alpha} {p:20} flip={100 * r['accept_flip']:.2f}% argmax={100 * r['argmax_flip']:.2f}% "
                          f"risk={100 * rr['risk']:.2f} [{100 * rr['risk_ci_low']:.2f},{100 * rr['risk_ci_high']:.2f}] cov={100 * rr['coverage']:.1f} "
                          f"dcov={100 * dc:.2f}pp gap={r['gap']}")
    summary = {n: json.loads((OUT / n / "analysis.json").read_text()) for n in MODELS if (OUT / n / "analysis.json").exists()}
    go_models = [n for n, a in summary.items() if any(r["gap"] for p, r in a["alphas"].get("0.05", {}).get("paths", {}).items() if p in DEFAULT_PATHS)]
    (OUT / "summary.json").write_text(json.dumps({"models": list(summary), "default_paths": DEFAULT_PATHS,
                                                  "models_with_gap_on_a_default_path_at_alpha_0.05": go_models,
                                                  "GO": len(go_models) >= 2}, indent=1) + "\n")
    print("models with a gap on a default path (alpha 0.05):", go_models)


if __name__ == "__main__":
    main()
