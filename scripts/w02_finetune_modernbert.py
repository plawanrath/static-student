"""Fine-tune ModernBERT for binary toxicity on Civil Comments (label: toxicity >= 0.5).

    .venv/bin/python scripts/w02_finetune_modernbert.py --size base   ->  models/modernbert-base-civil/

Training rows come from the Civil Comments train split only; the deployment-path study draws its pools from the
validation and test splits, so no evaluated comment was seen in training. One epoch, AdamW, linear decay, fp32 on the
Apple GPU. The report records the data revision, row counts, seed and wall-clock time.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

REPO = Path(__file__).resolve().parent.parent
DATA = ("google/civil_comments", "f2970eb3")
BACKBONE = {"base": ("answerdotai/ModernBERT-base", "8949b909"), "large": ("answerdotai/ModernBERT-large", "45bb4654")}
SEED = 20260926


def civil(split_files, n, seed, balance):
    df = pd.concat([pd.read_parquet(hf_hub_download(DATA[0], f, repo_type="dataset", revision=DATA[1])) for f in split_files], ignore_index=True)
    df = df[df["text"].str.len() > 0]
    df["y"] = (df["toxicity"] >= 0.5).astype(int)
    if balance:  # toxic comments are ~8% of the corpus; train on a 1:3 mix so the classifier sees enough of them
        pos = df[df.y == 1].sample(n=n // 4, random_state=seed)
        neg = df[df.y == 0].sample(n=n - n // 4, random_state=seed)
        df = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)
    else:
        df = df.sample(n=n, random_state=seed)
    return list(df["text"]), df["y"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", choices=list(BACKBONE))
    ap.add_argument("--n-train", type=int, default=40_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    repo, rev = BACKBONE[args.size]
    out = REPO / f"models/modernbert-{args.size}-civil"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev, num_labels=2, id2label={0: "non_toxic", 1: "toxic"},
                                                               label2id={"non_toxic": 0, "toxic": 1})
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev).train()
    texts, y = civil(["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"], args.n_train, SEED, balance=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(texts) // args.batch
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    t0, losses = time.time(), []
    for i in range(steps):
        b = slice(i * args.batch, (i + 1) * args.batch)
        enc = tok(texts[b], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
        loss = model(**enc, labels=torch.from_numpy(y[b]).to(dev)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        losses.append(float(loss))
        if i % 100 == 0:
            print(f"step {i}/{steps} loss {np.mean(losses[-100:]):.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval().to("cpu")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    rep = {"backbone": repo, "backbone_revision": rev, "data": f"{DATA[0]}@{DATA[1]}", "label": "toxicity >= 0.5", "train_split": "train",
           "n_train": len(texts), "positive_share": float(y.mean()), "epochs": 1, "batch": args.batch, "lr": args.lr, "max_len": args.max_len,
           "seed": SEED, "steps": steps, "final_loss_100": float(np.mean(losses[-100:])), "seconds": round(time.time() - t0, 1), "device": str(dev)}
    (out / "train_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
