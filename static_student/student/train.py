"""Train a float student in two phases (ADR: pointers and unit first; then the defer head on the frozen trunk, fitted to
the student's own errors on families it never trained on plus the curriculum's should-defer lines).

    python -m static_student.student.train --families data/curriculum/<run>/families.jsonl --size XS --out models/<name>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from static_student.student import data
from static_student.student.model import SIZES, Student


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model: Student, ids: np.ndarray, dev: torch.device, batch: int = 1024) -> dict[str, np.ndarray]:
    model.eval()
    out = {"ptr": [], "unit": [], "defer": [], "maxprob": []}
    for i in range(0, len(ids), batch):
        o = model(torch.from_numpy(ids[i:i + batch]).to(dev))
        lp = F.log_softmax(o["pointers"].float(), dim=-1)
        out["ptr"].append(lp.argmax(-1).cpu().numpy())
        out["unit"].append(o["unit"].argmax(-1).cpu().numpy())
        out["defer"].append(o["defer"].float().cpu().numpy())
        out["maxprob"].append((lp.max(-1).values.sum(-1) + F.log_softmax(o["unit"].float(), -1).max(-1).values).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def structs(rows, pred) -> list:
    return [data.decode(r.payload, pred["ptr"][i], int(pred["unit"][i])) for i, r in enumerate(rows)]


def wrong_if_accepted(rows, got) -> np.ndarray:
    return np.array([int(g is None or r.defer or g != (r.latency_us, r.user_id)) for r, g in zip(rows, got)])


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--size", default="XS", choices=list(SIZES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-rows", type=int, default=400_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    dev, t0 = device(), time.time()
    split = data.split_families(data.load_families(args.families), seed=args.seed)
    rows = {"train": data.render_rows(split["train"], args.train_rows, f"train:{args.seed}"),
            "dev": data.render_rows(split["dev"], 20_000, f"dev:{args.seed}"),
            "defer_fit": data.render_rows(split["defer_fit"], 40_000, f"defer_fit:{args.seed}")}
    enc = {k: data.encode(v) for k, v in rows.items()}
    print(f"[train] families train/dev/defer_fit = {[len(split[k]) for k in ('train', 'dev', 'defer_fit')]}, rendered in {time.time() - t0:.0f}s, device {dev}", flush=True)

    model = Student(SIZES[args.size]).to(dev)
    steps = args.epochs * (args.train_rows // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    tr = {k: torch.from_numpy(v) for k, v in enc["train"].items()}
    log, step = [], 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(args.train_rows)
        for i in range(0, args.train_rows - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            o = model(tr["ids"][idx].to(dev))
            loss = F.cross_entropy(o["pointers"].reshape(-1, o["pointers"].shape[-1]), tr["ptr"][idx].reshape(-1).to(dev)) \
                + F.cross_entropy(o["unit"], tr["unit"][idx].to(dev), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step()
            step += 1
            if step % 500 == 0:
                print(f"[train] step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        p = predict(model, enc["dev"]["ids"], dev)
        exact = 1 - wrong_if_accepted(rows["dev"], structs(rows["dev"], p))[[not r.defer for r in rows["dev"]]].mean()
        log.append({"epoch": ep, "dev_struct_exact": round(float(exact), 5)})
        print(f"[train] epoch {ep} dev exact on struct rows {exact:.4f}", flush=True)
        ckpt = Path(args.out)  # a run interrupted by a shutdown resumes from the last finished epoch's weights
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save({"config": SIZES[args.size].as_dict(), "state": {k: v.cpu() for k, v in model.state_dict().items()}, "epoch": ep, "log": log}, ckpt / "checkpoint_last.pt")

    # Phase 2: defer head on the frozen trunk. Target = 1 where accepting would be wrong, on families the trunk never saw.
    for p_ in model.parameters():
        p_.requires_grad_(False)
    for p_ in model.defer.parameters():
        p_.requires_grad_(True)
    pf = predict(model, enc["defer_fit"]["ids"], dev)
    target = torch.from_numpy(wrong_if_accepted(rows["defer_fit"], structs(rows["defer_fit"], pf)).astype(np.float32))
    with torch.no_grad():
        model.eval()
        cls = torch.cat([model.trunk(torch.from_numpy(enc["defer_fit"]["ids"][i:i + 1024]).to(dev))[:, 0].cpu() for i in range(0, len(target), 1024)])
    head = model.defer.cpu()
    opt2 = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(300):
        loss2 = F.binary_cross_entropy_with_logits(head(cls).squeeze(-1), target)
        opt2.zero_grad()
        loss2.backward()
        opt2.step()
    model.defer = head.to(dev)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": SIZES[args.size].as_dict(), "state": {k: v.cpu() for k, v in model.state_dict().items()}}, out / "student_fp.pt")
    report = {"size": args.size, "families_path": args.families, "params": model.n_params(), "families": {k: len(v) for k, v in split.items()}, "train_rows": args.train_rows,
              "epochs": args.epochs, "seed": args.seed, "log": log, "defer_fit_error_rate": round(float(target.mean()), 4),
              "defer_head_bce": round(float(loss2.item()), 4), "seconds": round(time.time() - t0, 1), "device": str(dev)}
    (out / "train_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
