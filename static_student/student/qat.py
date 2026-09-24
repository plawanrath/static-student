"""One QAT stage: start from a trained student (float, or a wider QAT stage), quantize the trunk to `bits`, and fine-tune
with the task loss plus distillation from the float student's own logits (the 24B teacher never produces logits).

    python -m static_student.student.qat --init models/<fp or wider stage> --teacher models/<fp> --bits 4 --out models/<name>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from static_student.student import data, quant
from static_student.student.model import Student, StudentConfig
from static_student.student.train import device, predict, structs, wrong_if_accepted


def load_student(path: Path) -> tuple[Student, dict]:
    ck = torch.load(path / "student.pt" if (path / "student.pt").exists() else path / "student_fp.pt", map_location="cpu")
    m = Student(StudentConfig(**ck["config"]))
    if ck.get("bits"):
        quant.to_qat(m, ck["bits"])
    m.load_state_dict(ck["state"])
    return m, ck


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--teacher", required=True, help="the float student whose logits are distilled")
    ap.add_argument("--bits", type=int, required=True, choices=[2, 3, 4])
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-rows", type=int, default=500_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kd", type=float, default=1.0)
    args = ap.parse_args(argv)

    dev, t0 = device(), time.time()
    rep = json.loads((Path(args.teacher) / "train_report.json").read_text())
    torch.manual_seed(rep["seed"])
    split = data.split_families(data.load_families(rep["families_path"]), seed=rep["seed"])
    rows_dev = data.render_rows(split["dev"], 20_000, f"dev:{rep['seed']}")
    tr = {k: torch.from_numpy(v) for k, v in data.encode(data.render_rows(split["train"], args.train_rows, f"qat:{args.bits}:{rep['seed']}")).items()}
    enc_dev = data.encode(rows_dev)

    fp, _ = load_student(Path(args.teacher))
    fp.to(dev).eval()
    model, _ = load_student(Path(args.init))
    model = quant.to_qat(model, args.bits).to(dev)

    def dev_exact(m) -> float:
        p = predict(m, enc_dev["ids"], dev)
        w = wrong_if_accepted(rows_dev, structs(rows_dev, p))
        return float(1 - w[[not r.defer for r in rows_dev]].mean())

    log = [{"epoch": -1, "dev_struct_exact": round(dev_exact(model), 5), "note": "after quantization, before fine-tuning"}]
    print(f"[qat] {args.bits}-bit, rendered in {time.time() - t0:.0f}s; float {dev_exact(fp):.4f}, quantized untrained {log[0]['dev_struct_exact']:.4f}", flush=True)
    steps = args.epochs * (args.train_rows // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(args.train_rows)
        for i in range(0, args.train_rows - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            ids = tr["ids"][idx].to(dev)
            o = model(ids)
            with torch.no_grad():
                t = fp(ids)
            task = F.cross_entropy(o["pointers"].reshape(-1, o["pointers"].shape[-1]), tr["ptr"][idx].reshape(-1).to(dev)) \
                + F.cross_entropy(o["unit"], tr["unit"][idx].to(dev), ignore_index=-100)
            kd = sum(F.kl_div(F.log_softmax(o[k].float(), -1), F.softmax(t[k].float(), -1), reduction="batchmean") for k in ("pointers", "unit")) / o["pointers"].shape[1]
            loss = task + args.kd * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step()
            step += 1
            if step % 500 == 0:
                print(f"[qat] step {step}/{steps} task {task.item():.4f} kd {kd.item():.4f} {time.time() - t0:.0f}s", flush=True)
        log.append({"epoch": ep, "dev_struct_exact": round(dev_exact(model), 5)})
        print(f"[qat] epoch {ep} dev exact {log[-1]['dev_struct_exact']:.4f}", flush=True)
        torch.save({"config": model.config.as_dict(), "bits": args.bits, "state": {k: v.cpu() for k, v in model.state_dict().items()}, "epoch": ep}, out / "checkpoint_last.pt")
    for m in model.modules():
        if isinstance(m, quant.QLinear):
            m.snap_scales()
    log.append({"epoch": "snapped", "dev_struct_exact": round(dev_exact(model), 5), "note": "step sizes rounded to float16, as shipped"})
    torch.save({"config": model.config.as_dict(), "bits": args.bits, "state": {k: v.cpu() for k, v in model.state_dict().items()}}, out / "student.pt")
    report = {**{k: rep[k] for k in ("size", "families_path", "seed")}, "bits": args.bits, "init": args.init, "teacher": args.teacher, "log": log,
              "bytes": quant.weight_bytes(model), "train_rows": args.train_rows, "epochs": args.epochs, "seconds": round(time.time() - t0, 1)}
    (out / "train_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
