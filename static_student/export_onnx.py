"""Export a student to ONNX, so that the same trained weights can be served by an engine for the deployment comparison.

    python -m static_student.export_onnx --model models/tel-mlx-S-fp --out build/onnx/tel_S_fp

The exported graph ends at the same three heads the kernel computes, so an engine baseline does exactly the work our
kernel does and no more. Padding is part of the input, as it must be for a fixed-shape engine graph; the variable
length our kernel exploits (ADR-0009) is reported separately rather than hidden in this comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from static_student.student import data
from static_student.student.model import MAX_LEN
from static_student.student.qat import load_student

REPO = Path(__file__).resolve().parent.parent


class ExportWrapper(torch.nn.Module):
    """One tensor in, three out: engines do not need our dataclasses and some of them dislike dict outputs."""

    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, ids: torch.Tensor):
        o = self.student(ids)
        return o["pointers"], o["unit"], o["defer"]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args(argv)
    mdir = REPO / args.model
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_student(mdir)
    model.eval()
    wrapper = ExportWrapper(model).eval()
    ids = torch.full((args.batch, MAX_LEN), 256, dtype=torch.int64)
    ids[:, 0] = 257
    path = out / "student.onnx"
    torch.onnx.export(wrapper, (ids,), str(path), opset_version=args.opset, input_names=["ids"],
                      output_names=["pointers", "unit", "defer"], dynamo=False)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    rows = data.render_rows(data.split_families(data.load_families(REPO / json.loads((mdir / "train_report.json").read_text())["families_path"]))["dev"], 64, "onnx")
    enc = data.encode(rows)["ids"][: args.batch * 8]
    max_dptr = max_dunit = 0.0
    for i in range(0, len(enc), args.batch):
        chunk = enc[i:i + args.batch]
        if len(chunk) < args.batch:
            break
        got = sess.run(None, {"ids": chunk.astype(np.int64)})
        with torch.no_grad():
            want = wrapper(torch.from_numpy(chunk))
        max_dptr = max(max_dptr, float(np.abs(got[0] - want[0].numpy()).max()))
        max_dunit = max(max_dunit, float(np.abs(got[1] - want[1].numpy()).max()))
    report = {"model": args.model, "bits": ck.get("bits"), "opset": args.opset, "batch": args.batch,
              "onnx_bytes": path.stat().st_size, "max_abs_delta_pointers_vs_torch": max_dptr, "max_abs_delta_unit_vs_torch": max_dunit,
              "onnxruntime": ort.__version__}
    (out / "export_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
