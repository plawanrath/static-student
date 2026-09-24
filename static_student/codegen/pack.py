"""A trained student -> the constant array, in the order the kernel reads it.

One `Packed` holds every tensor the kernel needs, already in its shipped precision: trunk weights as int8 levels plus
float16 group scales (exactly what QAT trained), embeddings and head matrices as int8 with one float32 scale per row,
LayerNorm parameters and biases as float32. The same object drives the NumPy oracle and the C emitter, so the layout
cannot drift between them.

Embeddings and heads are quantized here, after training, not during it. That is a deliberate difference between the
float model and the bits that ship, and it is one of the reasons the contract is certified on the kernel's own
outputs rather than on the float model's.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import torch

from static_student.codegen import kernel_math as km
from static_student.student import quant
from static_student.student.model import POINTERS, UNIT_CLASSES, Student

F32, I32 = np.float32, np.int32


def quantize_rows(w: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric with one scale per output row: w[o, :] ~ levels[o, :] * scale[o]. The emitter stores these levels in
    one byte each, so anything wider than 8 bits is a reference-only experiment and `Blob.i8` refuses to ship it."""
    qmax = F32(2 ** (bits - 1) - 1)
    scale = (np.abs(w).max(axis=1).astype(F32) / qmax).astype(F32)
    scale = np.maximum(scale, F32(1e-12)).astype(F32)
    levels = np.clip(km.round_to_int(w / scale[:, None]), -qmax, qmax).astype(I32)
    return levels, scale


@dataclass
class QuantLinear:
    levels: np.ndarray   # (out, in) int32 holding int8-range values
    scales: np.ndarray   # (out, in/GROUP) float32, cast from the float16 that ships
    bias: np.ndarray     # (out,) float32
    bits: int
    group: int = quant.GROUP


@dataclass
class RowLinear:
    levels: np.ndarray   # (out, in) int32
    scale: np.ndarray    # (out,) float32
    bias: np.ndarray | None


@dataclass
class Packed:
    config: dict
    bits: int
    tok: RowLinear
    pos: RowLinear
    ln_in: tuple[np.ndarray, np.ndarray]
    blocks: list[dict] = field(default_factory=list)
    pointers: RowLinear | None = None
    unit: RowLinear | None = None
    defer: RowLinear | None = None

    def bytes(self) -> dict[str, int]:
        trunk = sum((l.levels.size * self.bits + 7) // 8 + 2 * l.scales.size + 4 * l.bias.size
                    for b in self.blocks for l in b.values() if isinstance(l, QuantLinear))
        rows = sum(r.levels.size + 4 * r.scale.size + (4 * r.bias.size if r.bias is not None else 0)
                   for r in (self.tok, self.pos, self.pointers, self.unit, self.defer))
        norms = sum(4 * (w.size + b.size) for w, b in [self.ln_in] + [x for blk in self.blocks for x in (blk["ln1"], blk["ln2"])])
        return {"trunk": trunk, "embeddings_and_heads": rows, "norms": norms, "total": trunk + rows + norms}

    def sha256(self) -> str:
        h = hashlib.sha256()
        for name, arr in sorted(self.arrays().items()):
            h.update(name.encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, r in (("tok", self.tok), ("pos", self.pos), ("pointers", self.pointers), ("unit", self.unit), ("defer", self.defer)):
            out[f"{name}.levels"], out[f"{name}.scale"] = r.levels.astype(np.int8), r.scale
            if r.bias is not None:
                out[f"{name}.bias"] = r.bias
        out["ln_in.weight"], out["ln_in.bias"] = self.ln_in
        for i, blk in enumerate(self.blocks):
            for key, v in blk.items():
                if isinstance(v, QuantLinear):
                    out[f"b{i}.{key}.levels"], out[f"b{i}.{key}.scales"], out[f"b{i}.{key}.bias"] = v.levels.astype(np.int8), v.scales, v.bias
                else:
                    out[f"b{i}.{key}.weight"], out[f"b{i}.{key}.bias"] = v
        return out

    def manifest(self) -> dict:
        return {"config": self.config, "bits": self.bits, "group": quant.GROUP, "bytes": self.bytes(), "sha256": self.sha256(),
                "pointers": list(POINTERS), "unit_classes": list(UNIT_CLASSES)}


def _qlinear(m: quant.QLinear) -> QuantLinear:
    levels, scales = m.integer_weights()
    return QuantLinear(levels.numpy().astype(I32), scales.numpy().astype(F32), m.bias.detach().numpy().astype(F32), m.bits)


def _row(w: torch.Tensor, b: torch.Tensor | None) -> RowLinear:
    levels, scale = quantize_rows(w.detach().numpy().astype(F32))
    return RowLinear(levels, scale, None if b is None else b.detach().numpy().astype(F32))


def pack(model: Student) -> Packed:
    """The model must already carry its shipped step sizes (`snap_scales`), which `qat.py` does before saving."""
    model = model.to("cpu").eval()
    ln = lambda m: (m.weight.detach().numpy().astype(F32), m.bias.detach().numpy().astype(F32))  # noqa: E731
    bits = next((m.bits for m in model.modules() if isinstance(m, quant.QLinear)), 16)
    p = Packed(config=model.config.as_dict(), bits=bits, tok=_row(model.tok.weight, None), pos=_row(model.pos.weight, None), ln_in=ln(model.ln))
    for blk in model.blocks:
        entry: dict = {"ln1": ln(blk.ln1), "ln2": ln(blk.ln2)}
        for name in quant.TRUNK_LINEARS:
            m = getattr(blk, name)
            entry[name] = _qlinear(m) if isinstance(m, quant.QLinear) else QuantLinear(
                *(lambda lv, sc: (lv, sc[:, None].repeat(m.weight.shape[1] // quant.GROUP, 1)))(*quantize_rows(m.weight.detach().numpy().astype(F32))),
                m.bias.detach().numpy().astype(F32), 8)
        p.blocks.append(entry)
    p.pointers, p.unit, p.defer = (_row(h.weight, h.bias) for h in (model.pointers, model.unit, model.defer))
    return p


def save_manifest(p: Packed, path) -> None:
    from pathlib import Path
    Path(path).write_text(json.dumps(p.manifest(), indent=1) + "\n")
