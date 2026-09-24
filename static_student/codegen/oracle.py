"""The integer reference: what the generated C kernel computes, in NumPy, using only `kernel_math` primitives.

`forward` takes packed weights and a batch of byte sequences and returns the four pointer logit rows, the unit logits
and the defer logit. It is the bit-exactness target for the emitted C (gate G1a) and the model whose outputs the
certificate is computed on (C2). It is not a fast implementation and does not try to be.
"""
from __future__ import annotations

import numpy as np

from static_student.codegen import kernel_math as km
from static_student.codegen.pack import Packed, QuantLinear, RowLinear
from static_student.student.model import MAX_LEN, PAD
from static_student.student.quant import GROUP

F32 = np.float32
NEG = F32(-1e4)


def inv_sqrt_head_dim(hd: int) -> F32:
    """The attention scale, defined in exactly one place. `emit.py` writes this same value into the generated C: it is
    a one-unit-in-the-last-place difference at head_dim 24, and it is enough to break bit-exactness."""
    return F32(1.0 / np.sqrt(hd))


def _linear(x: np.ndarray, l: QuantLinear) -> np.ndarray:
    xq, s = km.quantize_activation(x)          # s is per token: (..., 1)
    y = km.dot_groups(xq, l.levels, l.scales, GROUP)
    return (y * s + l.bias).astype(F32)


def _row_apply(x: np.ndarray, r: RowLinear) -> np.ndarray:
    """Head matrices ship as int8 rows with one scale each; the activation stays in float32 here, as in the kernel."""
    y = km.dot_f32(x[..., None, :], (r.levels.astype(F32) * r.scale[:, None]))
    return (y + r.bias).astype(F32) if r.bias is not None else y.astype(F32)


def embed(p: Packed, ids: np.ndarray) -> np.ndarray:
    tok = (p.tok.levels[ids].astype(F32) * p.tok.scale[ids][..., None]).astype(F32)
    pos = (p.pos.levels.astype(F32) * p.pos.scale[:, None]).astype(F32)[None, : ids.shape[1]]
    return km.layer_norm((tok + pos).astype(F32), *p.ln_in)


def block(p: Packed, i: int, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blk = p.blocks[i]
    b, t, d = x.shape
    heads = p.config["heads"]
    hd = d // heads
    split = lambda z: z.reshape(b, t, heads, hd).transpose(0, 2, 1, 3)  # noqa: E731
    q, k, v = (split(_linear(x, blk[n])) for n in ("q", "k", "v"))
    scores = km.dot_f32(q[:, :, :, None, :], k[:, :, None, :, :]) * inv_sqrt_head_dim(hd)
    scores = (scores + mask[:, None, None, :]).astype(F32)
    a = km.softmax(scores, axis=-1)
    ctx = km.dot_f32(a[:, :, :, :, None].transpose(0, 1, 2, 4, 3), v.transpose(0, 1, 3, 2)[:, :, None, :, :])
    ctx = ctx.transpose(0, 2, 1, 3).reshape(b, t, d)
    x = km.layer_norm((x + _linear(ctx, blk["o"])).astype(F32), *blk["ln1"])
    h = km.gelu(_linear(x, blk["up"]))
    return km.layer_norm((x + _linear(h, blk["down"])).astype(F32), *blk["ln2"])


def forward(p: Packed, ids: np.ndarray, trim: bool = True) -> dict[str, np.ndarray]:
    """`trim` drops trailing all-padding columns. Under ADR-0009 (one activation scale per token) this cannot change a
    single bit: every per-token quantity depends on that token alone, and a padded key contributes exp(-1e4), which
    underflows to exactly 0.0, so the sequential sums see the same addends in the same order. `tests/test_oracle.py`
    checks it. Under the per-sequence scale it replaced, trimming changed results, and the kernel could never have
    skipped padding."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None]
    full = ids.shape[1]
    if trim:
        used = int((ids != PAD).any(axis=0).nonzero()[0].max()) + 1
        ids = ids[:, :used]
    pad = ids == PAD
    mask = np.where(pad, NEG, F32(0.0)).astype(F32)
    x = embed(p, ids)
    for i in range(len(p.blocks)):
        x = block(p, i, x, mask)
    ptr = _row_apply(x, p.pointers)                       # (b, t, 4)
    ptr = np.where(pad[..., None], NEG, ptr).astype(F32)
    if ptr.shape[1] < full:
        ptr = np.concatenate([ptr, np.full((ptr.shape[0], full - ptr.shape[1], ptr.shape[2]), NEG, F32)], axis=1)
    return {"pointers": ptr.transpose(0, 2, 1), "unit": _row_apply(x[:, 0], p.unit), "defer": _row_apply(x[:, 0], p.defer)[..., 0]}


def decisions(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """What the kernel hands to the struct conversion: four argmax positions, a unit class, and the contract score."""
    lp = (out["pointers"] - out["pointers"].max(-1, keepdims=True)).astype(F32)
    lu = (out["unit"] - out["unit"].max(-1, keepdims=True)).astype(F32)
    zp, zu = km.logf(km.seq_sum(km.expf(lp))), km.logf(km.seq_sum(km.expf(lu)))
    score = -zu.astype(F32)
    for r in range(zp.shape[1]):  # the argmax of each row contributes -log Z; summed in row order, as the kernel does
        score = (score - zp[:, r]).astype(F32)
    return {"ptr": out["pointers"].argmax(-1), "unit": out["unit"].argmax(-1), "defer": out["defer"], "maxprob": score}


def encode_ids(payloads: list[bytes], max_len: int = MAX_LEN) -> np.ndarray:
    from static_student.student.data import encode
    from static_student.tasks.telemetry import Row
    return encode([Row(payload=b, defer=True) for b in payloads])["ids"]
