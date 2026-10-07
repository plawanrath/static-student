"""An imported ModernBERT sequence classifier -> packed constants and its integer reference.

The compiler's second front end: instead of a student trained quantization-aware from a spec line, a trained
checkpoint is quantized after training (weights only, symmetric per group of 32 along the input dimension, 8 or 4
bits; activations int8 with one scale per token, as for the students). `forward` is the NumPy reference the generated C
must match bit for bit, written only with `kernel_math` primitives, one sequence at a time, with no padding.

Operators beyond the students: rotary position tables (computed once here and shipped as constants), alternating global
and sliding-window attention (a token attends to keys at distance <= local_attention // 2), gated linear units,
bias-free layer normalization, and mean pooling followed by the prediction head.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from static_student.codegen import kernel_math as km

F32, I32 = np.float32, np.int32
GROUP = 32
NEG = F32(-1e4)
MAX_TOKENS = 256


@dataclass
class QLin:
    levels: np.ndarray  # (out, in) int32 in [-qmax, qmax]
    scales: np.ndarray  # (out, in / GROUP) float32 (rounded through float16, which is what ships)


@dataclass
class Rows:
    levels: np.ndarray  # (rows, cols) int32 in [-127, 127]
    scale: np.ndarray   # (rows,) float32


@dataclass
class MBPacked:
    config: dict
    bits: int
    tok: Rows
    emb_norm: np.ndarray
    layers: list[dict] = field(default_factory=list)
    final_norm: np.ndarray | None = None
    head_dense: QLin | None = None
    head_norm: np.ndarray | None = None
    cls: Rows | None = None
    cls_bias: np.ndarray | None = None
    rope: dict = field(default_factory=dict)  # layer type -> (cos, sin), each (MAX_TOKENS, head_dim) float32

    def sha256(self) -> str:
        h = hashlib.sha256()
        for a in self.arrays():
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def arrays(self):
        yield self.tok.levels.astype(np.int8)
        yield self.tok.scale
        yield self.emb_norm
        for L in self.layers:
            for k in ("attn_norm", "mlp_norm"):
                if L[k] is not None:
                    yield L[k]
            for k in ("qkv", "o", "wi", "wo"):
                yield L[k].levels.astype(np.int8)
                yield L[k].scales
        yield self.final_norm
        yield self.head_dense.levels.astype(np.int8)
        yield self.head_dense.scales
        yield self.head_norm
        yield self.cls.levels.astype(np.int8)
        yield self.cls.scale
        yield self.cls_bias
        for t in sorted(self.rope):
            yield from self.rope[t]


def quantize_groups(w: np.ndarray, bits: int) -> QLin:
    qmax = F32(2 ** (bits - 1) - 1)
    out, n = w.shape
    assert n % GROUP == 0, f"input dimension {n} is not a multiple of {GROUP}"
    g = w.reshape(out, n // GROUP, GROUP).astype(F32)
    scale = (np.abs(g).max(axis=2) / qmax).astype(np.float16).astype(F32)
    scale = np.where(scale == 0, F32(1.0), scale).astype(F32)
    levels = np.clip(km.round_to_int(g / scale[..., None]), -qmax, qmax).astype(I32).reshape(out, n)
    return QLin(levels, scale)


def quantize_rows(w: np.ndarray) -> Rows:
    scale = np.maximum(np.abs(w).max(axis=1).astype(F32) / F32(127.0), F32(1e-12)).astype(F32)
    levels = np.clip(km.round_to_int(w.astype(F32) / scale[:, None]), -127, 127).astype(I32)
    return Rows(levels, scale)


def rope_tables(theta: float, head_dim: int, n: int = MAX_TOKENS) -> tuple[np.ndarray, np.ndarray]:
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.arange(n, dtype=np.float64)[:, None] * inv[None, :]
    ang = np.concatenate([ang, ang], axis=1)
    return np.cos(ang).astype(F32), np.sin(ang).astype(F32)


def pack(model, bits: int) -> MBPacked:
    """`model` is a transformers ModernBertForSequenceClassification in float32."""
    c = model.config
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    ln = lambda k: sd[k].astype(F32)  # noqa: E731  bias-free LayerNorm: weight only
    for flag in ("norm_bias", "mlp_bias", "attention_bias", "classifier_bias"):
        assert not getattr(c, flag), f"{flag} is not supported by the generated kernel"
    assert c.classifier_pooling == "mean" and c.hidden_activation == "gelu" and c.classifier_activation == "gelu"
    hd = c.hidden_size // c.num_attention_heads
    p = MBPacked(config={"hidden": c.hidden_size, "heads": c.num_attention_heads, "head_dim": hd, "inter": c.intermediate_size,
                         "layers": c.num_hidden_layers, "layer_types": list(c.layer_types), "window": c.local_attention // 2,
                         "labels": c.num_labels, "eps": float(c.norm_eps), "vocab": c.vocab_size},
                 bits=bits, tok=quantize_rows(sd["model.embeddings.tok_embeddings.weight"]), emb_norm=ln("model.embeddings.norm.weight"))
    for i in range(c.num_hidden_layers):
        pre = f"model.layers.{i}."
        p.layers.append({"attn_norm": None if i == 0 else ln(pre + "attn_norm.weight"), "mlp_norm": ln(pre + "mlp_norm.weight"),
                         "qkv": quantize_groups(sd[pre + "attn.Wqkv.weight"], bits), "o": quantize_groups(sd[pre + "attn.Wo.weight"], bits),
                         "wi": quantize_groups(sd[pre + "mlp.Wi.weight"], bits), "wo": quantize_groups(sd[pre + "mlp.Wo.weight"], bits)})
    p.final_norm = ln("model.final_norm.weight")
    p.head_dense = quantize_groups(sd["head.dense.weight"], bits)
    p.head_norm = ln("head.norm.weight")
    p.cls, p.cls_bias = quantize_rows(sd["classifier.weight"]), sd["classifier.bias"].astype(F32)
    for t, par in c.rope_parameters.items():
        p.rope[t] = rope_tables(float(par["rope_theta"]), hd)
    return p


def _norm(x, w, eps):
    return km.layer_norm(x, w, np.zeros_like(w), F32(eps))


def _lin(x: np.ndarray, q: QLin) -> np.ndarray:
    xq, s = km.quantize_activation(x)
    return (km.dot_groups(xq, q.levels, q.scales, GROUP) * s).astype(F32)


def _rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1).astype(F32)
    return (x * cos + rot * sin).astype(F32)


def layer(p: MBPacked, i: int, x: np.ndarray) -> np.ndarray:
    cfg, L = p.config, p.layers[i]
    t, d = x.shape
    heads, hd = cfg["heads"], cfg["head_dim"]
    kind = cfg["layer_types"][i]
    h = x if L["attn_norm"] is None else _norm(x, L["attn_norm"], cfg["eps"])
    qkv = _lin(h, L["qkv"]).reshape(t, 3, heads, hd)
    cos, sin = (a[:t][:, None, :] for a in p.rope[kind])
    q, k, v = _rotate(qkv[:, 0], cos, sin), _rotate(qkv[:, 1], cos, sin), qkv[:, 2]
    q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)          # (heads, t, hd)
    scores = (km.dot_f32(q[:, :, None, :], k[:, None, :, :]) * F32(hd ** -0.5)).astype(F32)  # (heads, t, t)
    if kind == "sliding_attention":
        dist = np.abs(np.arange(t)[:, None] - np.arange(t)[None, :])
        scores = (scores + np.where(dist > cfg["window"], NEG, F32(0.0))[None]).astype(F32)
    a = km.softmax(scores, axis=-1)
    ctx = km.dot_f32(a[:, :, None, :], v.transpose(0, 2, 1)[:, None, :, :])               # (heads, t, hd)
    x = (x + _lin(ctx.transpose(1, 0, 2).reshape(t, d), L["o"])).astype(F32)
    h = _norm(x, L["mlp_norm"], cfg["eps"])
    u = _lin(h, L["wi"])
    inp, gate = u[:, : cfg["inter"]], u[:, cfg["inter"]:]
    return (x + _lin((km.gelu(inp) * gate).astype(F32), L["wo"])).astype(F32)


def forward(p: MBPacked, ids: np.ndarray) -> np.ndarray:
    """One sequence of token ids (no padding) -> float32 logits (labels,)."""
    ids = np.asarray(ids, dtype=np.int64)
    assert ids.ndim == 1 and 0 < len(ids) <= MAX_TOKENS
    x = (p.tok.levels[ids].astype(F32) * p.tok.scale[ids][:, None]).astype(F32)
    x = _norm(x, p.emb_norm, p.config["eps"])
    for i in range(len(p.layers)):
        x = layer(p, i, x)
    x = _norm(x, p.final_norm, p.config["eps"])
    pooled = (km.seq_sum(x, axis=0) / F32(len(ids))).astype(F32)
    h = _norm(km.gelu(_lin(pooled[None], p.head_dense)), p.head_norm, p.config["eps"])[0]
    w = (p.cls.levels.astype(F32) * p.cls.scale[:, None]).astype(F32)
    return (km.dot_f32(h[None, :], w) + p.cls_bias).astype(F32)


def decision(logits: np.ndarray) -> tuple[int, F32]:
    """Predicted label and its softmax probability, computed with the kernel's own exp so that every machine agrees."""
    z = (logits - logits.max()).astype(F32)
    e = km.expf(z)
    return int(np.argmax(logits)), F32(F32(1.0) / km.seq_sum(e))
