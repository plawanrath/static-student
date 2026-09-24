"""Byte-level bidirectional encoder in BERT layout (learned absolute positions, post-LN, GELU) with pointer heads.

Position 0 is [CLS]. A pointer that selects position 0 means "field absent". Byte i of the payload sits at position
i + 1, so a gold span [a, b) becomes start = a + 1, end = b (the position of the last byte). Every sequence is padded
to MAX_LEN, so the shape, and with it the generated kernel, is fixed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, CLS = 256, 257
VOCAB = 260
MAX_LEN = 257  # [CLS] + 256 bytes
UNIT_CLASSES = ("ns", "us", "ms", "s", "min")
POINTERS = ("lat_start", "lat_end", "usr_start", "usr_end")


@dataclass
class StudentConfig:
    layers: int
    d_model: int
    heads: int
    ffn_mult: int = 4

    def as_dict(self) -> dict:
        return asdict(self)


SIZES = {"XS": StudentConfig(3, 96, 4), "S": StudentConfig(4, 192, 4), "M": StudentConfig(6, 384, 6)}


class Block(nn.Module):
    def __init__(self, c: StudentConfig):
        super().__init__()
        d = c.d_model
        self.heads = c.heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.up, self.down = nn.Linear(d, c.ffn_mult * d), nn.Linear(c.ffn_mult * d, d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        split = lambda z: z.view(b, t, self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        a = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(x)), split(self.v(x)), attn_mask=bias)
        x = self.ln1(x + self.o(a.transpose(1, 2).reshape(b, t, d)))
        return self.ln2(x + self.down(F.gelu(self.up(x))))


class Student(nn.Module):
    def __init__(self, c: StudentConfig):
        super().__init__()
        self.config = c
        self.tok, self.pos = nn.Embedding(VOCAB, c.d_model), nn.Embedding(MAX_LEN, c.d_model)
        self.ln = nn.LayerNorm(c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.pointers = nn.Linear(c.d_model, len(POINTERS))
        self.unit = nn.Linear(c.d_model, len(UNIT_CLASSES))
        self.defer = nn.Linear(c.d_model, 1)

    def trunk(self, ids: torch.Tensor) -> torch.Tensor:
        pad = ids == PAD
        bias = torch.zeros(ids.shape, dtype=torch.float32, device=ids.device).masked_fill(pad, -1e4)[:, None, None, :]
        x = self.ln(self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None])
        for blk in self.blocks:
            x = blk(x, bias)
        return x

    def heads(self, x: torch.Tensor, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        ptr = self.pointers(x).masked_fill((ids == PAD)[..., None], -1e4)  # (B, T, 4); padding can never be pointed at
        return {"pointers": ptr.transpose(1, 2), "unit": self.unit(x[:, 0]), "defer": self.defer(x[:, 0]).squeeze(-1)}

    def forward(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.trunk(ids), ids)

    def n_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trunk = sum(p.numel() for b in self.blocks for p in b.parameters())
        return {"total": total, "trunk": trunk}
