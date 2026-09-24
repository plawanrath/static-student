"""Quantization-aware training for the trunk: the training grid IS the grid the kernel reads.

Every trunk linear (q, k, v, o, up, down) gets weight-only fake quantization: symmetric, per group of 32 input
weights, integer levels in [-2^(b-1), 2^(b-1) - 1] at b = 2, 3 or 4 bits, one learned step size per group (LSQ-style,
straight-through rounding). Inputs of those linears are fake-quantized to int8 with one dynamic scale **per token**
(ADR-0009), which is what the integer kernel computes; a token's result then depends on that token alone, so the
kernel can stop at the end of the payload instead of running to the padded length. Embeddings, LayerNorm and heads stay in float here and ship at 8 bit.

  to_qat(student, bits)         replace the trunk linears in place, step sizes initialized from the float weights
  QLinear.integer_weights()     (int8 levels, float16 group scales): exactly what codegen packs into the constant array
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 32
TRUNK_LINEARS = ("q", "k", "v", "o", "up", "down")


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def fake_quant_activation(x: torch.Tensor) -> torch.Tensor:
    """int8, symmetric, one dynamic scale per token (ADR-0009): x ~ s * round(x / s), s = max_d |x[..., d]| / 127."""
    s = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    return _round_ste(x / s).clamp(-127, 127) * s


class QLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int, quantize_input: bool = True):
        super().__init__()
        if linear.in_features % GROUP:
            raise ValueError(f"in_features {linear.in_features} is not a multiple of the group size {GROUP}")
        self.bits, self.quantize_input = bits, quantize_input
        self.qn, self.qp = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        self.weight, self.bias = nn.Parameter(linear.weight.detach().clone()), nn.Parameter(linear.bias.detach().clone())
        g = self.weight.detach().view(linear.out_features, -1, GROUP)
        self.step = nn.Parameter((2 * g.abs().mean(-1) / (self.qp ** 0.5)).clamp_min(1e-5))  # LSQ initialization, one per group

    def _levels(self) -> tuple[torch.Tensor, torch.Tensor]:
        step = self.step.abs().clamp_min(1e-6)
        g = self.weight.view(self.weight.shape[0], -1, GROUP)
        return _round_ste(g / step[..., None]).clamp(self.qn, self.qp), step

    def quantized_weight(self) -> torch.Tensor:
        q, step = self._levels()
        return (q * step[..., None]).view_as(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(fake_quant_activation(x) if self.quantize_input else x, self.quantized_weight(), self.bias)

    @torch.no_grad()
    def integer_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Levels as int8 (out, in) and group scales as float16 (out, in / GROUP). The float16 cast is part of the contract:
        the oracle and the kernel both use the cast value, so training must be evaluated with it too (see snap_scales)."""
        q, step = self._levels()
        return q.view_as(self.weight).to(torch.int8), step.to(torch.float16)

    @torch.no_grad()
    def snap_scales(self) -> None:
        """Round the learned step sizes to float16, the precision that ships, so that the float model equals what the kernel computes."""
        self.step.copy_(self.step.abs().clamp_min(1e-6).to(torch.float16).to(self.step.dtype))


def to_qat(student: nn.Module, bits: int, quantize_input: bool = True) -> nn.Module:
    for blk in student.blocks:
        for name in TRUNK_LINEARS:
            lin = getattr(blk, name)
            src = lin if isinstance(lin, nn.Linear) else nn.Linear(lin.weight.shape[1], lin.weight.shape[0])
            if not isinstance(lin, nn.Linear):  # re-quantizing a QAT model at a lower width: start from its latent float weights
                src.weight.data, src.bias.data = lin.weight.data.clone(), lin.bias.data.clone()
            setattr(blk, name, QLinear(src, bits, quantize_input))
    return student


def weight_bytes(student: nn.Module) -> dict[str, int]:
    """Bytes of the constant array: packed trunk levels + float16 group scales + float biases, and everything else at 8 bit."""
    trunk = other = 0
    for m in student.modules():
        if isinstance(m, QLinear):
            trunk += (m.weight.numel() * m.bits + 7) // 8 + 2 * m.step.numel() + 4 * m.bias.numel()
    q_params = {id(p) for m in student.modules() if isinstance(m, QLinear) for p in m.parameters()}
    other = sum(p.numel() for p in student.parameters() if id(p) not in q_params)
    return {"trunk": trunk, "other_at_8bit": other, "total": trunk + other}
