import pytest
import torch

from static_student.student import quant
from static_student.student.model import SIZES, Student


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_levels_stay_on_the_grid_and_reconstruct_the_forward_weights(bits):
    torch.manual_seed(0)
    q = quant.QLinear(torch.nn.Linear(96, 64), bits)
    levels, scales = q.integer_weights()
    assert levels.dtype == torch.int8 and levels.min() >= -(2 ** (bits - 1)) and levels.max() <= 2 ** (bits - 1) - 1
    assert scales.shape == (64, 3) and scales.dtype == torch.float16
    q.snap_scales()
    levels, scales = q.integer_weights()
    rebuilt = (levels.float().view(64, 3, 32) * scales.float()[..., None]).view(64, 96)
    assert torch.equal(rebuilt, q.quantized_weight())  # what ships is exactly what the float model multiplies by


def test_error_shrinks_with_bits_and_gradients_reach_weights_and_steps():
    torch.manual_seed(1)
    lin = torch.nn.Linear(64, 32)
    errs = [(quant.QLinear(lin, b).quantized_weight() - lin.weight).abs().mean().item() for b in (2, 3, 4)]
    assert errs[0] > errs[1] > errs[2]
    q = quant.QLinear(lin, 3)
    q(torch.randn(5, 7, 64)).pow(2).mean().backward()
    assert q.weight.grad.abs().sum() > 0 and q.step.grad.abs().sum() > 0


def test_activation_fake_quant_scales_each_token_on_its_own():
    """ADR-0009: a token's scale depends on that token alone, so padding and neighbours cannot change its result."""
    torch.manual_seed(0)
    x = torch.randn(3, 10, 8) * torch.tensor([1.0, 10.0, 100.0])[:, None, None]
    y = quant.fake_quant_activation(x)
    for i in range(3):
        for t in range(10):
            s = x[i, t].abs().max() / 127
            assert torch.allclose(y[i, t] / s, (y[i, t] / s).round(), atol=1e-3)
            assert (y[i, t] - x[i, t]).abs().max() <= s / 2 + 1e-6
    grown = x.clone()
    grown[:, 5:] *= 1000  # later tokens change a lot; earlier ones must be untouched
    assert torch.equal(quant.fake_quant_activation(grown)[:, :5], y[:, :5])


def test_to_qat_converts_the_trunk_only_and_can_step_down():
    m = quant.to_qat(Student(SIZES["XS"]), 4)
    assert sum(isinstance(x, quant.QLinear) for x in m.modules()) == 3 * 6
    assert isinstance(m.pointers, torch.nn.Linear) and isinstance(m.tok, torch.nn.Embedding)
    b4 = quant.weight_bytes(m)
    m = quant.to_qat(m, 2)
    b2 = quant.weight_bytes(m)
    assert all(x.bits == 2 for x in m.modules() if isinstance(x, quant.QLinear)) and b2["trunk"] < b4["trunk"] and b2["other_at_8bit"] == b4["other_at_8bit"]
    ids = torch.full((2, 257), 256)
    ids[:, 0] = 257
    assert m(ids)["pointers"].shape == (2, 4, 257)
