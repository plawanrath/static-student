import numpy as np
import pytest
from scipy.special import erf as scipy_erf

from static_student.codegen import kernel_math as km


def test_expf_is_accurate_and_handles_the_tails():
    x = np.linspace(-80, 40, 20001, dtype=np.float32)
    got, want = km.expf(x), np.exp(x.astype(np.float64))
    assert np.abs(got / want - 1).max() < 3e-6
    assert km.expf(np.float32([-200.0]))[0] == 0.0 and km.expf(np.float32([0.0]))[0] == 1.0
    assert got.dtype == np.float32


def test_erff_and_gelu_track_the_library_versions():
    x = np.linspace(-8, 8, 20001, dtype=np.float32)
    assert np.abs(km.erff(x) - scipy_erf(x.astype(np.float64))).max() < 1e-6  # A&S 7.1.26 (1.5e-7) plus float32 rounding
    want = 0.5 * x.astype(np.float64) * (1 + scipy_erf(x.astype(np.float64) / np.sqrt(2)))
    assert np.abs(km.gelu(x) - want).max() < 1e-6


def test_round_to_int_rounds_halves_to_even():
    x = np.float32([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 1.4999999])
    assert km.round_to_int(x).tolist() == [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0, 1.0]


def test_sequential_sum_differs_from_pairwise_and_is_reproducible():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 4096)).astype(np.float32) * np.float32(1e3)
    a = km.seq_sum(x)
    assert np.array_equal(a, km.seq_sum(x)) and a.dtype == np.float32
    assert not np.array_equal(a, x.sum(-1))  # np.sum reassociates; the C loop does not


def test_layer_norm_and_softmax_match_reference_within_float32_noise():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 7, 64)).astype(np.float32)
    w, b = rng.standard_normal(64).astype(np.float32), rng.standard_normal(64).astype(np.float32)
    ref = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * w + b
    assert np.abs(km.layer_norm(x, w, b) - ref).max() < 1e-4
    s = km.softmax(x)
    assert np.abs(km.seq_sum(s) - 1).max() < 1e-6
    e = np.exp(x.astype(np.float64) - x.max(-1, keepdims=True))
    assert np.abs(s - e / e.sum(-1, keepdims=True)).max() < 1e-6


def test_activation_quantization_and_group_dots_are_exact():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 5, 64)).astype(np.float32)
    q, s = km.quantize_activation(x)
    assert q.dtype == np.int32 and np.abs(q).max() <= 127 and s.shape == (2, 5, 1)  # one scale per token (ADR-0009)
    assert np.abs(q * s - x).max() <= s.max() / 2 + 1e-6
    assert np.array_equal(km.quantize_activation(x[:, :3])[1], s[:, :3])  # a token's scale ignores later tokens
    levels = rng.integers(-8, 8, (16, 64)).astype(np.int32)
    scales = rng.random((16, 2)).astype(np.float32)
    got = km.dot_groups(q, levels, scales, 32)
    want = sum((q[..., g * 32:(g + 1) * 32] @ levels[:, g * 32:(g + 1) * 32].T).astype(np.float32) * scales[:, g] for g in range(2))
    assert np.array_equal(got, want)


def test_dot_f32_matches_a_scalar_loop():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((2, 5, 8)).astype(np.float32), rng.standard_normal((2, 3, 8)).astype(np.float32)
    got = km.dot_f32(a[:, :, None, :], b[:, None, :, :])
    want = np.zeros((2, 5, 3), np.float32)
    for i in range(8):
        want = (want + a[:, :, None, i] * b[:, None, :, i]).astype(np.float32)
    assert np.array_equal(got, want)


def test_logf_is_accurate_over_the_range_the_score_uses():
    x = np.concatenate([np.linspace(1, 300, 20001), np.geomspace(1e-6, 1, 5001)]).astype(np.float32)
    assert np.abs(km.logf(x) - np.log(x.astype(np.float64))).max() < 2e-6
    assert km.logf(np.float32([1.0]))[0] == 0.0 and km.logf(x).dtype == np.float32
