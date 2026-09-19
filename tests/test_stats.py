import numpy as np
import pytest

from static_student import stats


def test_bootstrap_ci_brackets_point():
    x = np.r_[np.ones(80), np.zeros(20)]
    r = stats.bootstrap_ci(x, n_resamples=2000)
    assert r.point == pytest.approx(0.8)
    assert r.ci_low < 0.8 < r.ci_high


def test_paired_diff_detects_gap_and_checks_shape():
    rng = np.random.default_rng(1)
    b = (rng.random(2000) < 0.6).astype(int)
    a = np.maximum(b, (rng.random(2000) < 0.3).astype(int))
    r = stats.paired_bootstrap_diff(a, b, n_resamples=2000)
    assert r.ci_low > 0 and r.p_value < 0.01
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([1, 0], [1])


def test_selective_risk():
    accepted = np.r_[np.ones(900), np.zeros(100)]
    wrong = np.zeros(1000); wrong[:9] = 1; wrong[950:] = 1  # errors on deferred inputs do not count
    r = stats.selective_risk_ci(wrong, accepted, n_resamples=2000)
    assert r["risk"] == pytest.approx(0.01)
    assert r["coverage"] == pytest.approx(0.9)
    assert r["risk_ci_low"] <= 0.01 <= r["risk_ci_high"]


def test_ratio_of_medians():
    rng = np.random.default_rng(2)
    r = stats.ratio_of_medians_ci(rng.normal(100, 5, 300), rng.normal(10, 1, 300), n_resamples=2000)
    assert 9 < r.ci_low < r.point < r.ci_high < 11


def test_mde_shrinks_with_n():
    assert stats.min_detectable_effect(20_000, baseline_rate=0.01) < stats.min_detectable_effect(2_000, baseline_rate=0.01)
