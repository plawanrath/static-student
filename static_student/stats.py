"""Statistics used by every comparison in this repository.

  - pass_rate(x): point estimate of a binary rate.
  - bootstrap_ci(x, n_resamples=10_000): percentile CI for a single rate or mean.
  - paired_bootstrap_diff(a, b, n_resamples=10_000): CI + one-sided p-value for (a - b) on paired outcomes
    (same payloads, same order).
  - selective_risk_ci(wrong, accepted): CI for the error rate on accepted inputs, plus coverage.
  - ratio_of_medians_ci(a, b): CI for median(a) / median(b) on two independent samples (e.g. start-up times).
  - min_detectable_effect(n, alpha=0.05, baseline_rate=0.5): power-analysis helper.

Outputs are plain dataclasses with `as_dict()` so results serialize straight to JSON.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    n: int
    n_resamples: int

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "n_resamples": self.n_resamples,
        }


@dataclass
class PairedDiffResult:
    point: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    n_resamples: int

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n": self.n,
            "n_resamples": self.n_resamples,
        }


def pass_rate(passed: np.ndarray | list[int]) -> float:
    arr = np.asarray(passed, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def bootstrap_ci(
    passed: np.ndarray | list[int],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile-method bootstrap CI for a binary pass-rate."""
    arr = np.asarray(passed, dtype=float)
    n = arr.size
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0, n_resamples)
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_resamples, n), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        point=float(arr.mean()),
        ci_low=float(low),
        ci_high=float(high),
        n=int(n),
        n_resamples=n_resamples,
    )


def paired_bootstrap_diff(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDiffResult:
    """Paired bootstrap CI for (rate_a - rate_b) + one-sided p-value that a > b.

    Paired across identical payload indices: both arrays must be same length
    and aligned by sample index.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"shape mismatch: {a_arr.shape} vs {b_arr.shape}")
    n = a_arr.size
    if n == 0:
        return PairedDiffResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, n_resamples)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a_arr[idx].mean(axis=1) - b_arr[idx].mean(axis=1)
    observed = float(a_arr.mean() - b_arr.mean())
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # One-sided p-value (test: a > b): fraction of bootstrap diffs ≤ 0.
    p_value = float((diffs <= 0).mean())
    return PairedDiffResult(
        point=observed,
        ci_low=float(low),
        ci_high=float(high),
        p_value=p_value,
        n=int(n),
        n_resamples=n_resamples,
    )


def min_detectable_effect(
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    baseline_rate: float = 0.5,
) -> float:
    """Normal-approximation MDE for a two-proportion test at sample size n per arm.

    Returned as an absolute difference in rates.
    """
    if n <= 0:
        return float("inf")
    from scipy.stats import norm  # type: ignore
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    p = baseline_rate
    # Two-proportion pooled SE approximation.
    se = math.sqrt(2 * p * (1 - p) / n)
    return float((z_a + z_b) * se)


def selective_risk_ci(
    wrong: np.ndarray | list[int],
    accepted: np.ndarray | list[int],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Error rate on accepted inputs (wrong & accepted) / accepted, and coverage, with percentile bootstrap CIs.

    Resamples payloads, so the ratio's denominator varies across resamples as it does in deployment.
    """
    w = np.asarray(wrong, dtype=float)
    acc = np.asarray(accepted, dtype=float)
    if w.shape != acc.shape:
        raise ValueError(f"shape mismatch: {w.shape} vs {acc.shape}")
    n = w.size
    err = w * acc
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    a_sum = acc[idx].sum(axis=1)
    risk = np.divide(err[idx].sum(axis=1), a_sum, out=np.full(n_resamples, np.nan), where=a_sum > 0)
    cov = a_sum / n
    q = [alpha / 2, 1 - alpha / 2]
    r_lo, r_hi = np.nanquantile(risk, q)
    c_lo, c_hi = np.quantile(cov, q)
    n_acc = int(acc.sum())
    return {
        "risk": float(err.sum() / n_acc) if n_acc else float("nan"),
        "risk_ci_low": float(r_lo), "risk_ci_high": float(r_hi),
        "coverage": float(acc.mean()), "coverage_ci_low": float(c_lo), "coverage_ci_high": float(c_hi),
        "n": int(n), "n_accepted": n_acc, "n_resamples": n_resamples,
    }


def ratio_of_medians_ci(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap CI for median(a) / median(b); a and b are independent samples."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ma = np.median(a_arr[rng.integers(0, a_arr.size, size=(n_resamples, a_arr.size))], axis=1)
    mb = np.median(b_arr[rng.integers(0, b_arr.size, size=(n_resamples, b_arr.size))], axis=1)
    low, high = np.quantile(ma / mb, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(float(np.median(a_arr) / np.median(b_arr)), float(low), float(high),
                           int(min(a_arr.size, b_arr.size)), n_resamples)
