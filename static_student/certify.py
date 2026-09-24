"""Certification of the acceptance threshold with Learn-then-Test (fixed-sequence testing, exact binomial p-values).

The contract: among inputs the student accepts, at most alpha are wrong, with probability at least 1 - delta over the
draw of the calibration pool, for inputs exchangeable with that pool. Thresholds are tested from the strictest to the
most permissive on a grid fixed before the calibration data is seen (it is placed with the dev pool, which is
disjoint from calibration, so that the sequence starts where a test has power); testing stops at the first threshold whose null
hypothesis "selective risk > alpha" cannot be rejected at level delta, which controls the family-wise error without a
multiplicity correction. The build fails if no threshold is certified or if the certified coverage is below the spec's
minimum. Scores are whatever the shipped kernel outputs; an input whose spans do not decode is never accepted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import binom

def coverage_grid(dev_score: np.ndarray, dev_emits: np.ndarray, start: float = 0.05, steps: int = 96) -> tuple[float, ...]:
    """Thresholds at which the DEV pool's coverage is start, ..., 1.0 of the inputs that emit, strictest first. A sequence that began at
    a threshold nobody clears could never reject its first hypothesis and would certify nothing. Uses no calibration data."""
    s = np.sort(np.asarray(dev_score, float)[np.asarray(dev_emits, bool)])[::-1]
    idx = np.unique(np.clip((np.linspace(start, 1.0, steps) * len(s)).astype(int) - 1, 0, len(s) - 1))
    return tuple(dict.fromkeys(float(s[i]) for i in idx))


class CertificationError(RuntimeError):
    """Raised when the contract cannot be certified. The build catches nothing: this is the failing build."""


@dataclass
class Certificate:
    alpha: float
    delta: float
    min_coverage: float
    tau: float
    n_calibration: int
    n_accepted: int
    k_wrong: int
    coverage: float
    empirical_risk: float
    p_value: float
    grid_sha256: str
    pool_sha256: str
    thresholds_tested: int

    def as_dict(self) -> dict:
        return asdict(self)


def grid_hash(grid) -> str:
    return hashlib.sha256(json.dumps([round(g, 12) for g in grid]).encode()).hexdigest()


def certify(score: np.ndarray, wrong: np.ndarray, emits: np.ndarray, alpha: float, delta: float, min_coverage: float,
            grid: tuple[float, ...], pool_sha256: str = "") -> Certificate:
    """score: higher is more confident. wrong: 1 if accepting the input would violate the contract. emits: the spans decode."""
    score, wrong, emits = np.asarray(score, float), np.asarray(wrong, int), np.asarray(emits, bool)
    n = len(score)
    best, tested = None, 0
    for tau in sorted(grid, reverse=True):  # strictest first
        acc = emits & (score >= tau)
        n_acc, k = int(acc.sum()), int(wrong[acc].sum())
        tested += 1
        p = float(binom.cdf(k, n_acc, alpha)) if n_acc else 1.0  # P(K <= k) under risk = alpha: small means risk < alpha
        if p > delta:
            break
        best = (tau, n_acc, k, p)
    if best is None:
        raise CertificationError(f"no threshold on the grid certifies risk <= {alpha} at confidence {1 - delta} (n = {n})")
    tau, n_acc, k, p = best
    cert = Certificate(alpha, delta, min_coverage, tau, n, n_acc, k, n_acc / n, k / n_acc, p, grid_hash(grid), pool_sha256, tested)
    if cert.coverage < min_coverage:
        raise CertificationError(f"certified coverage {cert.coverage:.3f} is below the contract's minimum {min_coverage}")
    return cert
