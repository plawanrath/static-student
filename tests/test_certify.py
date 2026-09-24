import numpy as np
import pytest

from static_student import certify as C


def _pool(rng, n, err_at):
    """Scores uniform on the grid's range; error probability is a function of the score."""
    s = -rng.uniform(0, 5, n)
    return s, (rng.random(n) < err_at(s)).astype(int), np.ones(n, bool)


def _grid(rng, err_at=lambda s: s * 0):
    s, _, e = _pool(rng, 20_000, err_at)
    return C.coverage_grid(s, e)


def test_certifies_a_good_student_and_reports_the_pool():
    rng = np.random.default_rng(0)
    s, w, e = _pool(rng, 20_000, lambda s: np.where(s > -2, 0.002, 0.2))
    g = _grid(rng)
    c = C.certify(s, w, e, alpha=0.01, delta=0.05, min_coverage=0.3, grid=g, pool_sha256="abc")
    assert -2.3 < c.tau < -1.5 and 0.3 < c.coverage < 0.45 and c.empirical_risk < 0.01 and c.p_value <= 0.05
    assert c.pool_sha256 == "abc" and c.grid_sha256 == C.grid_hash(g) and g[0] > g[-1] and len(g) > 50 and c.n_calibration == 20_000


def test_build_fails_for_a_broken_student_and_for_low_coverage():
    rng = np.random.default_rng(1)
    s, w, e = _pool(rng, 20_000, lambda s: np.full_like(s, 0.05))
    with pytest.raises(C.CertificationError, match="no threshold"):
        C.certify(s, w, e, 0.01, 0.05, 0.5, _grid(rng))
    s, w, e = _pool(rng, 20_000, lambda s: np.where(s > -0.5, 0.001, 0.3))
    with pytest.raises(C.CertificationError, match="below the contract"):
        C.certify(s, w, e, 0.01, 0.05, 0.5, _grid(rng))


def test_inputs_that_do_not_decode_are_never_accepted():
    rng = np.random.default_rng(2)
    s, w, _ = _pool(rng, 40_000, lambda s: np.full_like(s, 0.001))
    e = rng.random(40_000) < 0.6
    assert C.certify(s, np.where(e, w, 1), e, 0.01, 0.05, 0.3, _grid(rng)).coverage <= 0.61


@pytest.mark.slow
def test_guarantee_holds_at_the_boundary():
    """True risk crosses alpha inside the grid. The certified threshold's TRUE risk may exceed alpha in at most delta of pools."""
    rng = np.random.default_rng(3)
    err = lambda s: np.clip(0.004 + 0.02 * (-s), 0, 1)  # noqa: E731  risk of accepting {score >= tau} rises as tau falls
    big_s = -rng.uniform(0, 5, 2_000_000)
    big_w = rng.random(2_000_000) < err(big_s)
    g = _grid(rng)
    violations = trials = 0
    for _ in range(400):
        s, w, e = _pool(rng, 5_000, err)
        try:
            tau = C.certify(s, w, e, 0.01, 0.05, 0.0, g).tau
        except C.CertificationError:
            trials += 1
            continue
        trials += 1
        violations += big_w[big_s >= tau].mean() > 0.01
    assert violations / trials <= 0.05 + 0.025
