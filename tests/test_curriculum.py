import collections
from pathlib import Path

import pytest

from static_student.curriculum import build, teacher
from static_student.curriculum.spec import parse_spec
from static_student.tasks import telemetry as T

REPO = Path(__file__).resolve().parent.parent


def test_spec_lines_parse_to_the_contract():
    s = parse_spec(REPO / "specs/telemetry.spec")
    assert (s["alpha"], s["delta"], s["min_coverage"]) == (0.01, 0.05, 0.5)
    assert s["target"] == "extract latency + user id" and len(s["sha256"]) == 64
    u = parse_spec(REPO / "specs/useragent.spec")
    assert u["alpha"] == 0.01


def test_spec_rejects_a_missing_contract_term(tmp_path):
    p = tmp_path / "bad.spec"
    p.write_text("Semantic Target: x\nFallback: legacy parser\nContract: <=1% wrong on accepted inputs\n")
    with pytest.raises(ValueError):
        parse_spec(p)


@pytest.fixture(scope="module")
def built():
    t = teacher.StubTeacher()
    card = t.task_card(parse_spec(REPO / "specs/telemetry.spec"))
    return build.propose_families(t, card, 80, seed=2, reject=set())


def test_bucket_mix_and_validation(built):
    families, stats = built
    assert collections.Counter(f["bucket"] for f in families) == {"clean": 28, "near_miss": 20, "drifted": 20, "should_defer": 12}
    assert len({T.family_signature(f) for f in families}) == 80
    for f in families:
        assert T.check_family(f, n=50, seed=9) == []
        assert (f["bucket"] == "should_defer") == (f["defer_reason"] is not None) or f["bucket"] == "drifted"
        assert f["bucket"] != "drifted" or f["trace"]
    assert sum(c["accepted"] for c in stats.values()) == 80


def test_pools_split_by_family_and_rows_carry_gold(built):
    families, _ = built
    pools = build.render_pools(families, rows=4000, seed=2)
    train = {r["family_id"] for r in pools["tel-train"]}
    dev = {r["family_id"] for r in pools["tel-dev"]}
    assert train and dev and not train & dev and len(pools["tel-dev"]) == 400
    assert {r["bucket"] for r in pools["tel-dev"]} == set(teacher.BUCKETS)
    for d in pools["tel-train"][:500]:
        row = T.Row.from_dict({k: d[k] for k in ("payload", "defer", "reason", "latency_us", "user_id", "lat_span", "unit", "usr_span")})
        if not row.defer:
            assert T.to_micros(row.payload[slice(*row.lat_span)], row.unit) == row.latency_us
            assert row.payload[slice(*row.usr_span)].decode() == row.user_id
    assert build.render_pools(families, rows=300, seed=2) == build.render_pools(families, rows=300, seed=2)
