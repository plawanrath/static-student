import json
import random
from pathlib import Path

import pytest

from static_student.drift import heldout, operators, timeline
from static_student.tasks import telemetry as T

REPO = Path(__file__).resolve().parent.parent
SOURCES = json.loads((REPO / "data/drift/sources_v0.json").read_text())["sources"]
ALL_OPS = {**operators.OPERATORS, **heldout.OPERATORS}


@pytest.mark.parametrize("op", list(ALL_OPS))
def test_operator_output_keeps_gold_by_construction(op):
    rng = random.Random(op)
    applied = 0
    for src in SOURCES:
        for _ in range(6):
            out = operators.apply(op, src, rng, rng.random(), operators=ALL_OPS)
            if out is None:
                continue
            applied += 1
            assert out["trace"][-1] == op and out != src
            assert T.check_family(out, n=100, seed=1) == []
            assert out["defer_reason"] == T.surface_defer_reason(out)
    assert applied >= 12, "operator applies to too few source formats to matter"


def test_operators_do_not_mutate_their_input():
    before = json.dumps(SOURCES[0], sort_keys=True)
    operators.apply("D6", SOURCES[0], random.Random(0), 0.9)
    assert json.dumps(SOURCES[0], sort_keys=True) == before


def test_operators_compose():
    rng = random.Random(3)
    fam = SOURCES[1]
    for op in ("D1", "D4", "D6", "D3", "D7", "D5", "D8"):
        fam = operators.apply(op, fam, rng, 0.8) or fam
    assert len(fam["trace"]) >= 5 and T.check_family(fam, n=200) == []


def test_heldout_parameters_are_disjoint_from_in_curriculum_ones():
    for k, v in heldout.PARAMS.items():
        assert not set(map(str, v)) & set(map(str, operators.PARAMS[k])), k
    vocab = {T.base_key(k) for k in T.K_LAT + T.K_USR + T.DISTRACTOR_DURATIONS + T.DISTRACTOR_IDS}
    assert not {T.base_key(k) for k in heldout.PARAMS["lat_aliases"] + heldout.PARAMS["usr_aliases"]} & vocab


def test_decimal_comma_always_defers_and_never_reads_as_thousands():
    fam = {"container": "kv", "sep": " ", "assign": "=", "defer_reason": "decimal_comma", "slots": [
        {"role": "latency", "key": "lat", "value": {"gen": "duration", "unit": "s", "numfmt": "decimal_comma"}},
        {"role": "user", "key": "user", "value": {"gen": "id"}}]}
    rng = random.Random(0)
    for _ in range(500):
        row = T.render(fam, rng)
        num = row.payload.split(b"lat=")[1].split(b"s ")[0]
        assert row.defer and T.to_micros(num, "s") is None


def _cfg(**kw):
    return {**timeline.DEFAULTS, "name": "t", "track": "in_curriculum", "seed": 4, "steps": 10, "payloads_per_source": 20, **kw}


def test_timeline_is_deterministic_and_starts_undrifted():
    a, ev_a = timeline.sample(_cfg())
    b, ev_b = timeline.sample(_cfg())
    assert a == b and ev_a == ev_b
    assert len(a) == 10 and all(len(s) == 8 for s in a)
    assert all("trace" not in f for f in a[0]) and any(f.get("trace") for f in a[-1])
    assert timeline.render_step(_cfg(), a[5], 5) == timeline.render_step(_cfg(), b[5], 5)
    assert timeline.sample(_cfg(seed=5))[0] != a


@pytest.mark.parametrize("track, allowed", [("in_curriculum", set(operators.OPERATORS)), ("heldout_params", {f"D{i}" for i in range(1, 8)}),
                                            ("heldout_operators", set(heldout.OPERATORS)), ("none", set())])
def test_tracks_use_only_their_operators(track, allowed):
    steps, events = timeline.sample(_cfg(track=track, rate=0.6))
    assert {e["op"] for e in events} <= allowed
    assert {op for f in steps[-1] for op in f.get("trace", [])} <= allowed
    if track == "none":
        assert steps[0] == steps[-1]


def test_heldout_params_track_shows_unseen_keys():
    steps, _ = timeline.sample(_cfg(track="heldout_params", rate=0.8, only_operators=["D1"]))
    keys = {T.base_key(s["key"]) for f in steps[-1] if f.get("trace") for s in f["slots"] if s["role"] != "distractor" and s.get("key")}
    assert keys & {T.base_key(k) for k in heldout.PARAMS["lat_aliases"] + heldout.PARAMS["usr_aliases"]}
