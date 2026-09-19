import copy
import random

import pytest

from static_student.tasks import telemetry as T

LAT = {"role": "latency", "key": "lat", "value": {"gen": "duration", "unit": "ms", "unit_pos": "suffix"}}
USR = {"role": "user", "key": "usr", "value": {"gen": "id", "scheme": "decimal"}}
HOST = {"role": "distractor", "key": "host", "value": {"gen": "hostname"}}
UP = {"role": "distractor", "key": "upstream_latency", "value": {"gen": "duration", "unit": "ms"}}


def fam(container="kv", slots=(UP, LAT, USR, HOST), **kw):
    return {"bucket": "clean", "envelope": "<iso_ts> <level> <component>: ", "container": container,
            "slots": copy.deepcopy(list(slots)), "order": "shuffle", "defer_reason": None, **kw}


def lat(**value):
    key = value.pop("key", "lat")
    return {"role": "latency", "key": key, "value": {"gen": "duration", "unit": "ms", **value}}


FAMILIES = {
    "kv": fam(sep=" ", assign="="),
    "kv_colon_semicolon": fam(sep="; ", assign=": ", quote="double_strings"),
    "kv_space_unit": fam(slots=(lat(unit_pos="suffix_space", numfmt="decimal"), USR, HOST), sep=" ", assign="=", quote="double_all"),
    "kv_key_unit": fam(slots=(lat(key="svc.latencyMs", unit_pos="key", numfmt="decimal"), USR), sep=", ", assign="="),
    "kv_thousands": fam(slots=(lat(numfmt="thousands"), USR, UP), sep="|", assign="="),
    "kv_micro_sign": fam(slots=(lat(unit="us", unit_text="µs"), USR), sep=" ", assign="="),
    "logfmt": fam("logfmt", slots=(lat(numfmt="sci", unit="s"), USR, {"role": "distractor", "key": "msg", "value": {"gen": "free_text"}})),
    "json_flat": fam("json_flat", slots=(lat(key="latency_ms", unit_pos="key"), {**USR, "key": "user_id"}, HOST), envelope=""),
    "json_nested": fam("json_nested", json_spaced=True, slots=(
        {**lat(key="elapsed", unit="s", numfmt="decimal"), "path": ["req", "timing"]},
        {"role": "user", "key": "id", "path": ["user"], "value": {"gen": "id", "scheme": "uuid"}}, HOST)),
    "query_string": fam("query_string", envelope="<method> <path>?", suffix=" HTTP/1.1", slots=(
        lat(key="took", unit_pos="implicit"), {**USR, "key": "uid"}, {"role": "distractor", "key": "q", "value": {"gen": "free_text"}})),
    "csv": fam("csv_positional", order="fixed", envelope="", slots=(
        {"role": "distractor", "value": {"gen": "iso_ts"}}, {"role": "distractor", "value": {"gen": "method"}},
        {"role": "latency", "value": {"gen": "duration", "unit": "ms"}}, {"role": "user", "value": {"gen": "id", "scheme": "prefixed"}})),
    "free_text": fam("free_text", order="fixed", template="request by user {1} to {2} completed in {0}", slots=(
        {"role": "latency", "value": {"gen": "duration", "unit": "s", "numfmt": "decimal", "unit_pos": "suffix_space"}},
        {"role": "user", "value": {"gen": "id", "scheme": "hex"}}, {"role": "distractor", "value": {"gen": "path"}})),
}
DEFER = {
    "target_missing": fam(slots=(LAT, HOST), sep=" ", assign="="),
    "ambiguous_latency": fam(slots=(LAT, lat(key="elapsed"), USR), sep=" ", assign="="),
    "non_numeric": fam(slots=(lat(numfmt="nonnumeric"), USR), sep=" ", assign="="),
    "unit_unknown": fam(slots=(lat(unit_pos="none"), USR), sep=" ", assign="="),
    "out_of_range": fam(slots=(lat(numfmt="overflow"), USR), sep=" ", assign="="),
    "decimal_comma": fam(slots=(lat(numfmt="decimal_comma", unit="s"), USR), sep=" ", assign="="),
    "id_too_long": fam(slots=(LAT, {**USR, "value": {"gen": "id", "scheme": "too_long"}}), sep=" ", assign="="),
    "multi_record": fam(records=2, sep=" ", assign="="),
    "truncated": fam(damage={"kind": "truncate", "p": 1.0}, sep=" ", assign="="),
    "corrupted": fam("json_flat", damage={"kind": "corrupt", "p": 1.0}),
}
for _r, _f in DEFER.items():
    _f["defer_reason"] = _r
    _f["bucket"] = "should_defer"


def test_to_micros_is_exact_and_truncates():
    assert T.to_micros("812", "ms") == 812_000
    assert T.to_micros("0.812", "s") == 812_000
    assert T.to_micros("8.12e2", "ms") == 812_000
    assert T.to_micros("1,204.5", "ms") == 1_204_500
    assert T.to_micros("1999", "ns") == 1          # truncated toward zero, not rounded
    assert T.to_micros("0.1", "us") == 0
    assert T.to_micros("1.5", "min") == 90_000_000
    assert T.to_micros("0.1", "s") == 100_000      # no binary floating point on the path
    for bad in ("0,812", "12,34", "NaN", "-5", "", "1e", "99999999999999999999999"):
        assert T.to_micros(bad, "ms") is None, bad


def test_unit_from_key_and_base_key():
    assert [T.unit_from_key(k) for k in ("latency_ms", "latencyMs", "svc.latency.us", "lat", "user_id", "rt_secs")] == \
        ["ms", "ms", "us", None, None, "s"]
    assert T.base_key("svc.latencyMs") == "latency" and T.base_key("user_id") == "userid"
    assert T.base_key("upstream_latency") != T.base_key("latency")


@pytest.mark.parametrize("name", list(FAMILIES))
def test_clean_families_round_trip(name):
    f = FAMILIES[name]
    assert T.check_family(f, n=300, seed=1) == []
    rng = random.Random(0)
    for _ in range(200):
        row = T.render(f, rng)
        assert not row.defer and len(row.payload) <= T.MAX_PAYLOAD
        a, b = row.lat_span
        c, d = row.usr_span
        assert T.to_micros(row.payload[a:b], row.unit) == row.latency_us
        assert row.payload[c:d].decode() == row.user_id and 0 < len(row.user_id) <= T.USER_ID_MAX
        assert T.Row.from_dict(row.as_dict()) == row


@pytest.mark.parametrize("reason", list(DEFER))
def test_defer_constructions(reason):
    f = DEFER[reason]
    assert T.check_family(f, n=300, seed=2) == []
    rng = random.Random(3)
    rows = [T.render(f, rng) for _ in range(200)]
    assert all(r.defer and r.reason == reason and r.lat_span is None for r in rows)


def test_truncation_never_leaves_a_payload_that_reads_as_complete():
    """A cut inside a bare id or a bare number would look like a valid shorter value; the renderer must not make it."""
    clean = fam(slots=(lat(unit_text="sec", unit="s", numfmt="decimal"), USR), sep=" ", assign="=", order="shuffle")
    cut = {**clean, "damage": {"kind": "truncate", "p": 1.0}, "defer_reason": "truncated"}
    rng = random.Random(5)
    for _ in range(2000):
        row = T.render(cut, rng)
        assert row.defer and T.inverse(clean, row.payload) is None, row.payload


def test_partial_damage_mixes_labels():
    f = fam(damage={"kind": "truncate", "p": 0.3}, sep=" ", assign="=")
    assert T.check_family(f, n=300) == []
    rng = random.Random(7)
    rate = sum(T.render(f, rng).defer for _ in range(2000)) / 2000
    assert 0.25 < rate < 0.35


@pytest.mark.parametrize("mutate, fragment", [
    (lambda f: f["slots"].__setitem__(0, {"role": "distractor", "key": "latency_ms", "value": {"gen": "int"}}), "target alias"),
    (lambda f: f["slots"][1]["value"].update(unit_pos="key"), "does not carry unit"),
    (lambda f: f["slots"][1].update(key="took"), "implicit unit"),
    (lambda f: f.update(defer_reason="non_numeric"), "surface implies"),
    (lambda f: f.update(container="xml"), "container"),
])
def test_validate_family_rejects(mutate, fragment):
    f = copy.deepcopy(FAMILIES["kv"])
    mutate(f)
    errs = T.validate_family(f)
    assert errs and fragment in " ".join(errs)


def test_check_family_rejects_unparseable_and_overlong():
    unquoted_text = fam(slots=(LAT, USR, {"role": "distractor", "key": "msg", "value": {"gen": "free_text"}}), sep=" ", assign="=")
    assert "inverse" in T.check_family(unquoted_text, n=200)[0]
    long = fam(slots=[LAT, USR] + [{"role": "distractor", "key": f"trace_{i}", "value": {"gen": "id", "scheme": "uuid"}} for i in range(6)], sep=" ", assign="=")
    assert "exceeds" in T.check_family(long, n=50)[0]
    thousands_in_csv = fam(slots=(lat(numfmt="thousands"), USR), sep=",", assign="=")
    assert T.check_family(thousands_in_csv, n=200) != []


def test_render_is_deterministic_in_the_seed():
    f = FAMILIES["json_nested"]
    a = [T.render(f, random.Random(11)).payload for _ in range(3)]
    b = [T.render(f, random.Random(11)).payload for _ in range(3)]
    assert a == b
