"""In-curriculum drift operators D1-D8, D13, D14.

Each operator maps (family, rng, severity in [0, 1], params) to a new family, or None when it does not apply.
`params` supplies the value lists (aliases, delimiters, ...); the defaults below are the in-curriculum lists, and the
held-out-parameters track passes unseen ones. `apply` retries until the rewritten family passes `check_family`, so a
drifted family always carries gold by construction.
"""
from __future__ import annotations

import copy
import random

from static_student.tasks import telemetry as T

PARAMS = {
    "lat_aliases": [k for k in T.K_LAT if k not in T.IMPLICIT_UNITS],
    "usr_aliases": list(T.K_USR),
    "namespaces": ["svc", "http", "req", "app"],
    "delims": [(" ", "="), (", ", "="), ("; ", ": "), ("|", "="), (" ", ":"), ("\t", "="), (", ", ": "), (" ", " = ")],
    "distractor_prefixes": ["upstream", "db", "backend", "cache", "queue", "dns"],
    "key_unit_tokens": {"ns": ["ns", "nanos"], "us": ["us", "micros"], "ms": ["ms", "millis"], "s": ["s", "sec", "seconds"], "min": ["min"]},
}
_KEYED = ("kv", "logfmt", "json_flat", "json_nested", "query_string")


def _slot(fam: dict, role: str) -> dict | None:
    return next((s for s in fam["slots"] if s["role"] == role), None)


def _style(rng: random.Random, words: list[str]) -> str:
    how = rng.choice(("snake", "camel", "pascal", "upper", "kebab", "dot"))
    if how == "camel":
        return words[0].lower() + "".join(w.capitalize() for w in words[1:])
    if how == "pascal":
        return "".join(w.capitalize() for w in words)
    sep = {"snake": "_", "upper": "_", "kebab": "-", "dot": "."}[how]
    out = sep.join(w.lower() for w in words)
    return out.upper() if how == "upper" else out


def _set_latency_key(slot: dict, alias: str, rng: random.Random, p: dict, unit_in_key: bool) -> None:
    v = slot["value"]
    if unit_in_key:
        slot["key"] = _style(rng, alias.split("_") + [rng.choice(p["key_unit_tokens"][v["unit"]])])
        v["unit_pos"] = "key"
        v.pop("unit_text", None)
    else:
        slot["key"] = _style(rng, alias.split("_"))
        if v.get("unit_pos", "suffix") in ("key", "implicit"):
            v["unit_pos"] = "suffix"


def d1_key_rename(fam, rng, s, p):
    if fam["container"] not in _KEYED:
        return None
    lat, usr = _slot(fam, "latency"), _slot(fam, "user")
    edits = rng.sample(["lat", "usr", "ns"], k=min(3, 1 + int(2.99 * s)))
    if "lat" in edits and lat:
        in_key = lat["value"].get("unit_pos", "suffix") in ("key", "implicit") or rng.random() < 0.3
        _set_latency_key(lat, rng.choice(p["lat_aliases"]), rng, p, in_key)
    if "usr" in edits and usr:
        usr["key"] = _style(rng, rng.choice(p["usr_aliases"]).split("_"))
    if "ns" in edits:
        ns = rng.choice(p["namespaces"])
        for slot in (lat, usr):
            if slot and "." not in slot["key"] and slot["key"] not in T.IMPLICIT_UNITS:
                slot["key"] = f"{ns}.{slot['key']}"
    return fam


def d2_delimiters(fam, rng, s, p):
    if fam["container"] not in ("kv", "logfmt", "query_string"):
        return None
    sep, assign = rng.choice(p["delims"])
    if fam["container"] == "query_string":
        fam["envelope"] = fam.get("envelope", "").replace("?", " ")
    fam.update(container="kv", sep=sep, assign=assign)
    spaced = any(sl["value"]["gen"] == "free_text" or sl["value"].get("unit_pos") == "suffix_space" for sl in fam["slots"])
    if spaced and fam.get("quote", "none") == "none":
        fam["quote"] = "double_strings"
    return fam


def _near_miss_slot(fam, rng, p, role):
    tgt = _slot(fam, role)
    join = rng.choice(("_", "-"))
    if role == "latency":
        key = f"{rng.choice(p['distractor_prefixes'])}{join}{tgt['key'].rsplit('.', 1)[-1]}" if tgt and tgt.get("key") else None
        value = copy.deepcopy(tgt["value"]) if tgt else {"gen": "duration", "unit": "ms"}
        value.pop("numfmt", None) if value.get("numfmt") not in T.NUMFMTS else None
        return {"role": "distractor", "key": key, "value": value}
    return {"role": "distractor", "key": rng.choice(T.DISTRACTOR_IDS), "value": copy.deepcopy(tgt["value"]) if tgt else {"gen": "id"}}


def d3_reorder_insert(fam, rng, s, p):
    if fam["container"] == "free_text":
        return None
    slots = fam["slots"]
    if rng.random() < 0.5 + 0.5 * s:  # near-miss distractor immediately before the target it imitates
        role = rng.choice(("latency", "latency", "user"))
        new = _near_miss_slot(fam, rng, p, role)
        tgt = _slot(fam, role)
        if fam["container"] == "csv_positional":
            new.pop("key", None)
        elif not new.get("key") or any(sl.get("key") == new["key"] for sl in slots):
            return None
        if fam["container"] == "json_nested" and tgt and tgt.get("path"):
            new["path"] = list(tgt["path"])
        slots.insert(slots.index(tgt) if tgt else 0, new)
    else:
        rng.shuffle(slots)
        if fam["container"] != "csv_positional" and rng.random() < s:
            fam["order"] = "shuffle"
    return fam


def d4_unit(fam, rng, s, p):
    lat = _slot(fam, "latency")
    if not lat:
        return None
    v = lat["value"]
    ladder = ["ns", "us", "ms", "s", "min"]
    i = ladder.index(v["unit"])
    v["unit"] = ladder[max(0, min(4, i + rng.choice((-1, 1) if s < 0.5 else (-2, -1, 1, 2))))]
    v.pop("range_us", None)
    v.pop("unit_text", None)
    if v["unit"] in ("s", "min"):
        v["numfmt"] = "decimal"
    elif v.get("numfmt") == "decimal" and v["unit"] in ("ns", "us"):
        v["numfmt"] = "int"
    keyed = fam["container"] in _KEYED
    pos = v.get("unit_pos", "suffix")
    if keyed and (pos in ("key", "implicit") or rng.random() < 0.3):
        alias = T.base_key(lat["key"]) if lat["key"] not in T.IMPLICIT_UNITS else lat["key"]
        ns = lat["key"].rsplit(".", 1)[0] + "." if "." in lat["key"] and T.unit_from_key(lat["key"]) is None else ""
        _set_latency_key(lat, alias, rng, p, True)
        lat["key"] = ns + lat["key"]
    else:
        v["unit_pos"] = rng.choice(("suffix", "suffix", "suffix_space"))
        if v["unit"] in ("us", "s") and rng.random() < 0.3:
            v["unit_text"] = {"us": "µs", "s": "sec"}[v["unit"]]
    return fam


def d5_number_format(fam, rng, s, p):
    lat = _slot(fam, "latency")
    if not lat:
        return None
    cur = lat["value"].get("numfmt", "int")
    lat["value"]["numfmt"] = rng.choice([f for f in (("decimal", "thousands") if s < 0.5 else T.NUMFMTS) if f != cur])
    return fam


def d6_container(fam, rng, s, p):
    c = fam["container"]
    target = rng.choice([x for x in (("kv", "logfmt", "json_flat") if s < 0.5 else _KEYED) if x != c])
    if c not in _KEYED:  # positional and free-text formats have no keys: name the fields
        names = {"latency": "latency", "user": "user"}
        for i, sl in enumerate(fam["slots"]):
            sl["key"] = names.get(sl["role"]) or f"{sl['value']['gen']}{i if any(o is not sl and o['value']['gen'] == sl['value']['gen'] for o in fam['slots']) else ''}"
        fam.pop("template", None)
    fam["container"] = target
    if target == "kv":
        fam.setdefault("sep", " "), fam.setdefault("assign", "=")
        if fam["sep"] == ",":
            fam["sep"] = ", "
        fam["quote"] = "double_strings"
    if target == "query_string":
        fam["envelope"] = "<method> <path>?"
        for sl in fam["slots"]:
            if sl["value"].get("unit_pos") == "suffix_space":
                sl["value"]["unit_pos"] = "suffix"
    if target == "json_nested":
        groups = {"latency": rng.choice((["timing"], ["req"], ["http", "timing"])), "user": rng.choice((["user"], ["auth"], ["ctx", "user"]))}
        for sl in fam["slots"]:
            if sl["role"] in groups and sl["key"] not in groups[sl["role"]]:
                sl["path"] = groups[sl["role"]]
    return fam


def d7_quoting(fam, rng, s, p):
    c = fam["container"]
    if c in ("json_flat", "json_nested"):
        fam["json_spaced"] = not fam.get("json_spaced", False)
        usr = _slot(fam, "user")
        if usr and rng.random() < 0.5:
            usr["value"]["bare"] = not usr["value"].get("bare", False)
        return fam
    if c != "kv":
        return None
    fam["quote"] = rng.choice([q for q in T.QUOTES if q != fam.get("quote", "none")])
    if rng.random() < s:
        fam["assign"] = f" {fam.get('assign', '=').strip() or '='} "
    return fam


def d8_damage(fam, rng, s, p):
    fam["damage"] = {"kind": rng.choice(("truncate", "truncate", "corrupt")), "p": round(0.05 + 0.25 * s, 3)}
    return fam


def d13_nested_shadow(fam, rng, s, p):
    """A sub-object from another component carries the same leaf key as the target and comes first. A line regex has no notion of scope."""
    if fam["container"] not in ("json_flat", "json_nested"):
        return None
    scope = rng.choice(T.FOREIGN_SCOPES)
    for role in (["latency", "user"] if s > 0.5 else [rng.choice(["latency", "latency", "user"])]):
        tgt = _slot(fam, role)
        if not tgt or (tgt.get("path") or [None])[0] in T.FOREIGN_SCOPES:
            continue
        fam["slots"].insert(0, {"role": "distractor", "key": tgt["key"], "path": [scope], "value": copy.deepcopy(tgt["value"])})
    fam["container"] = "json_nested"
    return fam


def d14_quoted_echo(fam, rng, s, p):
    """A string field quotes a fragment of another line, with the target keys in it, before the real fields. A line regex has no notion of quoting."""
    if fam["container"] not in ("kv", "logfmt", "json_flat", "json_nested"):
        return None
    lat, usr = _slot(fam, "latency"), _slot(fam, "user")
    key = rng.choice(("msg", "error", "detail", "note", "last_event"))
    if not lat or not usr or any(sl.get("key") == key for sl in fam["slots"]):
        return None
    v = lat["value"]
    unit_text = {"suffix": v.get("unit_text", v["unit"]), "suffix_space": " " + v.get("unit_text", v["unit"])}.get(v.get("unit_pos", "suffix"), "")
    assign = fam.get("assign", "=") if fam["container"] == "kv" else "="
    fam["slots"].insert(0, {"role": "distractor", "key": key, "value": {"gen": "kv_echo", "keys": [lat["key"], usr["key"]], "assign": assign, "unit_text": unit_text}})
    if fam["container"] == "kv":
        fam["order"] = "fixed"
        if fam.get("quote", "none") == "none":
            fam["quote"] = "as_needed"  # quote the echoed fragment only; every other value keeps its step-0 surface
    return fam


OPERATORS = {"D1": d1_key_rename, "D2": d2_delimiters, "D3": d3_reorder_insert, "D4": d4_unit, "D5": d5_number_format,
             "D6": d6_container, "D7": d7_quoting, "D8": d8_damage, "D13": d13_nested_shadow, "D14": d14_quoted_echo}


def apply(op_id: str, fam: dict, rng: random.Random, severity: float, params: dict | None = None,
          operators: dict | None = None, tries: int = 8) -> dict | None:
    """Apply one operator. Returns a validated family, or None if the operator does not apply or never validates."""
    p = {**PARAMS, **(params or {})}
    fn = (operators or OPERATORS)[op_id]
    for _ in range(tries):
        out = fn(copy.deepcopy(fam), rng, severity, p)
        if out is None:
            return None
        out["defer_reason"] = T.surface_defer_reason(out)
        out.setdefault("trace", [])
        if out != fam and not T.check_family(out, n=150, seed=rng.randrange(2**31)):
            out["trace"] = [*fam.get("trace", []), op_id]
            return out
    return None
