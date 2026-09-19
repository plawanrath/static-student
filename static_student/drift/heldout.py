"""Held-out drift: operators D9-D12 and the held-out parameter lists for D1-D3.

Nothing in this module is visible to the teacher, the curriculum package or model selection. It is imported only by
the timeline sampler and the evaluation scripts. tests/test_heldout_isolation.py enforces both the import rule and
that none of the tokens below appears in a curriculum prompt, task card or accepted family.
"""
from __future__ import annotations

from static_student.drift.operators import _slot

# Unseen values for the in-curriculum operators (the "held-out parameters" track).
PARAMS = {
    "lat_aliases": ["turnaround", "servicing", "wall", "e2e", "sojourn", "cost", "spent", "handling"],
    "usr_aliases": ["actor", "subject", "member", "requester", "customer", "who", "caller", "identity"],
    "namespaces": ["labels", "meta", "evt", "attrs"],
    "delims": [(" ", "=>"), (" :: ", "="), (" ", "->"), ("; ", "="), (" / ", ": "), ("  ", "=")],
    "distractor_prefixes": ["origin", "tls", "acl", "gc", "disk", "mutex"],
}

_ENVELOPES = ["<iso_ts> stdout F ", "[<epoch>] ", "<host> <component>[<pid>] <level>: ", "<hex8> <iso_ts> | ", "<level>\t<date_time>\t"]
_BLOBS = [' {"k8s":{"pod":"<host>","ns":"prod"}}', ' {"trace":"<hex8><hex8>","sampled":true}', " -- <host> pid=<pid>", ' #{"v":2,"dc":"<host>"}']
_COLLISIONS = {"latency": ["{k}_budget", "{k}_slo", "{k}_target", "max_{k}", "{k}_limit"],
               "user": ["{k}_agent_id", "{k}_quota", "{k}_group", "last_{k}", "{k}_tz"]}
_ID_PATTERNS = ["dddd-xxxx", "urn:user:ddddd", "aaaaa.aaaaaa@corp", "dddddd/aa", "ddddddddddxxxxxxxxxxxxxxxx", "aa:dddd:dddd", "dddd+aaa"]
_TRANSLATIONS = {"latency": ["latenz", "dauer", "latence", "duree", "latencia", "duracion", "tempo"],
                 "user": ["benutzer", "nutzer", "utilisateur", "usuario", "utente", "gebruiker"]}
_KEYED = ("kv", "logfmt", "json_flat", "json_nested", "query_string")


def d9_envelope(fam, rng, s, p):
    if rng.random() < 0.5 or fam["container"] == "query_string":
        if fam["container"] == "query_string":
            return None
        fam["envelope"] = rng.choice(_ENVELOPES)
    else:
        fam["suffix"] = rng.choice(_BLOBS)
    return fam


def d10_collision(fam, rng, s, p):
    if fam["container"] not in _KEYED:
        return None
    for role in (["latency", "user"] if s > 0.5 else [rng.choice(["latency", "user"])]):
        tgt = _slot(fam, role)
        if not tgt:
            continue
        key = rng.choice(_COLLISIONS[role]).format(k=tgt["key"].rsplit(".", 1)[-1])
        if any(sl.get("key") == key for sl in fam["slots"]):
            continue
        value = dict(tgt["value"]) if role == "user" else {"gen": "duration", "unit": tgt["value"]["unit"], "unit_pos": tgt["value"].get("unit_pos", "suffix")}
        new = {"role": "distractor", "key": key, "value": value}
        if tgt.get("path"):
            new["path"] = list(tgt["path"])
        fam["slots"].insert(fam["slots"].index(tgt), new)
    return fam


def d11_id_scheme(fam, rng, s, p):
    usr = _slot(fam, "user")
    if not usr:
        return None
    usr["value"] = {"gen": "id", "scheme": "pattern", "pattern": rng.choice(_ID_PATTERNS)}
    return fam


def d12_localization(fam, rng, s, p):
    lat, usr = _slot(fam, "latency"), _slot(fam, "user")
    if not lat or not usr:
        return None
    if rng.random() < 0.4 and lat["value"].get("numfmt", "int") in ("int", "decimal"):
        lat["value"]["numfmt"] = "decimal_comma"  # ambiguous with a thousands group elsewhere: gold is DEFER
        if lat["value"]["unit"] in ("ns", "us"):
            lat["value"]["unit"] = "ms"
            if lat["value"].get("unit_pos") in ("key", "implicit"):
                return None
        return fam
    if fam["container"] not in _KEYED:
        return None
    usr["key"] = rng.choice(_TRANSLATIONS["user"])
    word = rng.choice(_TRANSLATIONS["latency"])
    v = lat["value"]
    if v.get("unit_pos", "suffix") in ("key", "implicit"):
        lat["key"], v["unit_pos"] = f"{word}_{v['unit']}", "key"
    else:
        lat["key"] = word
    return fam


OPERATORS = {"D9": d9_envelope, "D10": d10_collision, "D11": d11_id_scheme, "D12": d12_localization}


def tokens() -> set[str]:
    """Every held-out surface token, for the isolation test."""
    out = set(PARAMS["lat_aliases"] + PARAMS["usr_aliases"] + PARAMS["namespaces"] + PARAMS["distractor_prefixes"])
    out |= {w for ws in _TRANSLATIONS.values() for w in ws}
    out |= {"budget", "slo", "quota"}  # the distinctive collision words; "max", "target", ... are ordinary English
    return out


if __name__ == "__main__":  # token list for curriculum validation: the curriculum package reads a file, never this module
    print("\n".join(sorted(tokens())))
