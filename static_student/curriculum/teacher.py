"""Family proposers. A teacher maps (task card, bucket, recent accepted families) to one candidate family.

`StubTeacher` samples the family schema directly and needs no model; it exists so that the renderer, the validators
and the pool builder are exercised before the real teacher runs, and it doubles as a hand-authored generator that
owes nothing to the teacher.
"""
from __future__ import annotations

import copy
import random

from static_student.drift import operators
from static_student.tasks import telemetry as T

BUCKETS = {"clean": 0.35, "near_miss": 0.25, "drifted": 0.25, "should_defer": 0.15}

_ENVELOPES = ["", "", "<iso_ts> ", "<iso_ts> <level> <component>: ", "<date_time> <level> [<component>] ", "<syslog_ts> <host> app[<pid>]: ",
              "<epoch> ", "[<iso_ts>] ", "<host> <level> ", "<date_time> <host> <component> - "]
_TEMPLATES = ["request {d0} served for user {usr} in {lat}", "user={usr} finished {d0} took {lat}", "{d0} {d1} by {usr} ({lat})",
              "completed in {lat}: user {usr}, path {d0}", "[{usr}] {d0} {d1} {lat}", "slow request: {lat} elapsed, principal {usr}, status {d0}"]
_DISTRACTORS = [("host", {"gen": "hostname"}), ("status", {"gen": "status"}), ("path", {"gen": "path"}), ("bytes", {"gen": "bytes"}),
                ("method", {"gen": "method"}), ("client_ip", {"gen": "ip"}), ("level", {"gen": "level"}), ("ts", {"gen": "iso_ts"}),
                ("cached", {"gen": "bool"}), ("retries", {"gen": "int", "lo": 0, "hi": 5}), ("msg", {"gen": "free_text"}),
                ("region", {"gen": "choice", "choices": ["us-east-1", "eu-west-1", "ap-south-1"]})]


class StubTeacher:
    name = "stub"

    def task_card(self, spec: dict) -> dict:
        return {"target": spec["target"], "fields": [
                    {"role": "latency", "type": "duration -> uint64 microseconds", "synonyms": list(T.K_LAT), "units": list(T.UNITS)},
                    {"role": "user", "type": "byte string <= 36", "synonyms": list(T.K_USR), "schemes": list(T.ID_SCHEMES)}],
                "near_miss": {"durations": list(T.DISTRACTOR_DURATIONS), "ids": list(T.DISTRACTOR_IDS)},
                "defer_reasons": [r for r in T.DEFER_REASONS if r != "decimal_comma"], "containers": list(T.CONTAINERS)}

    def _clean(self, card: dict, rng: random.Random) -> dict:
        c = rng.choices(T.CONTAINERS, weights=(5, 3, 3, 2, 1, 1, 2))[0]
        keyed = c not in ("csv_positional", "free_text")
        unit = rng.choices(list(T.UNITS), weights=(1, 2, 6, 4, 1))[0]
        alias = rng.choice(card["fields"][0]["synonyms"])
        lat = {"role": "latency", "value": {"gen": "duration", "unit": unit, "numfmt": rng.choices(T.NUMFMTS, weights=(6, 4, 1, 1))[0]}}
        if keyed and alias in T.IMPLICIT_UNITS:
            lat["key"] = alias
            lat["value"].update(unit=T.IMPLICIT_UNITS[alias], unit_pos="implicit")
        elif keyed:
            operators._set_latency_key(lat, alias, rng, operators.PARAMS, unit_in_key=rng.random() < 0.35)
        if lat["value"].get("unit_pos") not in ("key", "implicit"):
            lat["value"]["unit_pos"] = rng.choice(("suffix", "suffix", "suffix_space"))
            if unit in ("us", "s") and rng.random() < 0.2:
                lat["value"]["unit_text"] = {"us": "µs", "s": "sec"}[unit]
        if lat["value"]["unit"] in ("s", "min") and lat["value"]["numfmt"] == "int":
            lat["value"]["numfmt"] = "decimal"
        usr = {"role": "user", "value": {"gen": "id", "scheme": rng.choice(card["fields"][1]["schemes"])}}
        if keyed:
            usr["key"] = operators._style(rng, rng.choice(card["fields"][1]["synonyms"]).replace("userId", "user_id").split("_"))
        ds = [{"role": "distractor", "key": k, "value": dict(v)} for k, v in rng.sample(_DISTRACTORS, rng.randrange(0, 5))]
        if not keyed:
            ds = [d for d in ds if d["value"]["gen"] != "free_text"][:2] or [{"role": "distractor", "key": "path", "value": {"gen": "path"}}]
            for d in ds:
                d.pop("key")
        slots = [lat, usr, *ds]
        rng.shuffle(slots)
        fam = {"envelope": rng.choice(_ENVELOPES), "container": c, "slots": slots, "order": "fixed"}
        if c == "kv":
            sep, assign = rng.choice(operators.PARAMS["delims"])
            fam.update(sep=sep, assign=assign, quote=rng.choice(T.QUOTES), order=rng.choice(("fixed", "shuffle")))
        elif c == "csv_positional":
            fam["sep"] = rng.choice((",", ";", "\t", "|"))
        elif c == "free_text":
            t = rng.choice(_TEMPLATES)
            names = {"lat": slots.index(lat), "usr": slots.index(usr)}
            names.update({f"d{j}": slots.index(d) for j, d in enumerate(ds)})
            if t.count("{d") != len(ds):
                t = " ".join(["{lat}", "{usr}"] + [f"{{d{j}}}" for j in range(len(ds))]) if rng.random() < 0.3 else \
                    "user {usr} " + " ".join(f"{{d{j}}}" for j in range(len(ds))) + " in {lat}"
            fam["template"] = t.format(**{k: "{%d}" % v for k, v in names.items()})
        elif c == "json_nested":
            for s in slots:
                if rng.random() < 0.6:
                    s["path"] = [rng.choice(("req", "http", "timing", "ctx", "data"))]
            fam["json_spaced"] = rng.random() < 0.5
        elif c == "json_flat":
            fam["json_spaced"] = rng.random() < 0.5
        elif c == "query_string":
            fam.update(envelope=rng.choice(("<method> <path>?", "<host> <method> <path>?", "GET /track?")), suffix=rng.choice(("", " HTTP/1.1")))
        return fam

    def propose(self, card: dict, bucket: str, examples: list[dict], rng: random.Random) -> dict:
        fam = self._clean(card, rng)
        lat = next(s for s in fam["slots"] if s["role"] == "latency")
        usr = next(s for s in fam["slots"] if s["role"] == "user")
        keyed = fam["container"] not in ("csv_positional", "free_text")
        if bucket == "near_miss" and fam["container"] != "free_text":
            for _ in range(rng.randrange(1, 3)):
                if rng.random() < 0.6:
                    d = {"role": "distractor", "key": rng.choice(card["near_miss"]["durations"]), "value": copy.deepcopy(lat["value"])}
                    d["value"].update(unit_pos=rng.choice(("suffix", "suffix_space")) if lat["value"]["unit_pos"] in ("key", "implicit") else lat["value"]["unit_pos"])
                    at = fam["slots"].index(lat)
                else:
                    d = {"role": "distractor", "key": rng.choice(card["near_miss"]["ids"]), "value": copy.deepcopy(usr["value"])}
                    at = fam["slots"].index(usr)
                if not keyed:
                    d.pop("key")
                elif any(s.get("key") == d["key"] for s in fam["slots"]):
                    continue
                fam["slots"].insert(at + rng.choice((0, 0, 1)), d)
        elif bucket == "should_defer":
            reason = rng.choice(card["defer_reasons"])
            if reason == "target_missing":
                fam["slots"].remove(rng.choice((lat, usr)))
                if fam["container"] == "free_text":
                    fam["container"], fam["sep"], fam["assign"] = "kv", " ", "="
                    for i, s in enumerate(fam["slots"]):
                        s.setdefault("key", {"latency": "latency", "user": "user"}.get(s["role"], f"f{i}"))
            elif reason == "ambiguous_latency" and keyed:
                other = copy.deepcopy(lat)
                other["key"] = rng.choice([k for k in operators.PARAMS["lat_aliases"] if T.base_key(k) != T.base_key(lat["key"])])
                other["value"].update(unit_pos="suffix") if other["value"]["unit_pos"] in ("key", "implicit") else None
                fam["slots"].insert(rng.randrange(len(fam["slots"]) + 1), other)
            elif reason in ("non_numeric", "out_of_range"):
                lat["value"]["numfmt"] = {"non_numeric": "nonnumeric", "out_of_range": "overflow"}[reason]
            elif reason == "unit_unknown" and keyed and lat["value"]["unit_pos"] not in ("key", "implicit"):
                lat["value"]["unit_pos"] = "none"
            elif reason == "id_too_long":
                usr["value"] = {"gen": "id", "scheme": "too_long"}
            elif reason == "multi_record":
                fam.update(records=2, record_sep=rng.choice((" | ", " ;; ", "  ", " ## ")))
            else:
                fam["damage"] = {"kind": "corrupt" if reason == "corrupted" else "truncate", "p": 1.0}
        fam["bucket"] = bucket
        fam["defer_reason"] = T.surface_defer_reason(fam)
        return fam
