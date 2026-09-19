"""The build-time teacher: an instruction-tuned LLM served by MLX that proposes format families as JSON.

The teacher sees the task card, the family schema and a few recently accepted families. It never sees a payload pool,
a label, or any held-out token, and nothing it writes is used as a label: a proposal that is not valid JSON, or that
fails validation, is rejected, never repaired. Apple silicon only; imported lazily by the curriculum build.
"""
from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path

from static_student.tasks import telemetry as T

LOCK = Path(__file__).resolve().parents[2] / "scripts/env/models.lock.txt"

SCHEMA_DOC = """A format family is ONE JSON object describing how one service writes one kind of log line. Fields:
- "envelope": text before the fields. Literal text plus any of <iso_ts> <date_time> <syslog_ts> <epoch> <host> <level> <component> <pid> <method> <path> <hex8>. May be "".
- "suffix": optional text after the fields (same placeholders).
- "container": one of "kv", "logfmt", "json_flat", "json_nested", "query_string", "csv_positional", "free_text".
- kv only: "sep" (between pairs), "assign" (between key and value), "quote": "none" | "double_all" | "double_strings" | "single_strings".
- json only: "json_spaced": true|false. json_nested slots may carry "path": ["parent", ...].
- free_text only: "template": a sentence that mentions every slot exactly once as {0}, {1}, ... (slot index).
- "order": "fixed" or "shuffle" (csv_positional and free_text must be "fixed").
- "slots": list of fields. Each slot: {"role": "latency" | "user" | "distractor", "key": "...", "value": {...}}. csv_positional and free_text slots have no "key".
  latency value: {"gen":"duration","unit":"ns|us|ms|s|min","unit_pos":"suffix|suffix_space|key|implicit|none","numfmt":"int|decimal|sci|thousands","unit_text": optional "µs" or "sec"}
    unit_pos "key": the key itself must end in the unit (latency_ms, elapsedMicros, rt.sec). unit_pos "implicit": only key "request_time" (s) or "took" (ms).
    unit_pos "none": a bare number whose key names no unit; such a line cannot be interpreted.
    numfmt "nonnumeric" writes "-", "NaN", "timeout"; numfmt "overflow" writes an absurdly large number.
  user value: {"gen":"id","scheme":"decimal|hex|uuid|prefixed","prefix": optional like "u-"}; scheme "too_long" writes an id longer than 36 bytes.
  distractor value: {"gen": one of "duration" (needs "unit"), "id", "hostname", "status", "path", "bytes", "int", "ip", "method", "level", "bool", "iso_ts", "free_text", "choice" (needs "choices": [...])}.
- optional "records": 2 with "record_sep" puts two records on one line.
- optional "damage": {"kind":"truncate"|"corrupt","p":1.0} damages every line.
Lines must stay under 256 bytes: at most 6 slots, short envelopes. A distractor key must never be a plain synonym of the target fields."""

BUCKET_DOC = {
    "clean": "Bucket CLEAN: an ordinary, realistic format that carries exactly one request latency and exactly one user id. Vary the service type, container, key naming style, unit and number format.",
    "near_miss": "Bucket NEAR-MISS: like CLEAN, but add one or two distractors that look like the target and sit right before or after it: another duration (database time, upstream latency, a timeout, a percentile) with the same unit style, or another id (request, trace, session, tenant).",
    "should_defer": "Bucket SHOULD-DEFER: a line from which the target must NOT be extracted. Use exactly this construction: {reason}.",
}
DEFER_DOC = {
    "target_missing": "leave out the latency slot or the user slot entirely",
    "ambiguous_latency": "two different latency slots with two different plausible latency keys, no rule to prefer one",
    "non_numeric": 'the latency value uses "numfmt":"nonnumeric"',
    "unit_unknown": 'the latency uses "unit_pos":"none" with a key that names no unit (not request_time, not took)',
    "out_of_range": 'the latency value uses "numfmt":"overflow"',
    "id_too_long": 'the user value uses "scheme":"too_long"',
    "multi_record": 'set "records": 2 and a "record_sep"',
    "truncated": 'set "damage": {"kind":"truncate","p":1.0}; quote the values or put the unit as a suffix so the cut is visible',
    "corrupted": 'set "damage": {"kind":"corrupt","p":1.0}',
}
_EXAMPLE = {"envelope": "<iso_ts> <level> <component>: ", "container": "kv", "sep": " ", "assign": "=", "quote": "none", "order": "shuffle",
            "slots": [{"role": "distractor", "key": "upstream_latency", "value": {"gen": "duration", "unit": "ms"}},
                      {"role": "latency", "key": "lat", "value": {"gen": "duration", "unit": "ms", "unit_pos": "suffix", "numfmt": "int"}},
                      {"role": "user", "key": "usr", "value": {"gen": "id", "scheme": "decimal"}},
                      {"role": "distractor", "key": "host", "value": {"gen": "hostname"}}]}


def _pinned() -> tuple[str, str]:
    line = next(l for l in LOCK.read_text().splitlines() if l and not l.startswith("#"))
    repo, rev = line.split("@")
    return repo, rev


def extract_json(text: str) -> dict | None:
    """The first balanced JSON object in the reply, or None. Code fences are tolerated; nothing is repaired."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


class MLXTeacher:
    def __init__(self, temperature: float = 0.9, max_tokens: int = 450, seed: int = 0):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        self.repo, self.revision = _pinned()
        self.name = f"{self.repo}@{self.revision[:8]}"
        self.model, self.tok = load(self.repo, revision=self.revision, tokenizer_config={"fix_mistral_regex": True})
        self.sampler, self.greedy = make_sampler(temp=temperature, top_p=0.95), make_sampler(temp=0.0)
        self.max_tokens, self.settings = max_tokens, {"temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens, "seed": seed}
        mx.random.seed(seed)
        self.transcript: list[dict] = []

    def _chat(self, system: str, user: str, sampler, max_tokens: int) -> str:
        from mlx_lm import generate
        prompt = f"[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}[/INST]"
        out = generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler)
        self.transcript.append({"system_sha": hash(system) & 0xFFFFFFFF, "user": user, "reply": out})
        return out

    def task_card(self, spec: dict) -> dict:
        """S1: the only place where the English target line is interpreted. Synonyms are filtered to usable keys."""
        user = (f'Semantic target of a log-field extractor: "{spec["target"]}".\nReturn ONE JSON object with: "latency_keys": 12 short key names real services use for '
                'the end-to-end request latency (no unit in the name), "user_keys": 10 key names for the acting user or principal id, "near_miss_durations": 8 key names of '
                'OTHER durations that are not the request latency, "near_miss_ids": 8 key names of OTHER ids that are not the user. snake_case, ASCII, no explanations.')
        raw = extract_json(self._chat("You are a senior SRE. You answer with JSON only.", user, self.greedy, 500)) or {}
        ok = lambda k: isinstance(k, str) and re.fullmatch(r"[a-z][a-z0-9_]{1,23}", k) and T.unit_from_key(k) is None  # noqa: E731
        lat = list(dict.fromkeys([*T.K_LAT, *filter(ok, raw.get("latency_keys", []))]))
        usr = list(dict.fromkeys([*T.K_USR, *filter(ok, raw.get("user_keys", []))]))
        targets = {T.base_key(k) for k in lat + usr}
        nm = lambda name, base: list(dict.fromkeys([*base, *[k for k in filter(ok, raw.get(name, [])) if T.base_key(k) not in targets]]))  # noqa: E731
        return {"target": spec["target"], "fields": [
                    {"role": "latency", "type": "duration -> uint64 microseconds", "synonyms": lat, "units": list(T.UNITS)},
                    {"role": "user", "type": "byte string <= 36", "synonyms": usr, "schemes": list(T.ID_SCHEMES)}],
                "near_miss": {"durations": nm("near_miss_durations", T.DISTRACTOR_DURATIONS), "ids": nm("near_miss_ids", T.DISTRACTOR_IDS)},
                "defer_reasons": list(DEFER_DOC), "containers": list(T.CONTAINERS), "teacher_raw": raw}

    def propose(self, card: dict, bucket: str, examples: list[dict], rng: random.Random) -> dict | None:
        system = ("You design synthetic log formats for testing a field extractor. You answer with exactly one JSON object and nothing else.\n\n" + SCHEMA_DOC
                  + "\n\nExample family:\n" + json.dumps(_EXAMPLE))
        reason = rng.choice(card["defer_reasons"]) if bucket == "should_defer" else None
        hint = {"container": rng.choices(T.CONTAINERS, weights=(5, 3, 3, 2, 1, 1, 2))[0], "unit": rng.choices(list(T.UNITS), weights=(1, 2, 6, 4, 1))[0]}
        shown = [{k: v for k, v in copy.deepcopy(f).items() if k not in ("family_id", "bucket", "trace", "defer_reason")} for f in examples]
        user = (f'Target: "{card["target"]}".\nLatency key ideas: {", ".join(card["fields"][0]["synonyms"])}.\nUser key ideas: {", ".join(card["fields"][1]["synonyms"])}.\n'
                f'Near-miss durations: {", ".join(card["near_miss"]["durations"])}. Near-miss ids: {", ".join(card["near_miss"]["ids"])}.\n\n'
                + BUCKET_DOC[bucket].format(reason=DEFER_DOC.get(reason, "")) + f'\nUse container "{hint["container"]}" and latency unit "{hint["unit"]}" unless the construction forbids it.\n'
                + ("Be different from these already accepted families:\n" + "\n".join(json.dumps(f) for f in shown) + "\n" if shown else "") + "Return the JSON object now.")
        fam = extract_json(self._chat(system, user, self.sampler, self.max_tokens))
        if fam is None or not isinstance(fam.get("slots"), list) or not all(isinstance(s, dict) and isinstance(s.get("value"), dict) for s in fam["slots"]):
            return None
        fam = {k: v for k, v in fam.items() if k in ("envelope", "suffix", "container", "sep", "assign", "quote", "json_spaced", "template", "order", "slots", "records", "record_sep", "damage")}
        fam["bucket"] = bucket
        try:
            fam["defer_reason"] = T.surface_defer_reason(fam) if fam.get("container") in T.CONTAINERS and all("role" in s for s in fam["slots"]) else None
        except (KeyError, TypeError):
            return None
        return fam
