"""Telemetry task: format families, a renderer whose gold labels hold by construction, and an inverse renderer.

A *family* is a small JSON object (container, envelope, slots, quoting, order). `render(family, rng)` expands it into
one payload and returns the gold label together with the byte spans of the two targets; the spans are recorded while
the payload is assembled, never located afterwards. `inverse(family, payload)` parses a payload with the family's own
format knowledge and is used to validate families (`check_family`).

Gold is a function of the payload bytes alone. Every rule that turns a payload into DEFER (unknown unit, non-numeric
latency, two candidate latencies, visible truncation, ...) is decidable from the surface, so two families can never
assign different labels to the same bytes. `check_family` rejects families that break this.

  to_micros(number_text, unit)      exact decimal -> microseconds, truncated toward zero (mirrored by the C kernel)
  validate_family(fam)              schema and consistency errors, as a list of strings
  render(fam, rng)                  -> Row
  inverse(fam, payload)             -> (latency_us, user_id) or None for DEFER
  check_family(fam, n, seed)        render n samples, compare with the inverse renderer, enforce the length limit
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

MAX_PAYLOAD = 256
USER_ID_MAX = 36
UINT64_MAX = 2**64 - 1

UNITS = {"ns": Fraction(1, 1000), "us": Fraction(1), "ms": Fraction(1000), "s": Fraction(10**6), "min": Fraction(60 * 10**6)}
UNIT_SURFACES = {"ns": "ns", "us": "us", "µs": "us", "ms": "ms", "s": "s", "sec": "s", "min": "min"}
KEY_UNIT_TOKENS = {
    "ns": "ns", "nanos": "ns", "nsec": "ns",
    "us": "us", "micros": "us", "usec": "us",
    "ms": "ms", "millis": "ms", "msec": "ms",
    "s": "s", "sec": "s", "secs": "s", "seconds": "s",
    "min": "min", "mins": "min", "minutes": "min",
}
IMPLICIT_UNITS = {"request_time": "s", "took": "ms"}  # conventions of widely used servers; all other bare numbers defer

K_LAT = ("latency", "lat", "elapsed", "duration", "rt", "resp_time", "took", "time", "request_time")
K_USR = ("user", "usr", "uid", "user_id", "userId", "account", "principal", "username", "acct")
DISTRACTOR_DURATIONS = ("db_time", "ttfb", "upstream_latency", "timeout", "latency_p99", "queue_wait", "dns_time")
DISTRACTOR_IDS = ("request_id", "trace_id", "session", "tenant", "target_user", "span_id", "device")

FOREIGN_SCOPES = ("upstream", "db", "backend", "cache", "queue", "dns")  # a JSON sub-object under one of these is another component's data

CONTAINERS = ("kv", "logfmt", "json_flat", "json_nested", "query_string", "csv_positional", "free_text")
QUOTES = ("none", "double_all", "double_strings", "single_strings", "as_needed")
DEFER_REASONS = ("target_missing", "ambiguous_latency", "non_numeric", "unit_unknown", "truncated", "corrupted",
                 "out_of_range", "id_too_long", "multi_record", "decimal_comma")
NUMFMTS = ("int", "decimal", "sci", "thousands")
ID_SCHEMES = ("decimal", "hex", "uuid", "prefixed")

MAX_NUMBER_BYTES = 40  # longer tokens are not latencies; the bound also keeps the exact conversion cheap for any span a model can point at
_NUM_RE = re.compile(rb"(?:[1-9]\d{0,2}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d{1,2})?\Z")
_LAT_TOKEN_RE = re.compile(rb"(?:[1-9]\d{0,2}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d{1,2})? ?(?:ns|us|\xc2\xb5s|ms|sec|s|min)\Z")
_ID_RE = re.compile(rb"[A-Za-z0-9._:@/+-]+\Z")
_JSON_NUM_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_KEY_TAIL_RES = (re.compile(r"(?<=[a-z0-9])([A-Z][a-z]*)\Z"), re.compile(r"[_.\-]([A-Za-z]+)\Z"))  # latencyMs, latency_ms


# ---------------------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Row:
    payload: bytes
    defer: bool
    reason: str | None = None
    latency_us: int | None = None
    user_id: str | None = None
    lat_span: tuple[int, int] | None = None  # bytes of the number only; the unit is a separate class label
    unit: str | None = None
    usr_span: tuple[int, int] | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["payload"] = self.payload.decode("latin-1")  # lossless for arbitrary bytes; encode("latin-1") restores
        return d

    @staticmethod
    def from_dict(d: dict) -> "Row":
        d = dict(d)
        d["payload"] = d["payload"].encode("latin-1")
        for k in ("lat_span", "usr_span"):
            d[k] = tuple(d[k]) if d.get(k) is not None else None
        return Row(**d)


def to_micros(number_text: bytes | str, unit: str) -> int | None:
    """Exact conversion of a rendered number to microseconds, truncated toward zero. None if not a number or out of
    range. Commas are accepted only as thousands separators (groups of exactly three digits)."""
    raw = number_text.encode() if isinstance(number_text, str) else bytes(number_text)
    if len(raw) > MAX_NUMBER_BYTES or not _NUM_RE.match(raw):
        return None
    try:
        d = Decimal(raw.decode().replace(",", ""))
    except InvalidOperation:
        return None
    n, den = d.as_integer_ratio()
    v = Fraction(n, den) * UNITS[unit]
    us = v.numerator // v.denominator
    return us if 0 <= us <= UINT64_MAX else None


def _unit_tail(key: str) -> tuple[int, str] | None:
    for rx in _KEY_TAIL_RES:
        m = rx.search(key)
        if m and m.group(1).lower() in KEY_UNIT_TOKENS:
            return m.start(), KEY_UNIT_TOKENS[m.group(1).lower()]
    return None


def unit_from_key(key: str) -> str | None:
    """Unit carried by a key suffix: latency_ms, latencyMs, svc.latency.us -> unit. None if the key carries none."""
    t = _unit_tail(key)
    return t[1] if t else None


def base_key(key: str) -> str:
    """Alias comparison form: namespace prefix and unit suffix removed, case and punctuation folded."""
    t = _unit_tail(key)
    k = (key[: t[0]] if t else key).rsplit(".", 1)[-1]
    return re.sub(r"[_\-.]", "", k).lower()


# ---------------------------------------------------------------------------------------------------------------
# Value generators (closed set; the teacher selects and parameterizes them, it never writes code)
# ---------------------------------------------------------------------------------------------------------------
_HOSTS = ("web", "api", "edge", "db", "cache", "auth", "gw", "worker", "ingest", "queue")
_LEVELS = ("INFO", "WARN", "ERROR", "DEBUG", "info", "warn", "error")
_COMPONENTS = ("http.server", "RequestHandler", "gateway", "authz", "api", "router", "ingest-svc", "c.e.web.Filter")
_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")
_PATH_PARTS = ("api", "v1", "v2", "items", "users", "orders", "search", "cart", "login", "static", "health", "export")
_WORDS = ("request", "completed", "ok", "cache", "miss", "hit", "retry", "upstream", "slow", "served", "handled",
          "response", "sent", "client", "closed", "done")
_ID_PREFIXES = ("u-", "usr_", "acct-", "U", "user:", "emp")
_NONNUMERIC = ("-", "NaN", "timeout", "null", "n/a", "inf")
_DEFAULT_RANGE_US = {"ns": (1, 5 * 10**5), "us": (50, 5 * 10**6), "ms": (10**3, 3 * 10**7),
                     "s": (10**4, 1.2 * 10**8), "min": (6 * 10**6, 3.6 * 10**9)}

Seg = tuple[str, str | None]  # (text, tag) with tag in {"lat", "usr", None}


def _number_text(rng: random.Random, us: float, unit: str, numfmt: str) -> str:
    x = float(Fraction(us) / UNITS[unit])
    if numfmt == "int":
        return str(max(1, round(x)))
    if numfmt == "decimal":
        return f"{x:.{rng.choice((1, 2, 3) if unit in ('ns', 'us', 'ms') else (2, 3, 3))}f}"
    if numfmt == "sci":
        m, e = f"{x:.{rng.choice((1, 2, 3))}e}".split("e")
        style = rng.choice(("e+02", "e2", "E+02"))
        exp = int(e)
        if style == "e2":
            return f"{m}e{exp}"
        return f"{m}{style[0]}{'+' if exp >= 0 else '-'}{abs(exp):02d}"
    if numfmt == "thousands":
        n = max(1, round(x))
        return f"{n:,}" if rng.random() < 0.7 else f"{x:,.1f}"
    if numfmt == "decimal_comma":  # never three fractional digits, so it cannot be read as a thousands group
        return f"{x:.{rng.choice((1, 2))}f}".replace(".", ",")
    raise ValueError(numfmt)


def _gen_duration(rng: random.Random, v: dict) -> list[Seg]:
    unit, tag = v["unit"], v.get("_tag")
    numfmt = v.get("numfmt", "int")
    if numfmt == "nonnumeric":
        return [(rng.choice(_NONNUMERIC), None)]
    if numfmt == "overflow":
        num = str(rng.randrange(10**20, 10**24))
    else:
        lo, hi = v.get("range_us") or _DEFAULT_RANGE_US[unit]
        us = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        num = _number_text(rng, us, unit, numfmt)
    pos = v.get("unit_pos", "suffix")
    segs: list[Seg] = [(num, tag)]
    if pos == "suffix":
        segs.append((v.get("unit_text", unit), None))
    elif pos == "suffix_space":
        segs.append((" " + v.get("unit_text", unit), None))
    return segs  # "key", "implicit" and "none" put nothing after the number


def _gen_id(rng: random.Random, v: dict) -> list[Seg]:
    scheme = v.get("scheme", "decimal")
    if scheme == "decimal":
        s = str(rng.randrange(10 ** rng.randrange(3, 10)))
    elif scheme == "hex":
        s = f"{rng.getrandbits(4 * (n := rng.choice((8, 12, 16, 24, 32)))):0{n}x}"
    elif scheme == "uuid":
        h = f"{rng.getrandbits(128):032x}"
        s = f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
        s = s.upper() if v.get("upper") else s
    elif scheme == "prefixed":
        s = v.get("prefix", rng.choice(_ID_PREFIXES)) + str(rng.randrange(10 ** rng.randrange(3, 9)))
    elif scheme == "too_long":
        s = f"{rng.getrandbits(4 * (n := rng.randrange(38, 56))):0{n}x}"
    elif scheme == "pattern":  # letters: d digit, x hex digit, a lowercase letter; everything else literal
        s = "".join(rng.choice("0123456789") if c == "d" else rng.choice("0123456789abcdef") if c == "x"
                    else rng.choice("abcdefghijklmnopqrstuvwxyz") if c == "a" else c for c in v["pattern"])
    else:
        raise ValueError(scheme)
    return [(s, v.get("_tag"))]


def _gen_kv_echo(rng: random.Random, v: dict) -> list[Seg]:
    """A quoted fragment of some other log line: words, then key=value pairs. Inside a string value it is text, not a field."""
    pairs = [f"{k}{v.get('assign', '=')}{val}" for k, val in zip(v["keys"], (f"{rng.randrange(1, 900)}{v.get('unit_text', '')}", f"{rng.choice(_ID_PREFIXES)}{rng.randrange(10, 99999)}"))]
    return [(" ".join([rng.choice(_WORDS), *pairs]), None)]


def _ts(rng: random.Random) -> tuple[int, int, int, int, int, int, int]:
    return (rng.randrange(2024, 2027), rng.randrange(1, 13), rng.randrange(1, 29), rng.randrange(24),
            rng.randrange(60), rng.randrange(60), rng.randrange(1000))


def _plain(fn):
    return lambda rng, v: [(fn(rng, v), None)]


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
GENERATORS = {
    "duration": _gen_duration,
    "id": _gen_id,
    "kv_echo": _gen_kv_echo,
    "hostname": _plain(lambda r, v: f"{r.choice(_HOSTS)}-{r.randrange(1, 40):02d}" + (".prod.internal" if r.random() < 0.3 else "")),
    "status": _plain(lambda r, v: str(r.choice((200, 200, 200, 201, 204, 301, 304, 400, 401, 403, 404, 429, 500, 502, 503)))),
    "path": _plain(lambda r, v: "/" + "/".join(r.choice(_PATH_PARTS) for _ in range(r.randrange(1, 4)))),
    "bytes": _plain(lambda r, v: str(r.randrange(0, 10 ** r.randrange(2, 8)))),
    "int": _plain(lambda r, v: str(r.randrange(v.get("lo", 0), v.get("hi", 1000)))),
    "ip": _plain(lambda r, v: ".".join(str(r.randrange(1, 255)) for _ in range(4))),
    "method": _plain(lambda r, v: r.choice(_METHODS)),
    "level": _plain(lambda r, v: r.choice(_LEVELS)),
    "bool": _plain(lambda r, v: r.choice(("true", "false"))),
    "choice": _plain(lambda r, v: r.choice(v["choices"])),
    "free_text": _plain(lambda r, v: " ".join(r.choice(_WORDS) for _ in range(r.randrange(2, 5)))),
    "iso_ts": _plain(lambda r, v: "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ" % _ts(r)),
}

_ENVELOPE_FIELDS = {
    "<iso_ts>": (lambda r: "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ" % _ts(r), r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z"),
    "<date_time>": (lambda r: "%04d-%02d-%02d %02d:%02d:%02d,%03d" % _ts(r), r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}"),
    "<syslog_ts>": (lambda r: (lambda t: "%s %2d %02d:%02d:%02d" % (_MONTHS[t[1] - 1], t[2], t[3], t[4], t[5]))(_ts(r)),
                    r"[A-Z][a-z]{2} [ \d]\d \d\d:\d\d:\d\d"),
    "<epoch>": (lambda r: str(r.randrange(1_700_000_000, 1_790_000_000)), r"\d{10}"),
    "<host>": (lambda r: f"{r.choice(_HOSTS)}-{r.randrange(1, 40):02d}", r"[a-z]+-\d\d"),
    "<level>": (lambda r: r.choice(_LEVELS), r"[A-Za-z]+"),
    "<component>": (lambda r: r.choice(_COMPONENTS), r"[\w.\-]+"),
    "<pid>": (lambda r: str(r.randrange(100, 65000)), r"\d+"),
    "<method>": (lambda r: r.choice(_METHODS), r"[A-Z]+"),
    "<path>": (lambda r: "/" + "/".join(r.choice(_PATH_PARTS) for _ in range(r.randrange(1, 4))), r"[/\w]+"),
    "<hex8>": (lambda r: f"{r.getrandbits(32):08x}", r"[0-9a-f]{8}"),
}
_ENV_TOKEN_RE = re.compile("|".join(re.escape(k) for k in _ENVELOPE_FIELDS))


def _render_template(rng: random.Random, tmpl: str) -> str:
    return _ENV_TOKEN_RE.sub(lambda m: _ENVELOPE_FIELDS[m.group(0)][0](rng), tmpl)


def _template_regex(tmpl: str) -> str:
    out, pos = [], 0
    for m in _ENV_TOKEN_RE.finditer(tmpl):
        out += [re.escape(tmpl[pos:m.start()]), _ENVELOPE_FIELDS[m.group(0)][1]]
        pos = m.end()
    return "".join(out) + re.escape(tmpl[pos:])


# ---------------------------------------------------------------------------------------------------------------
# Family schema
# ---------------------------------------------------------------------------------------------------------------
def _targets(fam: dict) -> tuple[list[dict], list[dict]]:
    return ([s for s in fam["slots"] if s["role"] == "latency"], [s for s in fam["slots"] if s["role"] == "user"])


def surface_defer_reason(fam: dict) -> str | None:
    """The DEFER reason implied by the family's own surface, or None if its undamaged payloads carry a struct.
    This is the single place where 'what defers' is decided; validate_family checks `defer_reason` against it."""
    lats, usrs = _targets(fam)
    if fam.get("records", 1) > 1:
        return "multi_record"
    if not lats or not usrs:
        return "target_missing"
    if len(lats) > 1 or len(usrs) > 1:
        return "ambiguous_latency"
    v = lats[0]["value"]
    if v.get("numfmt") == "nonnumeric":
        return "non_numeric"
    if v.get("numfmt") == "decimal_comma":
        return "decimal_comma"
    if v.get("numfmt") == "overflow":
        return "out_of_range"
    if v.get("unit_pos") == "none":
        return "unit_unknown"
    if usrs[0]["value"].get("scheme") == "too_long":
        return "id_too_long"
    dmg = fam.get("damage")
    if dmg and dmg.get("p", 0) >= 1:
        return "truncated" if dmg["kind"] == "truncate" else "corrupted"
    return None


def validate_family(fam: dict) -> list[str]:
    errs: list[str] = []
    if fam.get("container") not in CONTAINERS:
        return [f"container {fam.get('container')!r}"]
    if fam.get("quote", "none") not in QUOTES:
        errs.append(f"quote {fam.get('quote')!r}")
    if fam.get("order", "fixed") not in ("fixed", "shuffle"):
        errs.append(f"order {fam.get('order')!r}")
    slots = fam.get("slots") or []
    if not slots:
        return errs + ["no slots"]
    keyed = fam["container"] not in ("csv_positional", "free_text")
    for i, s in enumerate(slots):
        v = s.get("value") or {}
        if s.get("role") not in ("latency", "user", "distractor") or v.get("gen") not in GENERATORS:
            errs.append(f"slot {i}: role/gen")
            continue
        if keyed and not s.get("key"):
            errs.append(f"slot {i}: key required")
        key = s.get("key", "")
        if s["role"] == "latency":
            if v["gen"] != "duration" or v.get("unit") not in UNITS:
                errs.append(f"slot {i}: latency needs a duration with a unit")
                continue
            pos = v.get("unit_pos", "suffix")
            if pos in ("suffix", "suffix_space") and UNIT_SURFACES.get(v.get("unit_text", v["unit"])) != v["unit"]:
                errs.append(f"slot {i}: unit_text does not denote {v['unit']}")
            if pos == "key" and unit_from_key(key) != v["unit"]:
                errs.append(f"slot {i}: key {key!r} does not carry unit {v['unit']}")
            if pos == "implicit" and IMPLICIT_UNITS.get(key) != v["unit"]:
                errs.append(f"slot {i}: no implicit-unit convention for {key!r}")
            if pos == "none" and (unit_from_key(key) or key in IMPLICIT_UNITS):
                errs.append(f"slot {i}: key {key!r} implies a unit, so the surface does not defer")
            if pos in ("suffix", "suffix_space", "none") and keyed and unit_from_key(key):
                errs.append(f"slot {i}: key {key!r} carries a unit but unit_pos is {pos}")
            if key in IMPLICIT_UNITS and pos != "implicit":
                errs.append(f"slot {i}: {key!r} has an implicit unit; a cut value would still read as complete")
            if pos not in ("suffix", "suffix_space") and not keyed:
                errs.append(f"slot {i}: unkeyed containers need the unit on the value")
            if v.get("numfmt", "int") not in NUMFMTS + ("nonnumeric", "overflow", "decimal_comma"):
                errs.append(f"slot {i}: numfmt")
        elif s["role"] == "user":
            if v["gen"] != "id":
                errs.append(f"slot {i}: user needs an id generator")
        elif fam["container"] == "json_nested" and (s.get("path") or [None])[0] in FOREIGN_SCOPES:
            pass  # another component's sub-object may reuse any leaf key: the path tells the fields apart
        elif keyed and base_key(key) in {base_key(k) for k in K_LAT + K_USR}:
            errs.append(f"slot {i}: distractor key {key!r} is a target alias")
    if fam["container"] == "free_text":
        n = len(slots)
        refs = re.findall(r"\{(\d+)\}", fam.get("template", ""))
        if sorted(map(int, refs)) != list(range(n)):
            errs.append("free_text template must reference every slot exactly once")
    if fam["container"] in ("csv_positional", "free_text") and fam.get("order", "fixed") != "fixed":
        errs.append("unkeyed containers need a fixed order")
    dmg = fam.get("damage")
    if dmg and (dmg.get("kind") not in ("truncate", "corrupt") or not 0 <= dmg.get("p", -1) <= 1):
        errs.append("damage")
    if not errs and surface_defer_reason(fam) != fam.get("defer_reason"):
        errs.append(f"defer_reason {fam.get('defer_reason')!r} but the surface implies {surface_defer_reason(fam)!r}")
    return errs


# ---------------------------------------------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------------------------------------------
def _quote_segs(segs: list[Seg], policy: str) -> tuple[list[Seg], bool]:
    text = "".join(t for t, _ in segs)
    numeric = bool(_JSON_NUM_RE.match(text))
    q = {"none": "", "double_all": '"', "double_strings": "" if numeric else '"', "single_strings": "" if numeric else "'",
         "as_needed": '"' if (" " in text or not text) else ""}[policy]
    return ([(q, None), *segs, (q, None)] if q else segs), bool(q)


def _json_value(slot: dict, segs: list[Seg]) -> list[Seg]:
    text = "".join(t for t, _ in segs)
    if slot["value"].get("bare", slot["value"]["gen"] != "id") and _JSON_NUM_RE.match(text) and not (len(text) > 1 and text[0] == "0" and text[1].isdigit()):
        return segs
    esc = [(t.replace("\\", "\\\\").replace('"', '\\"'), tag) for t, tag in segs]
    return [('"', None), *esc, ('"', None)]


def _render_record(fam: dict, rng: random.Random) -> list[Seg]:
    slots = list(fam["slots"])
    if fam.get("order", "fixed") == "shuffle":
        rng.shuffle(slots)
    vals = []
    for s in slots:
        v = dict(s["value"])
        v["_tag"] = {"latency": "lat", "user": "usr"}.get(s["role"])
        vals.append(GENERATORS[v["gen"]](rng, v))
    c = fam["container"]
    out: list[Seg] = []
    if c in ("kv", "logfmt", "query_string"):
        sep, assign = {"logfmt": (" ", "="), "query_string": ("&", "=")}.get(c, (fam.get("sep", " "), fam.get("assign", "=")))
        for i, (s, segs) in enumerate(zip(slots, vals)):
            text = "".join(t for t, _ in segs)
            if c == "query_string":
                segs = [(t.replace(" ", "+"), tag) for t, tag in segs]
            elif c == "logfmt":  # logfmt quotes exactly the values that need it
                segs = [('"', None), *segs, ('"', None)] if (" " in text or not text) else segs
            else:
                segs, _ = _quote_segs(segs, fam.get("quote", "none"))
            out += ([(sep, None)] if i else []) + [(s["key"] + assign, None)] + segs
    elif c in ("json_flat", "json_nested"):
        colon, comma = ((": ", ", ") if fam.get("json_spaced") else (":", ","))
        tree: dict = {}
        for s, segs in zip(slots, vals):
            node = tree
            for p in (s.get("path") or []) if c == "json_nested" else []:
                node = node.setdefault(p, {})
                if not isinstance(node, dict):
                    raise ValueError("json path collides with a leaf")
            node[s["key"]] = _json_value(s, segs)

        def emit(node: dict) -> list[Seg]:
            o: list[Seg] = [("{", None)]
            for i, (k, val) in enumerate(node.items()):
                o += ([(comma, None)] if i else []) + [(json.dumps(k) + colon, None)]
                o += emit(val) if isinstance(val, dict) else val
            return o + [("}", None)]
        out = emit(tree)
    elif c == "csv_positional":
        sep = fam.get("sep", ",")
        for i, segs in enumerate(vals):
            out += ([(sep, None)] if i else []) + segs
    elif c == "free_text":
        pos = 0
        for m in re.finditer(r"\{(\d+)\}", fam["template"]):
            out += [(fam["template"][pos:m.start()], None)] + vals[int(m.group(1))]
            pos = m.end()
        out.append((fam["template"][pos:], None))
    return out


def _value_extents(segs: list[Seg], offsets: list[int], tag: str) -> tuple[int, int, bool] | None:
    """(start, end, protected) of the whole value token holding `tag`. `protected` means a cut inside the token is
    visible on the surface: the value is quoted, or a unit suffix follows the number."""
    for i, (_, t) in enumerate(segs):
        if t == tag:
            start, end = offsets[i], offsets[i + 1]
            quoted = i > 0 and segs[i - 1][0] in ('"', "'")
            suffixed = tag == "lat" and i + 1 < len(segs) and segs[i + 1][0].strip() in UNIT_SURFACES
            if suffixed:
                end = offsets[i + 2]
            return start, end, quoted or suffixed
    return None


def render(fam: dict, rng: random.Random) -> Row:
    """One payload with its gold label. The family must have passed validate_family."""
    segs: list[Seg] = [(_render_template(rng, fam.get("envelope", "")), None)]
    n_rec = fam.get("records", 1)
    for r in range(n_rec):
        segs += ([(fam.get("record_sep", " | "), None)] if r else []) + _render_record(fam, rng)
    segs.append((_render_template(rng, fam.get("suffix", "")), None))
    enc = [t.encode("utf-8") for t, _ in segs]
    offsets = [0]
    for b in enc:
        offsets.append(offsets[-1] + len(b))
    payload = b"".join(enc)

    reason = surface_defer_reason(fam)
    dmg = fam.get("damage")
    if reason in (None, "truncated", "corrupted") and dmg and rng.random() < dmg["p"]:
        ext = [e for e in (_value_extents(segs, offsets, "lat"), _value_extents(segs, offsets, "usr")) if e]
        last_end = max(e[1] for e in ext)
        if dmg["kind"] == "truncate":
            # Cut before the last target ends, never strictly inside an unprotected value: those cuts look complete.
            cuts = [c for c in range(1, last_end) if not any(a < c < b and not prot for a, b, prot in ext)]
            lat = _value_extents(segs, offsets, "lat")  # "0.812sec" cut to "0.812s" still reads as a latency
            cuts = [c for c in cuts if not (lat[0] < c < lat[1] and _LAT_TOKEN_RE.match(payload[lat[0]:c]))]
            payload = payload[: rng.choice(cuts)]
            reason = "truncated"
        else:
            a, b, _ = rng.choice(ext)
            k = rng.randrange(a, b)
            payload = payload[:k] + bytes([rng.choice((0x00, 0x07, 0x80, 0x9B, 0xC3, 0xFE, 0xFF))]) + payload[k + 1:]
            reason = "corrupted"
    elif reason in ("truncated", "corrupted"):
        reason = None
    if reason is None and len(payload) > MAX_PAYLOAD:
        reason = "too_long"
    if reason is not None:
        return Row(payload=payload, defer=True, reason=reason)

    (li,) = [i for i, (_, t) in enumerate(segs) if t == "lat"]
    (ui,) = [i for i, (_, t) in enumerate(segs) if t == "usr"]
    unit = _targets(fam)[0][0]["value"]["unit"]
    us = to_micros(enc[li], unit)
    if us is None:
        return Row(payload=payload, defer=True, reason="out_of_range")
    return Row(payload=payload, defer=False, latency_us=us, user_id=enc[ui].decode("ascii"), unit=unit,
               lat_span=(offsets[li], offsets[li + 1]), usr_span=(offsets[ui], offsets[ui + 1]))


# ---------------------------------------------------------------------------------------------------------------
# Inverse renderer: parse a payload with the family's own format knowledge
# ---------------------------------------------------------------------------------------------------------------
def _scan_pairs(body: bytes, sep: bytes, assign: bytes) -> list[tuple[bytes, bytes]] | None:
    pairs, pos, n = [], 0, len(body)
    while pos < n:
        i = body.find(assign, pos)
        if i < 0:
            return None
        key, v0 = body[pos:i], i + len(assign)
        if v0 < n and body[v0:v0 + 1] in (b'"', b"'"):
            j = body.find(body[v0:v0 + 1], v0 + 1)
            if j < 0:
                return None  # unterminated quote
            value, pos = body[v0 + 1:j], j + 1
            if pos < n:
                if not body.startswith(sep, pos):
                    return None
                pos += len(sep)
        else:
            j = body.find(sep, v0)
            j = n if j < 0 else j
            value, pos = body[v0:j], j + len(sep)
        pairs.append((key, value))
    return pairs


def _parse_latency(value: bytes, slot: dict) -> int | None:
    v = slot["value"]
    pos = v.get("unit_pos", "suffix")
    if pos in ("suffix", "suffix_space"):
        m = re.match(rb"(.*?)( ?)(ns|us|\xc2\xb5s|ms|sec|s|min)\Z", value, re.S)
        if not m or (m.group(2) == b" ") != (pos == "suffix_space"):
            return None
        return to_micros(m.group(1), UNIT_SURFACES[m.group(3).decode("utf-8")])
    if pos == "none":
        return None
    return to_micros(value, v["unit"])


def inverse(fam: dict, payload: bytes) -> tuple[int, str] | None:
    """(latency_us, user_id), or None for DEFER. Strict: anything the family's format does not explain defers."""
    if len(payload) > MAX_PAYLOAD or fam.get("records", 1) > 1:
        return None
    lats, usrs = _targets(fam)
    if len(lats) != 1 or len(usrs) != 1:
        return None
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    m = re.match(_template_regex(fam.get("envelope", "")), text)
    if not m:
        return None
    text = text[m.end():]
    if fam.get("suffix"):
        m = re.search("(?:" + _template_regex(fam["suffix"]) + r")\Z", text)
        if not m:
            return None
        text = text[: m.start()]
    body = text.encode("utf-8")
    c = fam["container"]
    found: dict[str, list[bytes]] = {"latency": [], "user": []}
    if c in ("kv", "logfmt", "query_string"):
        sep, assign = {"logfmt": (" ", "="), "query_string": ("&", "=")}.get(c, (fam.get("sep", " "), fam.get("assign", "=")))
        pairs = _scan_pairs(body, sep.encode(), assign.encode())
        if pairs is None:
            return None
        for s in lats + usrs:
            found[s["role"]] += [v for k, v in pairs if k == s["key"].encode()]
    elif c in ("json_flat", "json_nested"):
        try:
            obj = json.loads(body, parse_float=str, parse_int=str, parse_constant=str)
        except ValueError:
            return None
        for s in lats + usrs:
            node = obj
            for p in ((s.get("path") or []) if c == "json_nested" else []) + [s["key"]]:
                node = node.get(p) if isinstance(node, dict) else None
            if isinstance(node, str):
                found[s["role"]].append(node.encode("utf-8"))
    elif c == "csv_positional":
        cols = body.split(fam.get("sep", ",").encode())
        if len(cols) != len(fam["slots"]):
            return None
        for i, s in enumerate(fam["slots"]):
            if s["role"] in found:
                found[s["role"]].append(cols[i])
    elif c == "free_text":
        rx = re.sub(r"\\\{(\d+)\\\}", lambda g: f"(?P<s{g.group(1)}>.+?)", re.escape(fam["template"]))
        mm = re.fullmatch(rx.encode(), body, re.S)
        if not mm:
            return None
        for i, s in enumerate(fam["slots"]):
            if s["role"] in found:
                found[s["role"]].append(mm.group(f"s{i}"))
    if len(found["latency"]) != 1 or len(found["user"]) != 1:
        return None
    us = _parse_latency(found["latency"][0], lats[0])
    uid = found["user"][0]
    if us is None or not _ID_RE.match(uid) or len(uid) > USER_ID_MAX:
        return None
    return us, uid.decode("ascii")


def check_family(fam: dict, n: int = 50, seed: int = 0) -> list[str]:
    """Schema errors, or (if the schema is fine) the first disagreement between renderer and inverse renderer."""
    errs = validate_family(fam)
    if errs:
        return errs
    rng = random.Random(seed)
    for _ in range(n):
        try:
            row = render(fam, rng)
        except (ValueError, KeyError, IndexError) as e:
            return [f"render: {type(e).__name__}: {e}"]
        if row.reason == "too_long":
            return [f"payload exceeds {MAX_PAYLOAD} bytes"]
        got = inverse(fam, row.payload)
        want = None if row.defer else (row.latency_us, row.user_id)
        if got != want:
            return [f"inverse {got!r} != gold {want!r} on {row.payload!r}"]
        if not row.defer:
            a, b = row.lat_span
            c, d = row.usr_span
            if to_micros(row.payload[a:b], row.unit) != row.latency_us or row.payload[c:d].decode("ascii") != row.user_id:
                return [f"span does not reproduce the struct on {row.payload!r}"]
    return []


def family_signature(fam: dict) -> tuple:
    """Two families with the same signature are near-duplicates (curriculum validation rejects the second)."""
    lat = next((s["value"] for s in fam["slots"] if s["role"] == "latency"), {})
    return (fam["container"], fam.get("sep"), fam.get("assign"), fam.get("quote", "none"),
            re.sub(r"[^<>\w]+", " ", fam.get("envelope", "")).strip(), fam.get("template"),
            tuple(sorted((s.get("key", ""), s["role"]) for s in fam["slots"])),
            lat.get("unit"), lat.get("unit_pos", "suffix"), lat.get("numfmt", "int"), fam.get("defer_reason"))
