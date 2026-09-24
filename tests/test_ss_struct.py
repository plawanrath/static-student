"""The C struct conversion must agree with the Python task definition on every input, including the adversarial ones."""
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from static_student.student.model import UNIT_CLASSES
from static_student.tasks import telemetry as T

REPO = Path(__file__).resolve().parent.parent
EXE = REPO / "build" / "ss_struct_cli"
pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="no C toolchain")


@pytest.fixture(scope="module")
def run_c():
    subprocess.run(["make", "-C", str(REPO / "csrc")], check=True, capture_output=True)

    def go(cases):
        payload = "".join(f"{UNIT_CLASSES.index(u)}\t{s}\n" for s, u in cases).encode("latin-1")
        out = subprocess.run([str(EXE)], input=payload, capture_output=True, check=True).stdout.decode().splitlines()
        assert len(out) == len(cases)
        return [None if x == "X" else int(x) for x in out]
    return go


def _cases():
    rng = random.Random(0)
    out = []
    for s in ("812", "0.812", "8.12e2", "1,204.5", "1999", "0.1", "1.5", "1,000", "999,999,999", "1e2", "1E+2", "1e-2",
              "0", "00", "007", "0.0", ".5", "5.", "1.", "-5", "+5", "", " ", "1 ", " 1", "NaN", "inf", "1e", "1e+",
              "1,00", "1,0000", "12,345,678", "0,812", "1,204,5", "1.2.3", "1e999", "1e99", "9" * 40, "9" * 41,
              "3e81234567", "1e+99", "0.000000000000001", "18446744073709551615", "18446744073709551616",
              "1,234e2", "1.5e1", "123456789012345678901234567890", "1e-99", "0.1e1"):
        for u in UNIT_CLASSES:
            out.append((s, u))
    for _ in range(12000):                       # random well-formed and malformed tokens
        style = rng.randrange(7)
        if style == 0:
            s = str(rng.randrange(0, 10 ** rng.randrange(1, 12)))
        elif style == 1:
            s = f"{rng.random() * 10 ** rng.randrange(0, 6):.{rng.randrange(0, 6)}f}"
        elif style == 2:
            s = f"{rng.random() * 10:.{rng.randrange(1, 4)}e}"
        elif style == 3:
            s = f"{rng.randrange(1, 10 ** 9):,}"
        elif style == 4:
            s = "".join(rng.choice("0123456789.,eE+- abcdef") for _ in range(rng.randrange(1, 12)))
        elif style == 5:
            s = f"{rng.getrandbits(4 * rng.randrange(4, 20)):x}"
        else:                                    # long digit strings and extreme exponents, where exact arithmetic bites
            k = rng.randrange(1, 40)
            s = "".join(rng.choice("0123456789") for _ in range(k))
            if rng.random() < 0.5:
                cut = rng.randrange(1, max(2, k))
                s = s[:cut] + "." + s[cut:]
            if rng.random() < 0.4:
                s += rng.choice(("e", "E")) + rng.choice(("", "+", "-")) + str(rng.randrange(0, 100))
        out.append((s, rng.choice(UNIT_CLASSES)))
    return out


def test_c_matches_python_on_every_token(run_c):
    cases = _cases()
    got = run_c(cases)
    want = [T.to_micros(s.encode("latin-1"), u) for s, u in cases]
    bad = [(s, u, g, w) for (s, u), g, w in zip(cases, got, want) if g != w]
    assert not bad, f"{len(bad)} of {len(cases)} differ, first: {bad[:4]}"


def test_c_matches_python_on_rendered_payload_spans(run_c):
    """The spans the renderer emits are what the kernel points at, so they must convert identically."""
    fams = [json_loads(l) for l in (REPO / "data/curriculum/telemetry-stub-s2-12k/families.jsonl").read_text().splitlines()[:400]]
    rng = random.Random(1)
    cases, want = [], []
    for f in fams:
        for _ in range(5):
            row = T.render(f, rng)
            if not row.defer:
                cases.append((row.payload[slice(*row.lat_span)].decode("latin-1"), row.unit))
                want.append(row.latency_us)
    assert len(cases) > 500
    assert run_c(cases) == want


def json_loads(s):
    import json
    return json.loads(s)
