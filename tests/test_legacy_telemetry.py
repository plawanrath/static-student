import json
import random
import shutil
from pathlib import Path

import pytest

from static_student import legacy
from static_student.tasks import telemetry as T

SOURCES = json.loads((Path(__file__).resolve().parent.parent / "data/drift/sources_v0.json").read_text())["sources"]
pytestmark = pytest.mark.skipif(shutil.which("pkg-config") is None or shutil.which("make") is None, reason="native toolchain")


@pytest.fixture(scope="module", autouse=True)
def _built():
    try:
        legacy.build()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"legacy parser does not build here: {e}")


@pytest.mark.parametrize("src", SOURCES, ids=[s["id"] for s in SOURCES])
def test_exact_on_every_step0_source(src):
    rng = random.Random(0)
    rows = [T.render(src, rng) for _ in range(2000)]
    got = legacy.parse_many([r.payload for r in rows])
    assert got == [(r.latency_us, r.user_id) for r in rows]


def test_defers_on_garbage_and_overlong():
    got = legacy.parse_many([b"", b"hello world", b"duration=12ms", b"x" * 300,
                             b"duration=12ms user_id=" + b"a" * 37, b"duration=99999999999999999999999999s user_id=u-1"])
    assert got == [None] * 6


def test_struct_conversion_truncates():
    got = legacy.parse_many([b"duration=1.9999999s user_id=u-1", b"duration=0.5us user_id=u-1", b"duration=812ms user_id=u-1"])
    assert got == [(1_999_999, "u-1"), (0, "u-1"), (812_000, "u-1")]
