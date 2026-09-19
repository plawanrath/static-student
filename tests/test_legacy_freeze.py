import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_legacy_parser_is_unchanged_since_the_freeze():
    """The drift results are only meaningful if the legacy parser predates the drift operators and never chases them."""
    entries = [l.split(": ") for l in (REPO / "csrc/legacy/FREEZE").read_text().splitlines() if l and not l.startswith("#")]
    hashed = {k: v for k, v in entries if k != "frozen_utc"}
    assert len(hashed) == 3
    for rel, digest in hashed.items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == digest, f"{rel} changed after the freeze"
