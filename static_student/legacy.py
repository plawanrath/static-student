"""Python side of the legacy parsers: build the native binary and run payloads through it.

The legacy parser is C++ (csrc/legacy/); here it is a black box with the shared parser signature. `parse_many`
returns, per payload, None for MS_DEFER or (latency_us, user_id) for MS_OK.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BINARIES = {"telemetry": REPO / "build" / "legacy_telemetry_cli"}


def build(quiet: bool = True) -> None:
    subprocess.run(["make", "-C", str(REPO / "csrc")], check=True, capture_output=quiet)


def parse_many(payloads: list[bytes], task: str = "telemetry") -> list[tuple[int, str] | None]:
    exe = BINARIES[task]
    if not exe.exists():
        build()
    if any(b"\n" in p for p in payloads):
        raise ValueError("payloads are single lines")
    res = subprocess.run([str(exe)], input=b"".join(p + b"\n" for p in payloads), capture_output=True, check=True)
    lines = res.stdout.split(b"\n")[:-1]
    if len(lines) != len(payloads):
        raise RuntimeError(f"{len(lines)} results for {len(payloads)} payloads")
    out: list[tuple[int, str] | None] = []
    for ln in lines:
        if ln == b"D":
            out.append(None)
        else:
            _, us, uid = ln.split(b"\t", 2)
            out.append((int(us), uid.decode("latin-1")))
    return out


def outcomes(rows, results) -> dict[str, list[int]]:
    """Per-payload 0/1 indicators for a parser's results against gold rows.
    matched = returned a struct; wrong = matched and (gold defers or struct differs); correct = end state right."""
    matched = [int(r is not None) for r in results]
    wrong = [int(r is not None and (g.defer or r != (g.latency_us, g.user_id))) for g, r in zip(rows, results)]
    correct = [int((r is None and g.defer) or (r is not None and not g.defer and r == (g.latency_us, g.user_id)))
               for g, r in zip(rows, results)]
    return {"matched": matched, "wrong": wrong, "correct": correct}
