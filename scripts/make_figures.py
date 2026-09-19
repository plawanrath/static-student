"""Regenerate every paper figure from results/*.json. One function per figure; no number is typed by hand.

Usage: python scripts/make_figures.py [--out-dir results/figures]
Writes <name>.pdf + <name>.png per figure and manifest.json (figure -> source artifacts).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIGURES = {}  # name -> (function(out_dir) -> list of source artifact paths)


def figure(fn):
    FIGURES[fn.__name__] = fn
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "results" / "figures"))
    out = Path(ap.parse_args().out_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {name: [str(s) for s in fn(out)] for name, fn in FIGURES.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[figures] {len(manifest)} figure(s) -> {out}")


if __name__ == "__main__":
    main()
