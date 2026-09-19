"""The spec file: three lines, the only input the build pipeline reads."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def parse_spec(path: str | Path) -> dict:
    text = Path(path).read_text()
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    fields = dict(l.split(":", 1) for l in lines)
    if set(fields) != {"Semantic Target", "Fallback", "Contract"}:
        raise ValueError(f"{path}: expected exactly Semantic Target / Fallback / Contract, got {sorted(fields)}")
    c = fields["Contract"]
    alpha = re.search(r"<=\s*([\d.]+)\s*%\s*wrong", c)
    conf = re.search(r"confidence\s*([\d.]+)\s*%", c)
    cov = re.search(r"min coverage\s*([\d.]+)\s*%", c)
    if not (alpha and conf and cov):
        raise ValueError(f"{path}: contract must state '<=A% wrong', 'confidence C%' and 'min coverage M%'")
    return {"target": fields["Semantic Target"].strip(), "fallback": fields["Fallback"].strip(),
            "alpha": float(alpha.group(1)) / 100, "delta": round(1 - float(conf.group(1)) / 100, 6),
            "min_coverage": float(cov.group(1)) / 100, "sha256": hashlib.sha256(text.encode()).hexdigest()}
