"""Multiply a teacher-written curriculum with in-curriculum drift rewrites.

    .venv/bin/python scripts/w02_expand_curriculum.py --src telemetry-mlx-s0 --families 12000 --seed 0     ->  data/curriculum/<src>-x<k>/

A student that sees about a thousand families memorizes them (52.7% exact on unseen families); with about ten thousand it reads the format
(95.8%). One teacher night buys about a thousand families, so every accepted family is rewritten by 1-3 operators from the in-curriculum table
until the target count is reached. Every rewrite passes the same validation as a teacher proposal (inverse renderer, length, held-out tokens,
signature de-duplication). The teacher-written families are all kept.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from static_student.curriculum import build
from static_student.drift import operators
from static_student.tasks import telemetry as T

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--families", type=int, default=12_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reject-tokens", default="data/cache/heldout_tokens.txt")
    args = ap.parse_args()
    src = REPO / "data/curriculum" / args.src
    base = [json.loads(l) for l in (src / "families.jsonl").read_text().splitlines() if l.strip()]
    reject = set((REPO / args.reject_tokens).read_text().split())
    rng = random.Random(f"expand:{args.seed}")
    out, seen, stats = list(base), {T.family_signature(f) for f in base}, collections.Counter()
    parents = [f for f in base if f["bucket"] != "drifted"]
    while len(out) < args.families and stats["attempts"] < 20 * args.families:
        stats["attempts"] += 1
        parent = rng.choice(parents)
        fam = parent
        for _ in range(rng.randrange(1, 4)):
            fam = operators.apply(rng.choice(list(operators.OPERATORS)), fam, rng, rng.random()) or fam
        if fam is parent:
            stats["operator_not_applicable"] += 1
            continue
        bucket = "should_defer" if fam["defer_reason"] is not None else parent["bucket"] if parent["bucket"] != "should_defer" else "drifted"
        fam = {**fam, "bucket": bucket, "parent": parent["family_id"]}
        why = build.validate(fam, bucket, seen, reject, seed=rng.randrange(2**31))
        stats[why or "accepted"] += 1
        if why is None:
            fam["family_id"] = f"x{len(out):05d}"
            seen.add(T.family_signature(fam))
            out.append(fam)
    run = f"{args.src}-x{args.families // 1000}k"
    dst = REPO / "data/curriculum" / run
    dst.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(f, sort_keys=True, ensure_ascii=False) + "\n" for f in out)
    (dst / "families.jsonl").write_text(text)
    manifest = {"run": run, "source": args.src, "source_families_sha256": hashlib.sha256((src / "families.jsonl").read_bytes()).hexdigest(), "seed": args.seed,
                "families": len(out), "teacher_or_source_families": len(base), "rewrites": len(out) - len(base), "validation": dict(stats),
                "families_per_bucket": dict(collections.Counter(f["bucket"] for f in out)), "families_sha256": hashlib.sha256(text.encode()).hexdigest()}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
