"""Build a curriculum from a spec file (stages S1-S4; the S5 audit needs the real teacher and lives in audit.py).

    python -m static_student.curriculum.build specs/telemetry.spec --families 200 --seed 0 \
        --rows 50000 --out results/w01_curriculum_smoke --reject-tokens <file>

Writes data/curriculum/<run>/{manifest.json,families.jsonl,task_card.json} (tracked, small), the rendered pools under
data/processed/<run>/ (regenerated, never tracked), and a bucket / acceptance report under --out.

`--reject-tokens` is a file with one surface token per line. A family that uses any of them is rejected in S3. The
held-out drift lists reach validation only through this file; this package never imports them and no prompt ever
contains them.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import time
from pathlib import Path

from static_student.curriculum.spec import parse_spec
from static_student.curriculum.teacher import BUCKETS, StubTeacher
from static_student.drift import operators
from static_student.tasks import telemetry as T

REPO = Path(__file__).resolve().parents[2]


def family_tokens(fam: dict) -> set[str]:
    """Lower-cased word pieces of every key, template and envelope literal of a family."""
    text = " ".join([fam.get("envelope", ""), fam.get("suffix", ""), fam.get("template", "")]
                    + [s.get("key", "") for s in fam["slots"]] + [p for s in fam["slots"] for p in s.get("path", [])])
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", text) if w}


def validate(fam: dict, bucket: str, seen: set, reject: set[str], seed: int) -> str | None:
    """S3. Returns the rejection reason, or None if the family is accepted."""
    errs = T.check_family(fam, n=50, seed=seed)
    if errs:
        return "schema" if "inverse" not in errs[0] and "exceeds" not in errs[0] and "span" not in errs[0] else \
               "too_long" if "exceeds" in errs[0] else "inverse_mismatch"
    if (bucket == "should_defer") != (fam["defer_reason"] is not None):
        return "bucket_mismatch"
    if family_tokens(fam) & reject:
        return "held_out_token"
    if T.family_signature(fam) in seen:
        return "duplicate_signature"
    return None


def propose_families(teacher, card: dict, n: int, seed: int, reject: set[str], max_attempts: int = 50, on_accept=None) -> tuple[list[dict], dict]:
    rng = random.Random(f"families:{seed}")
    quota = {b: round(n * share) for b, share in BUCKETS.items()}
    quota["clean"] += n - sum(quota.values())
    accepted, seen = [], set()
    stats = {b: collections.Counter() for b in BUCKETS}
    for bucket in ("clean", "near_miss", "should_defer", "drifted"):  # drifted rewrites families accepted before it
        bases = [f for f in accepted if f["defer_reason"] is None]
        got = attempts = 0
        while got < quota[bucket] and attempts < max_attempts * max(1, quota[bucket]):
            attempts += 1
            if bucket == "drifted":
                fam = rng.choice(bases)
                for _ in range(rng.randrange(1, 4)):
                    fam = operators.apply(rng.choice(list(operators.OPERATORS)), fam, rng, rng.random()) or fam
                fam = {**fam, "bucket": "drifted"}
                if not fam.get("trace"):
                    stats[bucket]["operator_not_applicable"] += 1
                    continue
            else:
                fam = teacher.propose(card, bucket, accepted[-3:], rng)
                if fam is None:
                    stats[bucket]["invalid_json"] += 1
                    continue
            try:
                why = validate(fam, bucket, seen, reject, seed=rng.randrange(2**31))
            except Exception as e:  # noqa: BLE001  a teacher can write anything; a family that crashes validation is rejected
                why = f"crash:{type(e).__name__}"
            stats[bucket][why or "accepted"] += 1
            if why is None:
                fam["family_id"] = f"f{len(accepted):05d}"
                seen.add(T.family_signature(fam))
                accepted.append(fam)
                got += 1
                if on_accept:
                    on_accept(fam, stats)
    return accepted, {b: dict(c) for b, c in stats.items()}


def render_pools(families: list[dict], rows: int, seed: int, dev_share: float = 0.1) -> dict[str, list[dict]]:
    """S4. Pools are split by family: a family's payloads land in exactly one pool."""
    rng = random.Random(f"split:{seed}")
    by_bucket = collections.defaultdict(list)
    for f in families:
        by_bucket[f["bucket"]].append(f)
    split = {"tel-train": [], "tel-dev": []}
    for fams in by_bucket.values():  # stratified, so both pools carry every bucket
        rng.shuffle(fams)
        k = max(1, round(len(fams) * dev_share)) if len(fams) > 1 else 0
        split["tel-dev"] += fams[:k]
        split["tel-train"] += fams[k:]
    pools = {}
    for name, fams in split.items():
        n = rows if name == "tel-train" else max(1, round(rows * dev_share))
        r = random.Random(f"render:{seed}:{name}")
        out = []
        for i in range(n):
            fam = fams[i % len(fams)]
            out.append({**T.render(fam, r).as_dict(), "family_id": fam["family_id"], "bucket": fam["bucket"], "trace": fam.get("trace", [])})
        pools[name] = out
    return pools


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--families", type=int, default=200)
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", default="stub", choices=["stub", "mlx"])
    ap.add_argument("--run", default=None, help="run name (default: <task>-<teacher>-s<seed>)")
    ap.add_argument("--out", default=None, help="report directory")
    ap.add_argument("--reject-tokens", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    spec = parse_spec(args.spec)
    if args.teacher == "mlx":
        from static_student.curriculum.mlx_teacher import MLXTeacher  # lazy: Apple silicon only, loads 25 GB
        teacher = MLXTeacher(seed=args.seed)
    else:
        teacher = StubTeacher()
    run = args.run or f"{Path(args.spec).stem}-{args.teacher}-s{args.seed}"
    reject = set(Path(args.reject_tokens).read_text().split()) if args.reject_tokens else set()
    card = teacher.task_card(spec)
    removed = []  # S1: held-out names a teacher comes up with on its own are taken out of its view before any family is asked for
    for group in [f["synonyms"] for f in card["fields"]] + list(card["near_miss"].values()):
        removed += [k for k in group if family_tokens({"slots": [{"key": k}]}) & reject]
        group[:] = [k for k in group if not family_tokens({"slots": [{"key": k}]}) & reject]
    card["removed_held_out"] = len(removed)  # a count only: the names themselves stay out of every tracked file
    card.pop("teacher_raw", None)
    cdir, pdir = REPO / "data/curriculum" / run, REPO / "data/processed" / run
    cdir.mkdir(parents=True, exist_ok=True), pdir.mkdir(parents=True, exist_ok=True)
    (cdir / "task_card.json").write_text(json.dumps(card, indent=1) + "\n")
    progress = cdir / "families.partial.jsonl"  # a long teacher run leaves its accepted families here as it goes
    progress.write_text("")

    def on_accept(fam, stats):
        with progress.open("a") as fh:
            fh.write(json.dumps(fam, sort_keys=True, ensure_ascii=False) + "\n")
        n = sum(c.get("accepted", 0) for c in stats.values())
        if n % 10 == 0:
            print(f"[curriculum] {n} accepted, {sum(sum(c.values()) for c in stats.values())} proposed, {time.time() - t0:.0f}s", flush=True)

    families, stats = propose_families(teacher, card, args.families, args.seed, reject, max_attempts=4 if args.teacher == "mlx" else 50, on_accept=on_accept)
    progress.unlink()
    if getattr(teacher, "transcript", None):
        (pdir / "teacher_transcript.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in teacher.transcript))
    pools = render_pools(families, args.rows, args.seed)
    (cdir / "task_card.json").write_text(json.dumps(card, indent=1) + "\n")
    fam_text = "".join(json.dumps(f, sort_keys=True, ensure_ascii=False) + "\n" for f in families)
    (cdir / "families.jsonl").write_text(fam_text)
    for name, rows in pools.items():
        (pdir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    proposed = sum(sum(c.values()) for c in stats.values())
    manifest = {
        "run": run, "spec": args.spec, "spec_sha256": spec["sha256"], "teacher": teacher.name, "sampler": getattr(teacher, "settings", None), "seed": args.seed,
        "task_card_sha256": hashlib.sha256(json.dumps(card, sort_keys=True).encode()).hexdigest(),
        "families_sha256": hashlib.sha256(fam_text.encode()).hexdigest(),
        "reject_tokens_sha256": hashlib.sha256("\n".join(sorted(reject)).encode()).hexdigest() if reject else None,
        "families": len(families), "families_per_bucket": dict(collections.Counter(f["bucket"] for f in families)),
        "proposed": proposed, "acceptance_rate": round(len(families) / max(1, proposed), 4), "validation": stats,
        "pools": {name: {"n": len(rows), "families": len({r["family_id"] for r in rows}),
                         "defer_rate": round(sum(r["defer"] for r in rows) / len(rows), 4), "seed": args.seed} for name, rows in pools.items()},
    }
    (cdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if args.out:
        out = REPO / args.out
        out.mkdir(parents=True, exist_ok=True)
        rows = pools["tel-train"]
        report = {**manifest, "seconds": round(time.time() - t0, 1),
                  "defer_reasons_train": dict(collections.Counter(r["reason"] for r in rows if r["defer"])),
                  "containers": dict(collections.Counter(f["container"] for f in families)),
                  "units": dict(collections.Counter(r["unit"] for r in rows if not r["defer"])),
                  "payload_bytes": {"mean": round(sum(len(r["payload"]) for r in rows) / len(rows), 1), "max": max(len(r["payload"]) for r in rows)},
                  "train_dev_family_overlap": len({r["family_id"] for r in pools["tel-train"]} & {r["family_id"] for r in pools["tel-dev"]})}
        (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
        rs = random.Random(0)
        (out / "samples.txt").write_text("".join(
            f"{r['bucket']:<12} {('DEFER:' + r['reason']) if r['defer'] else (str(r['latency_us']) + 'us ' + r['user_id']):<48} {r['payload']!r}\n"
            for r in rs.sample(rows, 60)))
    print(json.dumps({k: manifest[k] for k in ("run", "families", "families_per_bucket", "proposed", "acceptance_rate", "pools")}, indent=1))
    return manifest


if __name__ == "__main__":
    main()
