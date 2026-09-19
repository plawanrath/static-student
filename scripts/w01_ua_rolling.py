"""Feasibility of rolling freeze dates for the User-Agent task (legacy parser alone).

    .venv/bin/python scripts/w01_ua_rolling.py        ->  results/w01_ua_rolling/

The regex list is frozen at every 1 January from 2015 on. A fixture first seen in year Y is scored against the list
frozen k years before 1 January of Y+1... i.e. staleness k = 0 means the newest list that predates the fixture's year.
Results are pooled by staleness, so short, realistic horizons use every fixture added since 2015. Also reported, per
freeze date: size of the pre-D pool, vocabulary size, and the share of later fixtures whose family is new.
"""
from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

import numpy as np

from static_student import stats

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/w01_ua_rolling"
spec = importlib.util.spec_from_file_location("ua_timeline", REPO / "scripts/w01_ua_timeline.py")
tl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tl)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seen = tl.first_seen()
    by: dict[str, dict] = {}
    for c in tl.fixtures(tl.UA_FILES):
        d, s = seen[c["_raw"]], c["user_agent_string"]
        if s not in by or d < by[s]["date"]:
            by[s] = {"ua": s, "date": d, "family": c["family"]}
    rows = list(by.values())
    years = list(range(2015, 2027))
    parsers, per_D = {}, {}
    for y in years:
        D = f"{y}-01-01"
        rev = tl.git("rev-list", "-1", f"--before={D}T00:00:00", tl.PINNED, "--", "regexes.yaml").strip()
        parsers[y] = tl.Frozen(rev)
        pre = [r for r in rows if r["date"] < D]
        post = [r for r in rows if r["date"] >= D]
        fam = collections.Counter(r["family"] for r in pre)
        per_D[D] = {"regex_commit": rev[:8], "patterns": len(parsers[y].rules), "pre": len(pre), "post": len(post), "families_pre": len(fam),
                    "families_pre_with_5plus": sum(v >= 5 for v in fam.values()),
                    "post_new_family_share": round(sum(r["family"] not in fam for r in post) / len(post), 4) if post else None}
    pooled = collections.defaultdict(lambda: {"no_match": [], "wrong": [], "new_family": []})
    vocab = {y: {r["family"] for r in rows if r["date"] < f"{y}-01-01"} for y in years}
    for r in rows:
        fy = int(r["date"][:4])
        if fy < 2015:
            continue
        for k in range(0, fy - 2015 + 1):
            y = fy - k  # list frozen on 1 January of year y; the fixture arrived during year fy
            got = parsers[y].parse(r["ua"])[0]
            p = pooled[k]
            p["no_match"].append(int(got == "Other" and r["family"] != "Other"))
            p["wrong"].append(int(got != "Other" and got != r["family"]))
            p["new_family"].append(int(r["family"] not in vocab[y]))
    by_k = {}
    for k in sorted(pooled):
        p = {m: np.array(v) for m, v in pooled[k].items()}
        if len(p["wrong"]) < 100:
            continue
        by_k[k] = {"n": len(p["wrong"]), "no_match": stats.bootstrap_ci(p["no_match"]).as_dict(), "silent_wrong_family": stats.bootstrap_ci(p["wrong"]).as_dict(),
                   "new_family_share": round(float(p["new_family"].mean()), 4),
                   "silent_wrong_family_among_known_families": round(float(p["wrong"][p["new_family"] == 0].mean()), 4) if (p["new_family"] == 0).any() else None,
                   "n_known_families": int((p["new_family"] == 0).sum())}
    summary = {"uap_core_commit": tl.PINNED, "staleness_years": by_k, "per_freeze_date": per_D,
               "note": "Staleness k: list frozen on 1 January of (fixture year - k). Buckets with n < 100 are dropped. Python re preview, family only."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for k, v in by_k.items():
        print(f"staleness {k}-{k + 1}y  n={v['n']:5d}  no-match {v['no_match']['point']:.3f}  wrong-family {v['silent_wrong_family']['point']:.3f}  new-family {v['new_family_share']:.3f}  "
              f"wrong|known {v['silent_wrong_family_among_known_families']} (n={v['n_known_families']})")
    for D, v in per_D.items():
        print(D, v)


if __name__ == "__main__":
    main()
