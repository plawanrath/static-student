"""W1 User-Agent fixture timeline: first-seen date per uap-core fixture, choice of the freeze date D, and a first
look at what the regex list frozen at D does on fixtures that arrived later.

    git clone https://github.com/ua-parser/uap-core.git data/raw/uap-core     # full history is required
    .venv/bin/python scripts/w01_ua_timeline.py                               ->  results/w01_ua_timeline/

First-seen date = earliest author date of a commit that adds a line carrying the fixture's user_agent_string, found in
one pass over `git log -p`. Fixtures are de-duplicated by exact string, keeping the earliest date. D is the latest
1 January such that at least 30% of fixtures are post-D and the post-D fixtures span at least three years.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml

from static_student import stats

REPO = Path(__file__).resolve().parent.parent
UAP = REPO / "data/raw/uap-core"
OUT = REPO / "results/w01_ua_timeline"
PINNED = "73e7340c3ed8055051607b296bf46ead7aa5f19e"
UA_FILES = ["tests/test_ua.yaml", "test_resources/firefox_user_agent_strings.yaml", "test_resources/pgts_browser_list.yaml",
            "test_resources/opera_mini_user_agent_strings.yaml", "test_resources/podcasting_user_agent_strings.yaml"]
OS_FILES = ["tests/test_os.yaml", "test_resources/additional_os_tests.yaml"]
_KEY = re.compile(r"^\s*-?\s*user_agent_string:\s*(.*\S)\s*$")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(UAP), *args], check=True, capture_output=True).stdout.decode("utf-8", "replace")


def first_seen() -> dict[str, str]:
    seen: dict[str, str] = {}
    date = None
    proc = subprocess.Popen(["git", "-C", str(UAP), "log", "--format=@@@%aI", "-p", "--no-color", "--no-renames", PINNED, "--", "*.yaml"],
                            stdout=subprocess.PIPE)
    for raw in proc.stdout:
        line = raw.decode("utf-8", "replace")
        if line.startswith("@@@"):
            date = line[3:13]
        elif line.startswith("+") and not line.startswith("+++"):
            m = _KEY.match(line[1:])
            if m and (m.group(1) not in seen or date < seen[m.group(1)]):
                seen[m.group(1)] = date
    proc.wait()
    return seen


def fixtures(files: list[str], rev: str = PINNED) -> list[dict]:
    out = []
    for f in files:
        text = git("show", f"{rev}:{f}")
        raw = [m.group(1) for m in map(_KEY.match, text.splitlines()) if m]
        cases = yaml.load(text, Loader=yaml.CSafeLoader)["test_cases"]
        assert len(raw) == len(cases), f
        out += [{**c, "_raw": r, "_file": f} for r, c in zip(raw, cases)]
    return out


class Frozen:
    """uap-core's documented matching algorithm over `user_agent_parsers`: first regex that matches wins."""

    def __init__(self, rev: str):
        self.rules, self.uncompilable = [], 0
        for r in yaml.load(git("show", f"{rev}:regexes.yaml"), Loader=yaml.CSafeLoader)["user_agent_parsers"]:
            try:
                self.rules.append((re.compile(r["regex"], re.I if r.get("regex_flag") == "i" else 0), r))
            except re.error:
                self.uncompilable += 1

    def parse(self, ua: str) -> tuple[str, str | None, str | None]:
        for rx, r in self.rules:
            m = rx.search(ua)
            if not m:
                continue
            g = lambda i: m.group(i) if m.re.groups >= i else None  # noqa: E731
            fam = r.get("family_replacement")
            fam = re.sub(r"\$(\d)", lambda k: g(int(k.group(1))) or "", fam).strip() if fam else g(1)
            return fam, r.get("v1_replacement") or g(2), r.get("v2_replacement") or g(3)
        return "Other", None, None


def evaluate(rows: list[dict], D: str) -> dict:
    """Split at D, freeze the regex list at the last commit before D, and score it on both sides."""
    rev = git("rev-list", "-1", f"--before={D}T00:00:00", PINNED, "--", "regexes.yaml").strip()
    pre, post = [r for r in rows if r["date"] < D], [r for r in rows if r["date"] >= D]
    fam = collections.Counter(r["family"] for r in pre)
    cum, K = 0, 0
    for _, k in fam.most_common():
        cum, K = cum + k, K + 1
        if cum / len(pre) >= 0.99:
            break
    vocab = {f for f, _ in fam.most_common(K)}
    frozen, head = Frozen(rev), Frozen(PINNED)
    out = {"D": D, "pre": len(pre), "post": len(post), "post_share": round(len(post) / len(rows), 4),
           "legacy_regexes_commit": rev, "legacy_regexes_commit_date": git("log", "-1", "--format=%aI", rev).strip(),
           "user_agent_parsers_at_D": len(frozen.rules), "user_agent_parsers_at_pinned": len(head.rules),
           "python_re_uncompilable": {"at_D": frozen.uncompilable, "at_pinned": head.uncompilable},
           "families_pre_D": len(fam), "K_99pct": K, "post_D_fixtures_outside_vocab": round(sum(r["family"] not in vocab for r in post) / len(post), 4)}
    table = {}
    for name, parser, pool in (("frozen_pre_D", frozen, pre), ("frozen_post_D", frozen, post), ("pinned_post_D", head, post)):
        got = [parser.parse(r["ua"]) for r in pool]
        gold = [(r["family"], r["major"], r["minor"]) for r in pool]
        no_match = np.array([g[0] == "Other" and t[0] != "Other" for g, t in zip(got, gold)], dtype=int)
        wrong_family = np.array([g[0] != "Other" and g[0] != t[0] for g, t in zip(got, gold)], dtype=int)
        wrong_version = np.array([g[0] == t[0] and g != t for g, t in zip(got, gold)], dtype=int)
        table[name] = {"n": len(pool), "no_match": stats.bootstrap_ci(no_match).as_dict(), "silent_wrong_family": stats.bootstrap_ci(wrong_family).as_dict(),
                       "silent_wrong_version_only": stats.bootstrap_ci(wrong_version).as_dict(), "exact": stats.bootstrap_ci(1 - no_match - wrong_family - wrong_version).as_dict()}
        if name == "frozen_post_D":
            per_year = collections.defaultdict(lambda: [0, 0, 0, 0])
            for r, a, b, c in zip(pool, no_match, wrong_family, wrong_version):
                y = per_year[r["date"][:4]]
                y[0] += 1; y[1] += int(a); y[2] += int(b); y[3] += int(c)
            table["frozen_post_D_by_year"] = {y: {"n": v[0], "no_match": round(v[1] / v[0], 4), "silent_wrong_family": round(v[2] / v[0], 4),
                                                  "silent_wrong_version_only": round(v[3] / v[0], 4)} for y, v in sorted(per_year.items())}
    out["python_re_preview"] = table
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--also", nargs="*", default=["2015-01-01"], help="other freeze dates to evaluate next to the rule's choice")
    args = ap.parse_args()
    assert git("rev-parse", "HEAD").strip() == PINNED or git("cat-file", "-t", PINNED).strip() == "commit"
    OUT.mkdir(parents=True, exist_ok=True)
    seen = first_seen()
    ua, os_ = fixtures(UA_FILES), fixtures(OS_FILES)
    by_string: dict[str, dict] = {}
    for c in ua:
        d = seen[c["_raw"]]
        s = c["user_agent_string"]
        if s not in by_string or d < by_string[s]["date"]:
            by_string[s] = {"ua": s, "date": d, "family": c["family"], "major": c.get("major"), "minor": c.get("minor"), "file": c["_file"]}
    os_by = {c["user_agent_string"]: c["family"] for c in os_}
    rows = sorted(by_string.values(), key=lambda r: (r["date"], r["ua"]))
    for r in rows:
        r["os_family"] = os_by.get(r["ua"])
    n = len(rows)
    years = collections.Counter(r["date"][:4] for r in rows)
    last = max(r["date"] for r in rows)

    candidates = []
    for y in range(int(min(years)) + 1, int(max(years)) + 1):
        D = f"{y}-01-01"
        post = [r for r in rows if r["date"] >= D]
        span = (dt.date.fromisoformat(last) - dt.date.fromisoformat(min(r["date"] for r in post))).days / 365.25 if post else 0
        candidates.append({"D": D, "pre": n - len(post), "post": len(post), "post_share": round(len(post) / n, 4), "post_span_years": round(span, 2),
                           "ok": len(post) / n >= 0.30 and span >= 3})
    ok = [c for c in candidates if c["ok"]]
    summary = {"uap_core_commit": PINNED, "fixtures_raw": len(ua), "fixtures_unique": n, "with_os_gold": sum(r["os_family"] is not None for r in rows),
               "per_file_unique": dict(collections.Counter(r["file"] for r in rows)), "first_seen_by_year": dict(sorted(years.items())), "candidates": candidates}
    if not ok:
        summary["D"] = None
        summary["flag"] = "no 1 January satisfies post-D share >= 30% with a span >= 3 years"
    else:
        summary.update(rule_choice=ok[-1], **evaluate(rows, ok[-1]["D"]))
        summary["preview_note"] = ("Preview with Python's re, not the RE2 legacy build; gold is uap-core's own fixture label, which the pinned regex list "
                                   "reproduces by construction, so pinned_post_D is a sanity check of this harness, not a baseline.")
    summary["alternatives"] = [evaluate(rows, D) for D in args.also]
    bulk = max(years.items(), key=lambda kv: kv[1])
    summary["flag"] = (f"{bulk[1]} of {n} fixtures ({bulk[1] / n:.0%}) first appear in {bulk[0]}, one bulk import of a historical browser list; "
                       "the freeze-date rule is therefore satisfied only by a D that leaves a few hundred pre-D fixtures.") if bulk[1] / n > 0.5 else None
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (OUT / "fixture_dates.jsonl").write_text("".join(json.dumps({"date": r["date"], "file": r["file"], "family": r["family"]}) + "\n" for r in rows))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("candidates", "python_re_preview")}, indent=1))
    for k, v in [kv for e in [summary, *summary["alternatives"]] for kv in [("D=" + str(e.get("D")), {}), *e.get("python_re_preview", {}).items()]]:
        print(k, {a: (round(b["point"], 4) if isinstance(b, dict) and "point" in b else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
