"""Stamp a DOI into every citation: CITATION.cff, README.md, scripts/release/common.py and the 25 Hub model cards.

    .venv/bin/python scripts/release/add_doi.py 10.5281/zenodo.NNNNNNN            # local files + Hub cards
    .venv/bin/python scripts/release/add_doi.py 10.5281/zenodo.NNNNNNN --dry-run  # show what would change

Use the Zenodo *concept* DOI (the one that always resolves to the latest version) so the citation survives new
releases. The Hub cards are patched in place (download README.md, replace the BibTeX block, upload), so the
checkpoints do not need to be on this machine.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REPO = common.REPO
BIB_RE = re.compile(r"@software\{rath2026staticstudent,\n(.*?)\n\}", re.S)


def bibtex_with_doi(doi: str) -> str:
    bib = common.BIBTEX
    if "doi" in bib:
        bib = re.sub(r"  doi     = \{[^}]*\},\n", "", bib)
    return bib.replace("  url     = {", f"  doi     = {{{doi}}},\n  url     = {{")


def patch_bibtex(text: str, doi: str) -> str:
    return BIB_RE.sub(lambda m: bibtex_with_doi(doi), text)


def patch_cff(text: str, doi: str) -> str:
    text = re.sub(r"^doi: .*\n", "", text, flags=re.M)
    text = re.sub(r"^identifiers:\n(  .*\n)*", "", text, flags=re.M)
    return text.replace("repository-code:", f"doi: {doi}\nidentifiers:\n  - type: doi\n    value: {doi}\n    description: Zenodo archive of the released code (concept DOI, all versions)\nrepository-code:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doi", help="e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--local-only", action="store_true", help="skip the Hub cards")
    args = ap.parse_args()
    doi = args.doi.strip().removeprefix("https://doi.org/")

    edits = {
        REPO / "CITATION.cff": patch_cff,
        REPO / "README.md": patch_bibtex,
        REPO / "scripts" / "release" / "common.py": patch_bibtex,
    }
    for path, fn in edits.items():
        old = path.read_text()
        new = fn(old, doi)
        print(f"{'would change' if args.dry_run else 'changed'}: {path.relative_to(REPO)}" if new != old else f"unchanged: {path.relative_to(REPO)}")
        if not args.dry_run and new != old:
            path.write_text(new)
    if args.local_only:
        return

    api = HfApi()
    ids = sorted(m.id for m in api.list_models(author=common.NAMESPACE) if m.id.split("/", 1)[1].startswith(common.PREFIX))
    for rid in ids:
        local = Path(hf_hub_download(rid, "README.md"))
        old = local.read_text()
        new = patch_bibtex(old, doi)
        if new == old:
            print(f"unchanged: {rid}")
            continue
        if args.dry_run:
            print(f"would update card: {rid}")
            continue
        api.upload_file(path_or_fileobj=new.encode(), path_in_repo="README.md", repo_id=rid,
                        commit_message=f"model card: add DOI {doi}")
        print(f"updated card: {rid}")


if __name__ == "__main__":
    main()
