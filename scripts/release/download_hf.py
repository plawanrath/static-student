"""Download released checkpoints from the Hub into models/<name>/ so the RUNBOOK commands find them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, NAMESPACE, PREFIX, repo_id  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="e.g. tel-mlx-S-w4 modernbert-large-civil")
    ap.add_argument("--all", action="store_true", help="every static-student-* repo in the namespace")
    args = ap.parse_args()
    names = list(args.names)
    if args.all:
        names += [m.id.split("/", 1)[1][len(PREFIX):] for m in HfApi().list_models(author=NAMESPACE) if m.id.split("/", 1)[1].startswith(PREFIX)]
    if not names:
        ap.error("give model names or --all")
    for n in sorted(set(names)):
        out = snapshot_download(repo_id(n), local_dir=MODELS / n)
        print(f"{repo_id(n)} -> {out}")


if __name__ == "__main__":
    main()
