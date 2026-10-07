"""Create or update one Hub repo per release/<name>/ and put them all in one collection.

Repos are created private; `--public` flips every repo in the release to public (do this once the GitHub repository
the cards link to is public).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COLLECTION_DESCRIPTION, COLLECTION_TITLE, NAMESPACE, RELEASE, repo_id  # noqa: E402


def release_dirs(names: list[str]) -> list[Path]:
    if names:
        return [RELEASE / n for n in names]
    return sorted(p for p in RELEASE.iterdir() if p.is_dir())


def find_collection(api: HfApi):
    for c in api.list_collections(owner=NAMESPACE):
        if c.title == COLLECTION_TITLE:
            return api.get_collection(c.slug)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--public", action="store_true", help="make every repo (and the collection) public; no upload")
    ap.add_argument("--no-collection", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    dirs = release_dirs(args.names)

    if args.public:
        for d in dirs:
            api.update_repo_settings(repo_id(d.name), private=False)
            print(f"public: {repo_id(d.name)}")
        c = find_collection(api)
        if c:
            api.update_collection_metadata(c.slug, private=False)
            print(f"public: https://huggingface.co/collections/{c.slug}")
        return

    for d in dirs:
        rid = repo_id(d.name)
        api.create_repo(rid, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(repo_id=rid, folder_path=str(d), repo_type="model",
                          commit_message=f"release {d.name} from static-student")
        print(f"uploaded: https://huggingface.co/{rid}")

    if not args.no_collection:
        c = find_collection(api) or api.create_collection(title=COLLECTION_TITLE, namespace=NAMESPACE,
                                                          description=COLLECTION_DESCRIPTION, private=True)
        have = {it.item_id for it in c.items}
        for d in dirs:
            if repo_id(d.name) not in have:
                api.add_collection_item(c.slug, item_id=repo_id(d.name), item_type="model", exists_ok=True)
        print(f"collection: https://huggingface.co/collections/{c.slug}")


if __name__ == "__main__":
    main()
