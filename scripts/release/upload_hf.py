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
from common import COLLECTION_DESCRIPTION, COLLECTION_TITLE, NAMESPACE, PREFIX, RELEASE, repo_id  # noqa: E402


def release_names(api: HfApi, names: list[str]) -> list[str]:
    """Model names: the ones given, else the local export, else every static-student-* repo on the Hub."""
    if names:
        return names
    if RELEASE.exists():
        return sorted(p.name for p in RELEASE.iterdir() if p.is_dir())
    return sorted(m.id.split("/", 1)[1][len(PREFIX):] for m in api.list_models(author=NAMESPACE)
                  if m.id.split("/", 1)[1].startswith(PREFIX))


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
    ap.add_argument("--collection-only", action="store_true", help="skip uploads; create/update the collection")
    args = ap.parse_args()
    api = HfApi()
    names = release_names(api, args.names)

    if args.public:
        for n in names:
            api.update_repo_settings(repo_id(n), private=False)
            print(f"public: {repo_id(n)}")
        c = find_collection(api)
        if c:
            api.update_collection_metadata(c.slug, private=False)
            print(f"public: https://huggingface.co/collections/{c.slug}")
        return

    for n in [] if args.collection_only else names:
        rid = repo_id(n)
        api.create_repo(rid, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(repo_id=rid, folder_path=str(RELEASE / n), repo_type="model",
                          commit_message=f"release {n} from static-student")
        print(f"uploaded: https://huggingface.co/{rid}")

    if not args.no_collection:
        c = find_collection(api) or api.create_collection(title=COLLECTION_TITLE, namespace=NAMESPACE,
                                                          description=COLLECTION_DESCRIPTION, private=True)
        have = {it.item_id for it in c.items}
        for n in names:
            if repo_id(n) not in have:
                api.add_collection_item(c.slug, item_id=repo_id(n), item_type="model", exists_ok=True)
        print(f"collection: https://huggingface.co/collections/{c.slug}")


if __name__ == "__main__":
    main()
