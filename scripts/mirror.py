"""A private Hugging Face dataset copy of `data/`: the suites, raw sources, normalized splits and `data/clean`.

Rebuilding `data/` takes `make data data-raw-* normalize clean-data`: many downloads and several minutes. With this
copy, a fresh machine (the GPU box) gets the same bytes in one resumable download. Both directions check integrity.
The upload refuses pinned files that don't match `locks/`, and sends a sha256 manifest of every file. The download
checks every file against that manifest, then the pinned ones against `locks/` again.

Licences decide what goes up (`den/licences.py`): the notices (`README.md` as the dataset card, `LICENSES.md`,
`LICENSES/`, a `SOURCE-LICENSE.md` per raw source) are written first; files whose sources forbid redistribution are
never uploaded, and files with no stated licence only to a private repo (a repo that is already public always gets
the public rules). What was left out is listed in the card and the
manifest, with the command that rebuilds it from the pinned originals.

    uv run python -m scripts.mirror upload   --repo <org>/den-data         # make upload-data DATA_REPO=<org>/den-data
    uv run python -m scripts.mirror download --repo <org>/den-data[@rev]   # make download-data DATA_REPO=...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from den.fetch import digest, read_lock
from den.licences import IGNORE, data_files, notices

DATA = Path("data")
LOCKS = Path("locks")
MANIFEST = "MANIFEST.json"
REBUILD = "make data data-raw-train data-raw-new data-raw-eval breadth normalize clean-data"


def files(root: Path, leave_out: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    """Every file under `root` that belongs in the copy, by relative path: its size and sha256."""
    skip = set(leave_out)
    return {rel: {"bytes": (root / rel).stat().st_size, "sha256": digest(root / rel)}
            for rel in data_files(root) if rel not in skip}  # fmt: skip


def pinned(root: Path, present_only: bool) -> list[str]:
    """Pinned files under `root` that differ from `locks/` (and, unless `present_only`, those missing)."""
    problems = []
    for path in sorted(LOCKS.glob("*.json")):
        lock = read_lock(path)
        if lock is None or Path(lock["root"]) != DATA:
            continue
        for key, entry in lock["entries"].items():
            for rel, f in entry["files"].items():
                if not (root / rel).is_file():
                    if not present_only:
                        problems.append(f"{path.name} {key}: missing {rel}")
                elif digest(root / rel) != f["sha256"]:
                    problems.append(f"{path.name} {key}: {rel} differs from its pin")
    return problems


def upload(repo: str, root: Path = DATA, private: bool = True) -> str:
    """Upload `root` as a dataset repo, with its manifest; returns the commit to pin a download to."""
    from huggingface_hub import HfApi, create_repo, upload_large_folder

    if problems := pinned(root, present_only=True):
        raise SystemExit("pinned files differ from locks/, not uploading:\n  " + "\n  ".join(problems[:20]))
    api = HfApi()
    public = not private
    if not public and api.repo_exists(repo, repo_type="dataset") and not api.dataset_info(repo).private:
        print(f"{repo} is public: uploading under public rules (files with no stated licence are left out too)")
        public = True  # the stricter copy; a private one would publish what may only be kept privately
    classified, excluded = notices(root, public=public, repo=repo)  # writes the card and licence files into root
    listed = files(root, excluded)
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": listed,
                "licences": {rel: {"class": k, "sources": keys} for rel, (k, keys) in classified.items()},
                "excluded": excluded, "rebuild_excluded": REBUILD}  # fmt: skip
    total = sum(f["bytes"] for f in listed.values())
    print(f"uploading {len(listed)} files, {total / 1e9:.2f} GB, to datasets/{repo}; leaving out {len(excluded)} "
          f"(licence: see README.md)", flush=True)  # fmt: skip
    create_repo(repo, repo_type="dataset", private=not public, exist_ok=True)
    if not public and not api.dataset_info(repo).private:  # made public between the check and now: refuse
        raise SystemExit(f"{repo} became public during the upload; run it again to upload under public rules")
    stale = sorted(set(api.list_repo_files(repo, repo_type="dataset")) & set(excluded))
    if stale:  # an earlier upload carried files the licences now leave out
        api.delete_files(
            repo, delete_patterns=stale, repo_type="dataset", commit_message="den: remove files the licences exclude"
        )
    upload_large_folder(repo, root, repo_type="dataset", private=not public, ignore_patterns=[*IGNORE, *excluded])
    api.upload_file(  # last: a manifest on the Hub means every file before it landed
        path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode(),
        path_in_repo=MANIFEST,
        repo_id=repo,
        repo_type="dataset",
        commit_message="den data manifest",
    )
    remote = set(api.list_repo_files(repo, repo_type="dataset"))
    if absent := sorted(set(listed) - remote):
        raise SystemExit(f"uploaded, but {repo} lacks {len(absent)} files ({absent[:3]}): run the same command again")
    sha: str = api.dataset_info(repo).sha or "main"
    return sha


def download(spec: str, root: Path = DATA) -> list[str]:
    """Download `<org>/<name>[@revision]` into `root`; returns every integrity problem (none: the copy is exact)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    repo, _, revision = spec.partition("@")
    manifest = json.loads(
        Path(hf_hub_download(repo, MANIFEST, repo_type="dataset", revision=revision or None)).read_text("utf-8")
    )
    root.mkdir(exist_ok=True)
    snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision or None,
        local_dir=root,
        ignore_patterns=[MANIFEST, ".gitattributes"],
    )
    problems = []
    for rel, f in manifest["files"].items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif path.stat().st_size != f["bytes"] or digest(path) != f["sha256"]:
            problems.append(f"{rel} differs from the uploaded file")
    for rel, why in sorted(manifest.get("excluded", {}).items()):
        print(f"  not in this copy: {rel}  ({why})")
    if manifest.get("excluded"):
        print(f"rebuild them from the pinned originals: {manifest.get('rebuild_excluded', REBUILD)}")
    return problems + pinned(root, present_only=True)


def upload_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.mirror upload")
    p.add_argument("--repo", required=True, help="<org>/<name>: a Hugging Face dataset repo, created private")
    p.add_argument(
        "--public",
        action="store_true",
        help="publish publicly (leaves out unlicensed files too); automatic for a public repo",
    )
    args = p.parse_args(argv)
    sha = upload(args.repo, private=not args.public)
    print(f"https://huggingface.co/datasets/{args.repo}  commit {sha}")
    print(f"pin it: make download-data DATA_REPO={args.repo}@{sha}")
    return 0


def download_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.mirror download")
    p.add_argument("--repo", required=True, help="<org>/<name>[@revision] of a `make upload-data` copy")
    args = p.parse_args(argv)
    problems = download(args.repo)
    for problem in problems:
        print(f"  BAD {problem}")
    print(f"data/ from datasets/{args.repo}: " + ("ok, every file matches" if not problems else "FAILED"))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    commands = {"upload": upload_main, "download": download_main}
    if not args or args[0] not in commands:
        raise SystemExit("usage: python -m scripts.mirror {upload,download} --repo <org>/<name>[@revision]")
    return commands[args[0]](args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
