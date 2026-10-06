"""A private Hugging Face dataset copy of `data/`: the suites, raw sources, normalized splits and `data/clean`.

Rebuilding `data/` takes `make data data-raw-* normalize clean-data`: many downloads and several minutes. With this
copy, a fresh machine (the GPU box) gets the same bytes in one resumable download. Both directions check integrity.
The upload refuses pinned files that don't match `locks/`, and sends a sha256 manifest of every file. The download
checks every file against that manifest, then the pinned ones against `locks/` again.

    den data-upload   --repo <org>/den-data          # make upload-data DATA_REPO=<org>/den-data
    den data-download --repo <org>/den-data[@rev]    # make download-data DATA_REPO=<org>/den-data
"""

from __future__ import annotations

import argparse
import json
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .fetch import digest, read_lock

DATA = Path("data")
LOCKS = Path("locks")
MANIFEST = "MANIFEST.json"
IGNORE = (".normalize/*", ".cache/*", "*/.cache/*", "*.part", "*.tmp", ".DS_Store", "*/.DS_Store", MANIFEST)


def files(root: Path) -> dict[str, dict[str, Any]]:
    """Every file under `root` that belongs in the copy, by relative path: its size and sha256."""
    found = (p for p in sorted(root.rglob("*")) if p.is_file())
    return {
        rel: {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in found
        if not any(fnmatch(rel := p.relative_to(root).as_posix(), pattern) for pattern in IGNORE)
    }


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
    listed = files(root)
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": listed}
    total = sum(f["bytes"] for f in listed.values())
    print(f"uploading {len(listed)} files, {total / 1e9:.2f} GB, to datasets/{repo}", flush=True)
    create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    upload_large_folder(repo, root, repo_type="dataset", private=private, ignore_patterns=list(IGNORE))
    api = HfApi()
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
    return problems + pinned(root, present_only=True)


def upload_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den data-upload")
    p.add_argument("--repo", required=True, help="<org>/<name>: a Hugging Face dataset repo, created private")
    p.add_argument("--public", action="store_true", help="publish publicly; private by default")
    args = p.parse_args(argv)
    sha = upload(args.repo, private=not args.public)
    print(f"https://huggingface.co/datasets/{args.repo}  commit {sha}")
    print(f"pin it: make download-data DATA_REPO={args.repo}@{sha}")
    return 0


def download_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den data-download")
    p.add_argument("--repo", required=True, help="<org>/<name>[@revision] of a `den data-upload` copy")
    args = p.parse_args(argv)
    problems = download(args.repo)
    for problem in problems:
        print(f"  BAD {problem}")
    print(f"data/ from datasets/{args.repo}: " + ("ok, every file matches" if not problems else "FAILED"))
    return 1 if problems else 0
