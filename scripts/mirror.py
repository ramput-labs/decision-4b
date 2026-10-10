"""A Hugging Face dataset copy of `data/`: the suites, raw sources, normalized splits and `data/clean`.

`make data-download` is the one way to get `data/` on a new machine (the Mac, the local RTX box, a cloud H100): one
resumable download, then a rebuild of only what the copy could not carry. Both directions check integrity. The
upload refuses pinned files that don't match `locks/` and sends a sha256 manifest of every file, the ones it leaves
out included. The download checks every file against that manifest, then the pinned ones against `locks/` again.

Licences decide what goes up (`den/licences.py`): the notices (`README.md` as the dataset card, `LICENSES.md`,
`LICENSES/`, a `SOURCE-LICENSE.md` per raw source) are written first; files whose sources forbid redistribution are
never uploaded, and files with no stated licence only to a private repo (a repo that is already public always gets
the public rules). The download rebuilds what was left out (`rebuild`) and runs only the steps whose outputs are
missing or wrong: the pinned originals are fetched item by item, then breadth, normalize and clean, which are
deterministic and so rewrite the files the copy did carry byte for byte. Rebuilt files are checked against the
uploader's sha256 (`excluded_files`; older manifests only get an existence check for unpinned files), so a second
download on the same machine skips the rebuild.

    uv run python -m scripts.mirror upload   --repo <org>/den-data         # make data-upload DATA_REPO=<org>/den-data
    uv run python -m scripts.mirror download --repo <org>/den-data[@rev]   # make data-download DATA_REPO=...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from den.fetch import digest, fetch_item, locked_digests, read_lock
from den.licences import IGNORE, data_files, notices
from den.paths import DATA, LOCKS, model_dir
from den.pins import ITEMS, MODELS

MANIFEST = "MANIFEST.json"
REBUILD = "make data-download (fetch, breadth, normalize, clean: only the steps whose files are missing)"
STEPS = ("fetch", "breadth", "normalize", "clean")
BREADTH_MODEL = (
    "qwen3.5-4b"  # scripts.build_breadth admits records with Kev's tokenizer, models/qwen3.5-4b/tokenizer.json
)


def files(root: Path, leave_out: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    """Every file under `root` that belongs in the copy, by relative path: its size and sha256."""
    skip = set(leave_out)
    return {rel: {"bytes": (root / rel).stat().st_size, "sha256": digest(root / rel)}
            for rel in data_files(root) if rel not in skip}  # fmt: skip


def locked() -> dict[str, tuple[str, str, str]]:
    """Every pinned file under data/, by relative path: (lock file, item key, sha256)."""
    pins: dict[str, tuple[str, str, str]] = {}
    for path in sorted(LOCKS.glob("*.json")):
        lock = read_lock(path)
        if lock is None or Path(lock["root"]) != DATA:
            continue
        for key, entry in lock["entries"].items():
            pins |= {rel: (path.name, key, f["sha256"]) for rel, f in entry["files"].items()}
    return pins


class Hashes:
    """sha256 of files under `root`, each read once: a 3.6 GB copy is checked against its manifest and then against
    `locks/`, and the second pass reuses the first. `forget` drops paths a rebuild may have rewritten."""

    def __init__(self, root: Path) -> None:
        self.root, self.known = root, dict[str, str]()

    def __call__(self, rel: str) -> str:
        if rel not in self.known:
            self.known[rel] = digest(self.root / rel)
        return self.known[rel]

    def forget(self, rels: Iterable[str]) -> None:
        for rel in rels:
            self.known.pop(rel, None)


def pinned(root: Path, present_only: bool, sha: Callable[[str], str] | None = None) -> list[str]:
    """Pinned files under `root` that differ from `locks/` (and, unless `present_only`, those missing)."""
    sha = sha or Hashes(root)
    problems = []
    for rel, (lock, key, want) in locked().items():
        if not (root / rel).is_file():
            if not present_only:
                problems.append(f"{lock} {key}: missing {rel}")
        elif sha(rel) != want:
            problems.append(f"{lock} {key}: {rel} differs from its pin")
    return problems


def upload(repo: str, root: Path = DATA, private: bool = True, tag: str | None = None) -> str:
    """Upload `root` as a dataset repo, with its manifest; returns the commit to pin a download to."""
    from huggingface_hub import HfApi, create_repo

    from den.publish import upload_folder

    if problems := pinned(root, present_only=True):
        raise SystemExit("pinned files differ from locks/, not uploading:\n  " + "\n  ".join(problems[:20]))
    api = HfApi()
    public = not private
    if not public and api.repo_exists(repo, repo_type="dataset") and not api.dataset_info(repo).private:
        print(f"{repo} is public: uploading under public rules (files with no stated licence are left out too)")
        public = True  # the stricter copy; a private one would publish what may only be kept privately
    classified, excluded = notices(root, public=public, repo=repo)  # writes the card and licence files into root
    listed = files(root, excluded)
    # the left-out files' hashes (never their bytes): the download checks its rebuild against them
    left_out = {rel: f for rel, f in files(root).items() if rel in excluded}
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": listed,
                "licences": {rel: {"class": k, "sources": keys} for rel, (k, keys) in classified.items()},
                "excluded": excluded, "excluded_files": left_out, "rebuild_excluded": REBUILD}  # fmt: skip
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
    upload_folder(repo, root, "dataset", private=not public, ignore=[*IGNORE, *excluded])
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
    if tag:  # a data version: `make data-download DATA_REPO=<repo>@<tag>` and `den release --data-repo` name it
        if tag in {t.name for t in api.list_repo_refs(repo, repo_type="dataset").tags}:
            raise SystemExit(f"{repo} already has a tag {tag}: data versions are immutable (commit {sha} is uploaded)")
        api.create_tag(repo, tag=tag, revision=sha, repo_type="dataset", tag_message=f"den data {tag}")
    return sha


def step(rel: str, pins: dict[str, tuple[str, str, str]]) -> str:
    """The rebuild step that writes `rel`: fetch (a pinned file), breadth, normalize (`<split>/sources/`) or clean."""
    parts = rel.split("/")
    if rel in pins:
        return "fetch"
    if parts[0] == "clean":
        return "clean"
    if parts[0] in ("dev", "test") and parts[1:] == ["breadth.jsonl"]:
        return "breadth"
    if len(parts) > 2 and parts[1] == "sources":
        return "normalize"
    return "all"  # a path no single step owns: run every step


def wanted(root: Path, excluded: Iterable[str], expected: dict[str, dict[str, Any]], sha: Hashes) -> list[str]:
    """The left-out files that are missing here or differ from the uploader's (or the pinned) sha256. An unpinned file
    the manifest has no hash for can't be checked, so it is rebuilt."""
    pins = locked()
    out = []
    for rel in sorted(excluded):
        want = expected.get(rel, {}).get("sha256") or (pins[rel][2] if rel in pins else None)
        if not (root / rel).is_file() or want is None or sha(rel) != want:
            out.append(rel)
    return out


def tokenizer() -> None:
    """Breadth admits records with Kev's tokenizer. A box training a smaller model needn't download 9 GB of qwen3.5-4b
    for it: fetch that one pinned file at the pinned commit (every Qwen3.5 size ships the same bytes)."""
    from huggingface_hub import hf_hub_download

    model = MODELS[BREADTH_MODEL]
    path = model_dir(BREADTH_MODEL) / "tokenizer.json"
    pin = next(f.sha256 for f in model.verify if f.path == "tokenizer.json")
    if path.is_file() and digest(path) == pin:
        return
    hf_hub_download(model.source.repo, "tokenizer.json", revision=model.source.revision, local_dir=path.parent)
    if digest(path) != pin:
        raise SystemExit(f"{path} differs from its pin {pin}")


def fetch(root: Path, rels: Iterable[str]) -> int:
    """Only the pinned items that hold `rels`; files already present and matching are not downloaded again."""
    items = {item.key: item for item in ITEMS}
    pins = locked()
    keys = sorted({pins[rel][1] for rel in rels if rel in pins})
    digests: dict[str, Any] = {}
    for path in sorted(LOCKS.glob("*.json")):
        if (lock := read_lock(path)) is not None and Path(lock["root"]) == DATA:
            digests |= locked_digests(lock)
    for key in keys:
        print(f"  fetch {key}", flush=True)
        fetch_item(items[key], root, digests)
    return 0


def rebuild(root: Path, missing: list[str]) -> None:
    """The steps that write `missing`, in order, and nothing else. Needs no model: breadth's tokenizer is fetched."""
    from den.cli import main as den
    from scripts import build_breadth as breadth

    pins = locked()
    needed = {step(rel, pins) for rel in missing}
    if "all" in needed:
        needed = set(STEPS)
    if "breadth" in needed:
        tokenizer()
    runs: dict[str, Callable[[], int]] = {
        "fetch": lambda: fetch(root, missing),
        "breadth": lambda: breadth.main([]),
        "normalize": lambda: den(["normalize"]),
        "clean": lambda: den(["clean"]),
    }
    for name in (s for s in STEPS if s in needed):
        print(f"rebuild: {name}", flush=True)
        if code := runs[name]():
            raise SystemExit(f"rebuilding the left-out files failed at {name} (exit {code}); fix it and rerun")


def download(spec: str, root: Path = DATA, rebuild_excluded: bool = True) -> list[str]:
    """Download `<org>/<name>[@revision]` into `root` and rebuild what the licences left out of it; returns every
    integrity problem (none: `root` is complete and exact)."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    repo, _, revision = spec.partition("@")
    # one commit for the manifest and the files, so an upload landing mid-download can't mix two copies
    commit = HfApi().dataset_info(repo, revision=revision or None).sha or revision or "main"
    print(f"data commit: {repo}@{commit}", flush=True)  # what `make release DATA_REPO=` should record
    manifest = json.loads(
        Path(hf_hub_download(repo, MANIFEST, repo_type="dataset", revision=commit)).read_text("utf-8")
    )
    root.mkdir(exist_ok=True)
    snapshot_download(
        repo,
        repo_type="dataset",
        revision=commit,
        local_dir=root,
        ignore_patterns=[MANIFEST, ".gitattributes"],
    )
    sha = Hashes(root)
    problems = []
    for rel, f in manifest["files"].items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif path.stat().st_size != f["bytes"] or sha(rel) != f["sha256"]:
            problems.append(f"{rel} differs from the uploaded file")
    excluded: dict[str, str] = manifest.get("excluded", {})
    expected: dict[str, dict[str, Any]] = manifest.get("excluded_files", {})
    missing = wanted(root, excluded, expected, sha)
    print(f"{len(manifest['files'])} files downloaded and checked; {len(excluded)} left out by their licences, "
          f"{len(excluded) - len(missing)} of them already here and verified")  # fmt: skip
    if missing and not problems and rebuild_excluded:
        for rel in missing:
            print(f"  to rebuild: {rel}  ({excluded[rel]})")
        rebuild(root, missing)
        sha.forget(excluded)
        problems += [f"rebuilt {rel} is missing" for rel in missing if not (root / rel).is_file()]
        problems += [f"rebuilt {rel} differs from the uploader's copy" for rel in missing
                     if (root / rel).is_file() and rel in expected and sha(rel) != expected[rel]["sha256"]]  # fmt: skip
    elif missing:
        print(f"{len(missing)} left-out files not rebuilt; rerun without --no-rebuild: {REBUILD}")
    return problems + pinned(root, present_only=True, sha=sha)


def upload_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.mirror upload")
    p.add_argument("--repo", required=True, help="<org>/<name>: a Hugging Face dataset repo, created private")
    p.add_argument(
        "--public",
        action="store_true",
        help="publish publicly (leaves out unlicensed files too); automatic for a public repo",
    )
    p.add_argument("--tag", help="tag this data version on the Hub (e.g. data-v1); tags are immutable")
    args = p.parse_args(argv)
    sha = upload(args.repo, private=not args.public, tag=args.tag)
    print(f"https://huggingface.co/datasets/{args.repo}  commit {sha}")
    print(f"pin it: make data-download DATA_REPO={args.repo}@{args.tag or sha}")
    return 0


def download_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.mirror download")
    p.add_argument("--repo", required=True, help="<org>/<name>[@revision] of a `make data-upload` copy")
    p.add_argument("--no-rebuild", action="store_true", help="don't rebuild the files the licences left out")
    args = p.parse_args(argv)
    started = time.time()
    problems = download(args.repo, rebuild_excluded=not args.no_rebuild)
    for problem in problems:
        print(f"  BAD {problem}")
    took = f"{(time.time() - started) / 60:.1f} min"
    print(f"data/ from datasets/{args.repo}: " + (f"ok, every file matches ({took})" if not problems else "FAILED"))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    commands = {"upload": upload_main, "download": download_main}
    if not args or args[0] not in commands:
        raise SystemExit("usage: python -m scripts.mirror {upload,download} --repo <org>/<name>[@revision]")
    return commands[args[0]](args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
