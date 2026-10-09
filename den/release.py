"""Versioned releases: v1, v2, ... each a published run, tagged on the Hub and recorded in git.

A release is immutable. `den release create` publishes a finished run (`den publish`), tags that Hub commit with the
version, and writes `releases/<version>.json`: the base, LoRA and head, every stage's data (sha256 of each file and
the lines used), the dev and locked-test results, the model files' sha256 (`integrity.json`), the Hub commit, the data
repo commit it was trained from, and its parent version. Commit the record: it is what `release:<version>` resolves.

The next version continues from the last one, `den train --init-from release:v1` (the adapter and head come from the
tagged Hub commit, so the GPU box that trained v1 need not exist), on the new data plus replay of the old
(`make train-next`). Before v2 is released it must not regress on v1: on every dev file both were evaluated on, no
accuracy drop above `--max-drop` (the rule `den compare` ships rounds by), unless the reason is given and recorded.

    den release create --version v1 --run runs/round2 --repo <org>/<name> --data-repo <org>/den-datasets@<commit>
    den release create --version v2 --run runs/v2 --repo <org>/<name> --parent v1
    den release list
    den evaluate --run release:v1 dev/core.jsonl          # any --run / --init-from takes release:<version>
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from .evidence import keys, name, paired

RELEASES = Path("releases")
VERSION = re.compile(r"v\d+(\.\d+)*")
MAX_DROP = 0.01  # the largest dev accuracy drop on any file a new version may have (`den compare`'s rule)


def _json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return loaded


def load(version: str, root: Path | None = None) -> dict[str, Any]:
    if not (path := (root or RELEASES) / f"{version}.json").is_file():
        raise SystemExit(f"no release {version}: {path} is missing (den release list)")
    return _json(path)


def resolve(spec: str, root: Path | None = None) -> str:
    """`release:<version>` -> `hf:<repo>@<version>`, the tagged Hub commit; anything else unchanged."""
    if not spec.startswith("release:"):
        return spec
    record = load(spec.removeprefix("release:"), root)
    return f"hf:{record['hub']['repo']}@{record['version']}"


def locate(run: str, files: list[str] | None = None) -> Path:
    """A run directory: a local path, or `hf:<org>/<name>[@<revision>]` / `release:<version>` downloaded once into the
    Hub cache (only `files`, when given: continuing from a run needs its adapter and head, not its merged weights)."""
    run = resolve(run)
    if not run.startswith("hf:"):
        return Path(run)
    from huggingface_hub import snapshot_download

    repo, _, revision = run.removeprefix("hf:").partition("@")
    return Path(snapshot_download(repo, revision=revision or None, allow_patterns=files))


def regressions(parent: dict[str, Any], child: dict[str, Any], max_drop: float = MAX_DROP) -> dict[str, float]:
    """Dev files (both evaluated, never test) where the child's accuracy fell over `max_drop` below the parent's."""
    shared = sorted(f for f in set(parent) & set(child) if Path(f.split("+")[0]).parts[0] != "test" and "+" not in f)
    drops = {f: parent[f]["accuracy"] - child[f]["accuracy"] for f in shared}
    return {f: round(d, 4) for f, d in drops.items() if d > max_drop}


def record(
    version: str,
    run: Path,
    repo: str,
    revision: str,
    parent: str | None = None,
    data_repo: str | None = None,
    regression_note: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """The release record of a finished, evaluated run."""
    from .publish import stages

    trained, evaluated, head = _json(run / "run.json"), _json(run / "eval.json"), _json(run / "head.json")
    data, _, data_revision = (data_repo or "").partition("@")
    return {
        "version": version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent": parent,
        "hub": {"repo": repo, "revision": revision, "tag": version},
        "data": {"repo": data or None, "revision": data_revision or None},
        "base": trained.get("model") or trained.get("base"),
        "lora": trained.get("lora"),
        "head": {k: head.get(k) for k in ("kind", "dim", "layers", "heads", "rep", "proj", "hidden_size")},
        "temperature": head.get("temperature"),
        "commit": trained.get("commit"),
        "stages": [
            {
                "name": path.name,
                "data": r.get("data"),
                "replay": r.get("replay"),
                "sources_cap": r.get("sources_cap"),
                "lr": r.get("lr"),
                "epochs": r.get("epochs"),
                "steps": r.get("steps"),
                "init_from": r.get("init_from"),
                "dataset_sha256": r.get("dataset_sha256"),
                "data_used": r.get("data_used"),
            }
            for path, r in stages(run)
        ],
        "dev": {f: m for f, m in evaluated.items() if Path(f.split("+")[0]).parts[0] != "test"},
        "test": {f: m for f, m in evaluated.items() if Path(f.split("+")[0]).parts[0] == "test"},
        "files": _json(run / "integrity.json").get("files", {}),
        "regression_note": regression_note,
        "evidence": name(str(run)),  # reports/runs/<this>: rows and reports, for paired comparisons with later versions
        "paired_vs_parent": against(parent, str(run), root),
    }


def against(parent: str | None, run: str, root: Path | None = None) -> dict[str, Any] | None:
    """The paired comparison with the parent version on every dev file both have committed rows for (`evidence`)."""
    if not parent:
        return None
    folder = load(parent, root).get("evidence")
    if not folder:
        return None
    earlier = f"evidence:{folder}"
    shared = sorted(k for k in set(keys(run)) & set(keys(earlier)) if Path(k).parts[0] != "test")
    return {k: got for k in shared if (got := paired(earlier, run, k)) is not None} or None


def check(
    version: str,
    run: Path,
    parent: str | None,
    root: Path = RELEASES,
    max_drop: float = MAX_DROP,
    accept: str | None = None,
) -> dict[str, float]:
    """Everything a release needs before anything is uploaded; returns the accepted regressions (if any)."""
    if not VERSION.fullmatch(version):
        raise SystemExit(f"version {version!r}: use v1, v2, v2.1, ...")
    if (root / f"{version}.json").exists():
        raise SystemExit(
            f"release {version} exists ({root / f'{version}.json'}): releases are immutable, use a new one"
        )
    integrity = _json(run / "integrity.json")
    if not integrity.get("ok"):
        raise SystemExit(f"{run}: integrity.json is missing or failed (den check-run --run {run})")
    evaluated = _json(run / "eval.json")
    if not any(Path(f.split("+")[0]).parts[0] == "test" for f in evaluated):
        raise SystemExit(f"{run}: no locked-test results in eval.json (make eval-test RUN={run})")
    if not parent:
        return {}
    previous = load(parent, root)
    trained = _json(run / "run.json")
    if (trained.get("model") or trained.get("base")) != previous.get("base"):
        raise SystemExit(f"{run} is on another base than {parent}: a new base starts a new line (no --parent)")
    if not any(f in evaluated for f in previous["dev"] if "+" not in f):
        raise SystemExit(f"{run} shares no dev file with {parent}: evaluate it on {sorted(previous['dev'])[:4]} ...")
    if (drops := regressions(previous["dev"], evaluated, max_drop)) and not accept:
        listed = ", ".join(f"{f} -{d:.4f}" for f, d in drops.items())
        raise SystemExit(
            f"{run} regresses on {parent} beyond {max_drop}: {listed}. Fix it, or release anyway with "
            "--accept-regression '<why>' (recorded in the release)."
        )
    return drops


def create(
    version: str,
    run: Path,
    repo: str,
    parent: str | None,
    data_repo: str | None,
    private: bool,
    accept: str | None,
    logs: list[Path],
    root: Path = RELEASES,
) -> dict[str, Any]:
    """Check, publish, tag the Hub commit with the version, and write the record."""
    from huggingface_hub import HfApi

    from .publish import publish

    drops = check(version, run, parent, root, accept=accept)
    api = HfApi()
    if api.repo_exists(repo) and version in {t.name for t in api.list_repo_refs(repo).tags}:
        raise SystemExit(f"{repo} already has a tag {version}: releases are immutable")
    publish(run, repo, private=private, logs=logs)
    revision = str(api.model_info(repo).sha)
    api.create_tag(repo, tag=version, revision=revision, tag_message=f"den release {version}")
    note = f"{accept} (dev drops: {drops})" if drops else None
    entry = record(version, run, repo, revision, parent, data_repo, note, root)
    root.mkdir(exist_ok=True)
    (root / f"{version}.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return entry


def headline(entry: dict[str, Any], split: str) -> str:
    results = [m["accuracy"] for m in entry[split].values() if "accuracy" in m]
    return f"{sum(results) / len(results):.4f} ({len(results)} files)" if results else "-"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den release")
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("create", help="check, publish, tag and record a version")
    c.add_argument("--version", required=True, help="v1, v2, v2.1, ...")
    c.add_argument("--run", required=True, type=Path, help="the finished run: merged, integrity-checked, tested once")
    c.add_argument("--repo", required=True, help="<org>/<name> on the Hub; every version is a tag in it")
    c.add_argument("--parent", help="the version this one continues from (its regression gate)")
    c.add_argument("--data-repo", help="<org>/<name>@<commit> of the `make data-upload` copy it was trained from")
    c.add_argument("--accept-regression", dest="accept", help="release despite dev drops above 1 point, and why")
    c.add_argument("--logs", nargs="*", type=Path, default=[])
    c.add_argument("--public", action="store_true")
    c.add_argument("--dry-run", action="store_true", help="run every check; publish and record nothing")
    sub.add_parser("list", help="every version, its parent and headline numbers")
    s = sub.add_parser("show", help="one release record")
    s.add_argument("version")
    args = p.parse_args(argv)
    if args.action == "create":
        if args.dry_run:
            drops = check(args.version, args.run, args.parent, accept=args.accept)
            print(f"{args.version}: ready to release" + (f" (accepted dev drops: {drops})" if drops else ""))
            return 0
        entry = create(
            args.version, args.run, args.repo, args.parent, args.data_repo, not args.public, args.accept, args.logs
        )
        print(f"released {args.version}: hf:{args.repo}@{args.version} (commit {entry['hub']['revision']})")
        print(f"recorded {RELEASES / (args.version + '.json')}: commit it")
        return 0
    if args.action == "show":
        print(json.dumps(load(args.version), indent=2))
        return 0
    versions = sorted(
        (_json(path) for path in RELEASES.glob("*.json")), key=lambda r: [int(x) for x in r["version"][1:].split(".")]
    )
    print(f"{'version':<9}{'parent':<9}{'created':<22}{'dev accuracy':<22}{'test accuracy':<22}model")
    for r in versions:
        dev, test = headline(r, "dev"), headline(r, "test")
        print(
            f"{r['version']:<9}{r['parent'] or '-':<9}{r['created']:<22}{dev:<22}{test:<22}"
            f"hf:{r['hub']['repo']}@{r['version']}"
        )
    return 0
