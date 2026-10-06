"""Every number the docs print traces to committed evidence (like Kev's scripts/verify_claims.py).

`reports/claims.json` lists claims: {"printed": the exact string in the docs, "in": files it must appear in,
"source": a JSON file under the repo (usually reports/runs/<run>/<file>/report.json or releases/<v>.json),
"path": the keys to walk to the number, and optionally "scale" (100 for a percentage)}. A claim passes when the
string is in every file and the source value, scaled and rounded to the printed precision, gives the printed digits.
`make check` runs it, so a number in a README or model card can't drift from what was measured.

    uv run python -m scripts.verify_claims
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CLAIMS = Path("reports/claims.json")


def value(source: Path, path: list[Any]) -> Any:  # noqa: ANN401 (a JSON value)
    node: Any = json.loads(source.read_text(encoding="utf-8"))
    for key in path:
        node = node[key]
    return node


def problems(claim: dict[str, Any], root: Path = Path(".")) -> list[str]:
    printed, where = str(claim["printed"]), f"{claim['printed']!r} from {claim['source']}"
    found = []
    for doc in claim["in"]:
        if printed not in (root / doc).read_text(encoding="utf-8"):
            found.append(f"{where}: not printed in {doc}")
    try:
        number = float(value(root / claim["source"], claim["path"])) * float(claim.get("scale", 1))
    except (OSError, KeyError, IndexError, TypeError, ValueError) as e:
        return [*found, f"{where}: can't read {claim['path']}: {e!r}"]
    digits = re.search(r"-?\d+(?:\.(\d+))?", printed)
    if digits is None:
        return [*found, f"{where}: no number in the printed string"]
    places = len(digits.group(1) or "")
    if f"{number:.{places}f}" != digits.group(0):
        found.append(f"{where}: the source says {number:.{places}f}, the docs print {digits.group(0)}")
    return found


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.verify_claims")
    p.add_argument("--claims", type=Path, default=CLAIMS)
    args = p.parse_args(argv)
    claims = json.loads(args.claims.read_text(encoding="utf-8"))
    found = [problems(claim) for claim in claims]
    for problem in (p for ps in found for p in ps):
        print(f"BAD {problem}")
    print(f"claims: {sum(not ps for ps in found)} of {len(claims)} trace to their evidence")
    return 1 if any(found) else 0


if __name__ == "__main__":
    raise SystemExit(main())
