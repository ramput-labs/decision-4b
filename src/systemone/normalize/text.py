from __future__ import annotations

import html
import json
import re
import unicodedata
from collections.abc import Mapping

from ..records import fingerprint

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BARE_ENTITY = re.compile(r"(?<!&)(#\d+);|(?<![&\w])(quot|amp|lt|gt);")
_SPACES = re.compile("[ \t\u00a0]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_KEV_TEXT_FIELDS = ("text", "premise", "passage", "content", "question", "sentence")


def clean(text: str, *, escapes: bool = False, markup: bool = False) -> str:
    """`escapes` undoes literal \\n and \\" (Yelp, AG News); `markup` also repairs AG News' '&'-less entities."""
    if escapes:
        text = text.replace("\\n", "\n").replace('\\"', '"').replace("\\", " ")
    if markup:
        text = html.unescape(_BARE_ENTITY.sub(lambda m: f"&{m[1] or m[2]};", _BREAK.sub("\n", text)))
    text = _CONTROL.sub("", unicodedata.normalize("NFC", text))
    lines = (_SPACES.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def provenance(row: Mapping[str, object]) -> str:
    """Kev's `_meta.text_sha256`: normalized exactly as Kev's builders do, so records match across suites."""
    text = next((v for k in _KEV_TEXT_FIELDS if isinstance(v := row.get(k), str)), None)
    if text is None:
        text = json.dumps(row, sort_keys=True, default=str)
    return fingerprint(text)
