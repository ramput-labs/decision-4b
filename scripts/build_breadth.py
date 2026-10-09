"""breadth-v1, Kev's eval-only panel of 14 public datasets, rebuilt from our pinned raw sources.

Kev keeps breadth-v1's record files private (several sources forbid redistribution), but publishes the sha256 of both
partitions and the script that builds them (scripts/build_breadth_v1.py at KEV_COMMIT). This is that script ported: the
same mappings, seeds, selection and admission check, reading `data/sources/eval/breadth/` (fetched and sha256-checked by
`den data raw-eval`, which `make data-download` runs) instead of downloading. It writes `data/dev/breadth.jsonl` and
`data/test/breadth.jsonl` only when both are byte-identical to Kev's, so our breadth numbers sit on exactly the panel
Kev-4B was scored on.

Never trained on: the files sit beside Kev's suites, so `normalize` keeps their texts out of every trainable source.
Build it before `make data-normalize`. Keep the logic byte-faithful to Kev's: every seed, sort and key order shows in
the sha256.

    uv run python -m scripts.build_breadth        # ~10 minutes (BM25 over BRIGHT); `make data-breadth`
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import random
import re
import struct
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from den.api import option_text, render
from den.pins import BREADTH_SHA256 as EXPECTED

Rec = dict[str, Any]
RAW = Path("data/sources/eval/breadth")
VERSION = SEED = "breadth-v1"
PARTITIONS = ("development", "test")
OUT = {"development": Path("data/dev/breadth.jsonl"), "test": Path("data/test/breadth.jsonl")}
EVAL_RECORDS = 150
CONTRACTNLI_DOCS, CONTRACTNLI_PER_DOC = 40, 4
MAX_OPTIONS = 10
# Kev's admission: the 8k serving context with 1,024 tokens of headroom on the state and 512 on the row
ADMISSION = {"max_state": 8192 - 1024, "max_branch": 8192 - 512, "max_packed": 8192 + 8192}
TOKENIZER = Path("models/qwen3.5-4b/tokenizer.json")  # Qwen/Qwen3.5-4B-Base@1001bb4, Kev's ADMISSION_TOKENIZER

AREAS = {
    "musr": "knowledge",
    "sata_bench": "knowledge",
    "chessbench": "knowledge",
    "contractnli": "language",
    "hellaswag": "language",
    "clinc150": "retrieval",
    "sgd": "retrieval",
    "bright": "retrieval",
    "bfcl": "tools",
    "toolret": "tools",
    "apibank": "tools",
    "routerbench": "tools",
    "humicroedit": "arts",
    "cfcolor": "arts",
}

MUSR = (
    "TAUR-Lab/MuSR",
    "7c365b439a222150f317764d4f16ae6c96d7d94a",
    ("murder_mystery.csv", "object_placements.csv", "team_allocation.csv"),
)
SATA = ("sata-bench/sata-bench", "ba43a7ab537adfa3498e3a160a6d1eafbefc95c1", "data_main.json")
CHESS_URL = "https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag"
CONTRACTNLI_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
HELLASWAG = ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "data/validation-00000-of-00001.parquet")
CLINC = (
    "clinc/clinc_oos",
    "155b9c710419136e17307b80d0a13e68cd46b4ec",
    {"development": "plus/validation-00000-of-00001.parquet", "test": "plus/test-00000-of-00001.parquet"},
)
SGD_COMMIT = "e852981ae34990f4358979625854259302feaa78"
BRIGHT = ("xlangai/BRIGHT", "3066d29c9651a576c8aba4832d249807b181ecae")
BRIGHT_DOMAINS = (
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
)
BFCL = ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda")
BFCL_CATEGORIES = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
)
TOOLRET = (
    "mangopy/ToolRet-Queries",
    "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445",
    "mangopy/ToolRet-Tools",
    "e06c38c75612b6536bd959e08cdd345894aba6a7",
)
TOOLRET_EXCLUDED = ("apibank", "apigen")
APIBANK = ("liminghao1630/API-Bank", "12e8158b7628c168f07e8f31fbbe3445e99f44cf", "test-data/level-1-api.json")
ROUTERBENCH = ("withmartian/routerbench", "784021482c3f320c6619ed4b3bb3b41a21424fcb", "routerbench_0shot.pkl")
ROUTER_MODELS = (
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
)
HUMICROEDIT_URL = "https://cs.rochester.edu/u/nhossain/semeval-2020-task-7-dataset.zip"
CFCOLOR_URL = "https://www.dgp.toronto.edu/~donovan/cfcolor/cfcolor.zip"
SPLITLINES = re.compile("[\u0085\u2028\u2029]")
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

# ---------------------------------------------------------------- Kev's shared helpers (kev.suite, build_devtools_v1)


def h(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def seeded(seed: str) -> random.Random:
    return random.Random(h(SEED, seed))


def normalise_text(text: str) -> str:
    return " ".join(text.casefold().split())


def text_digest(text: str) -> str:
    return hashlib.sha256(normalise_text(text).encode()).hexdigest()


def state_key(state: Any) -> str:  # noqa: ANN401 (a str or a JSON object)
    return text_digest(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True))


def line_safe(record: Rec) -> bool:
    """No character that str.splitlines() would break a JSONL line on."""
    return not SPLITLINES.search(
        json.dumps({"state": record["state"], "questions": record["questions"]}, ensure_ascii=False)
    )


def round_robin(classes: dict[str, list[Any]], n: int, admit: Callable[[Any], bool]) -> list[Any]:
    """Up to n items round-robin over `classes` in sorted label order, skipping what `admit` rejects."""
    order = sorted(classes)
    cursors = dict.fromkeys(order, 0)
    out: list[Any] = []
    live = list(order)
    while live and len(out) < n:
        for c in list(live):
            if len(out) >= n:
                break
            items = classes[c]
            while cursors[c] < len(items) and not admit(items[cursors[c]]):
                cursors[c] += 1
            if cursors[c] >= len(items):
                live.remove(c)
                continue
            out.append(items[cursors[c]])
            cursors[c] += 1
    return out


def deal_groups(weights: dict[str, int], shares: dict[str, float], seed: str) -> dict[str, str]:
    """{group: split}: groups in sha256(seed:group) order, each to the split least filled for its share."""
    filled = dict.fromkeys(shares, 0.0)
    out = {}
    for g in sorted(weights, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()):
        s = min((s for s in shares if shares[s] > 0), key=lambda s: (filled[s] / shares[s], list(shares).index(s)))
        out[g] = s
        filled[s] += weights[g]
    return out


def read_jsonl(path: Path) -> list[Rec]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def parquet_rows(path: Path) -> list[Rec]:
    import pyarrow.parquet as pq

    rows: list[Rec] = pq.read_table(path).to_pylist()
    return rows


def user_tokens(tokenizer: Tokenizer, text: str) -> list[int]:
    """Kev's `user_tokens`: text can't spell a control token (`<|x|>` is read as `<¦x¦>`)."""
    ids: list[int] = tokenizer.encode(_SPECIAL.sub(r"<¦\1¦>", text), add_special_tokens=False).ids
    return ids


def fits(record: Rec, tokenizer: Tokenizer, max_state: int, max_branch: int, max_packed: int) -> bool:
    """Kev's `kev.model.fits(materialize(record))`: the record encodes, untruncated, in Kev's packed layout:
    `<state> state`, then per question `<q> instructions (<opt> option </opt>)... <decide>`."""
    state = 1 + len(user_tokens(tokenizer, render(record["state"])))
    if state > max_state:
        return False
    total = state
    for q in record["questions"].values():
        if q["type"] == "noul":
            options = [option_text("no", None), option_text("yes", None)]
        elif q["type"] == "choice":
            options = [option_text(k, v) for k, v in q["criteria"].items()]
        else:
            options = [render(x) for x in q["criteria"]]
        branch = 1 + len(user_tokens(tokenizer, render(q["instructions"])))
        branch += sum(2 + len(user_tokens(tokenizer, o)) for o in options) + 1
        if branch > max_branch - state:
            return False
        total += branch
    return total <= max_packed


# ---------------------------------------------------------------- builder helpers (pure)


def q_choice(instr: str, criteria: dict[str, Any], label: str, src: str) -> Rec:
    if label not in criteria:
        raise ValueError(f"label {label!r} not among options ({src})")
    if not 2 <= len(criteria) <= MAX_OPTIONS:
        raise ValueError(f"{src}: {len(criteria)} options")
    return {"type": "choice", "instructions": instr, "criteria": dict(criteria), "label": label, "src": src}


def q_noul(instr: str, label: object, src: str) -> Rec:
    return {"type": "noul", "instructions": instr, "label": bool(label), "src": src}


def candidate(
    dataset: str,
    cid: str,
    state: object,
    questions: Rec,
    group: str,
    stratum: str,
    split: str | None = None,
    **meta: object,
) -> Rec:
    x: Rec = {
        "state": state,
        "questions": questions,
        "_stratum": stratum,
        "_meta": {
            "id": f"{dataset}/{cid}",
            "source": dataset,
            "area": AREAS[dataset],
            "group_id": f"{dataset}/{group}",
            "text_sha256": state_key(state),
            **meta,
        },
    }
    if split:
        x["_split"] = split
    return x


def split_halves(cands: list[Rec], dataset: str) -> list[Rec]:
    weights = Counter(x["_meta"]["group_id"] for x in cands)
    where = deal_groups(dict(weights), {"test": 0.5, "development": 0.5}, f"{SEED}:{dataset}")
    for x in cands:
        x["_split"] = where[x["_meta"]["group_id"]]
    return cands


def clean_html(text: str | None) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())


def truncate_words(text: str | None, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " …"


class BM25:
    """Okapi BM25 (k1 1.2, b 0.75) over scikit-learn's word tokenizer, English stop words removed. `top` returns up to
    k indices by descending score (rounded to 1e-6), ties by index."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer  # type: ignore[import-untyped]

        self.vec = CountVectorizer(
            lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w\w+\b", dtype=np.float64
        )
        tf = self.vec.fit_transform(texts).tocsr()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        length = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * length / max(length.mean(), 1e-9))
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.weights = tf

    def top(self, query: str, k: int, skip: Callable[[int], bool] = lambda i: False) -> list[int]:
        import numpy as np

        q = self.vec.transform([query])
        q.data[:] = 1.0
        scores = np.round(np.asarray((self.weights @ q.T).todense()).ravel(), 6)
        order = np.lexsort((np.arange(len(scores)), -scores))
        out: list[int] = []
        for i in order:
            if scores[i] <= 0:
                break
            if not skip(int(i)):
                out.append(int(i))
            if len(out) == k:
                break
        return out


# ---------------------------------------------------------------- per-dataset mappings (pure)


def musr_candidate(subset: str, index: int, row: Rec) -> Rec:
    choices = ast.literal_eval(row["choices"])
    keys = [f"opt_{i + 1}" for i in range(len(choices))]
    q = q_choice(
        row["question"].strip(), dict(zip(keys, choices, strict=True)), keys[int(row["answer_index"])], f"musr_{subset}"
    )
    return candidate(
        "musr",
        f"{subset}/{index}",
        row["narrative"].strip(),
        {"answer": q},
        f"{subset}/{index}",
        subset,
        provenance={"via": f"hf:{MUSR[0]}@{MUSR[1]}:{subset}.csv", "row": index},
    )


def sata_candidate(index: int, row: Rec, rng: random.Random) -> Rec | None:
    answers, distractors = (
        [str(x).strip() for x in (v if isinstance(v, list) else ast.literal_eval(v))]
        for v in (row["answer groups"], row["distractor groups"])
    )
    if not answers or len({normalise_text(x) for x in answers + distractors}) != len(answers) + len(distractors):
        return None
    if not 2 <= len(answers) + len(distractors) <= MAX_OPTIONS:
        return None
    options = [(x, True) for x in answers] + [(x, False) for x in distractors]
    rng.shuffle(options)
    question = re.sub(r"^Multi Label Question:\s*", "", clean_html(row["question"]))
    passage = re.sub(r"^Paragraph:\s*", "", clean_html(row["paragraph"]))
    qs = {
        f"cand_{i + 1}": q_noul(f'Is "{text}" a correct answer to the question?', ok, "sata_bench")
        for i, (text, ok) in enumerate(options)
    }
    state = {"passage": passage, "question": question}
    return candidate(
        "sata_bench",
        str(index),
        state,
        qs,
        text_digest(question + passage)[:20],
        f"answers_{min(len(answers), 3)}",
        provenance={"via": f"hf:{SATA[0]}@{SATA[1]}:{SATA[2]}", "row": index},
    )


PIECES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}


def fen_board(fen: str) -> dict[str, str]:
    board = {}
    for r, rank in enumerate(fen.split()[0].split("/")):
        f = 0
        for c in rank:
            if c.isdigit():
                f += int(c)
            else:
                board[f"{'abcdefgh'[f]}{8 - r}"] = c
                f += 1
    return board


def ascii_board(fen: str) -> str:
    board = fen_board(fen)
    ranks = (f"{rank} " + " ".join(board.get(f"{f}{rank}", ".") for f in "abcdefgh") for rank in range(8, 0, -1))
    return "\n".join(ranks) + "\n  a b c d e f g h"


def describe_move(fen: str, uci: str) -> str:
    board = fen_board(fen)
    src, dst, promo = uci[:2], uci[2:4], uci[4:]
    piece = board.get(src, "?")
    colour = "white" if piece.isupper() else "black"
    text = f"{colour} {PIECES.get(piece.lower(), 'piece')} {src}-{dst}"
    if piece.lower() == "k" and src[0] == "e" and dst[0] in "cg" and src[1] == dst[1]:
        text += " (castling)"
    elif dst in board:
        text += f" capturing the {PIECES.get(board[dst].lower(), 'piece')}"
    if promo:
        text += f", promoting to {PIECES[promo.lower()]}"
    return text


def chess_options(
    moves: dict[str, float], rng: random.Random, tie: float = 0.01, limit: int = MAX_OPTIONS
) -> tuple[list[str], str] | None:
    best = max(sorted(moves), key=lambda m: moves[m])
    others = sorted(m for m in moves if m != best and moves[m] <= moves[best] - tie)
    if not others:
        return None
    rng.shuffle(others)
    keys = [best, *others[: limit - 1]]
    rng.shuffle(keys)
    return keys, best


def chess_phase(fen: str) -> str:
    n = sum(1 for c in fen.split()[0] if c.isalpha())
    return "opening" if n >= 26 else "middlegame" if n >= 13 else "endgame"


def chess_candidate(fen: str, moves: dict[str, float], rng: random.Random) -> Rec | None:
    picked = chess_options(moves, rng)
    if picked is None:
        return None
    keys, best = picked
    state = {"fen": fen, "to_move": "white" if fen.split()[1] == "w" else "black", "board": ascii_board(fen)}
    q = q_choice(
        "Which move is strongest for the side to move?", {m: describe_move(fen, m) for m in keys}, best, "chessbench"
    )
    return candidate(
        "chessbench",
        h(fen)[:16],
        state,
        {"best_move": q},
        h(fen)[:16],
        chess_phase(fen),
        values={m: round(moves[m], 6) for m in keys},
        provenance={"via": CHESS_URL, "fen": fen},
    )


def read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value, i


def decode_action_value(record: bytes) -> tuple[str, str, float]:
    """(fen, move, win probability) of one searchless_chess action-value record (apache_beam TupleCoder)."""
    n, i = read_varint(record, 0)
    fen = record[i : i + n].decode()
    i += n
    n, i = read_varint(record, i)
    move = record[i : i + n].decode()
    i += n
    value: float = struct.unpack(">d", record[i : i + 8])[0]
    return fen, move, value


def bag_records(path: Path) -> Iterator[bytes]:
    """Records of an uncompressed bagz file: the records, then one little-endian int64 end offset per record."""
    data = path.read_bytes()
    (start,) = struct.unpack("<Q", data[-8:])
    ends = struct.unpack(f"<{(len(data) - start) // 8}q", data[start:])
    prev = 0
    for end in ends:
        yield data[prev:end]
        prev = end


CNLI_LABELS = {"Entailment": "entailment", "Contradiction": "contradiction", "NotMentioned": "not_mentioned"}
CNLI_CRITERIA = {
    "entailment": "The contract entails the statement",
    "contradiction": "The contract contradicts the statement",
    "not_mentioned": "The contract does not mention it",
}


def contractnli_questions(
    annotations: Rec, hypotheses: dict[str, str], rng: random.Random, k: int = CONTRACTNLI_PER_DOC
) -> Rec:
    classes: dict[str, list[str]] = defaultdict(list)
    for key in sorted(annotations):
        classes[CNLI_LABELS[annotations[key]["choice"]]].append(key)
    for v in classes.values():
        rng.shuffle(v)
    chosen = round_robin(classes, k, lambda _: True)
    return {
        key: q_choice(
            f'Statement: "{hypotheses[key]}" Does the contract entail, contradict or not mention this statement?',
            CNLI_CRITERIA,
            CNLI_LABELS[annotations[key]["choice"]],
            "contractnli",
        )
        for key in sorted(chosen)
    }


def hellaswag_candidate(index: int, row: Rec) -> Rec:
    keys = [f"opt_{i + 1}" for i in range(len(row["endings"]))]
    q = q_choice(
        "Which ending most plausibly continues the scene?",
        dict(zip(keys, row["endings"], strict=True)),
        keys[int(row["label"])],
        "hellaswag",
    )
    return candidate(
        "hellaswag",
        f"val/{row['ind']}",
        {"activity": row["activity_label"], "context": row["ctx"]},
        {"ending": q},
        row["source_id"],
        row["split_type"],
        provenance={
            "via": f"hf:{HELLASWAG[0]}@{HELLASWAG[1]}:{HELLASWAG[2]}",
            "row": index,
            "source_id": row["source_id"],
        },
    )


OOS = "out_of_scope"


def clinc_options(gold: str, domains: dict[str, list[str]], rng: random.Random) -> tuple[Rec, str]:
    domain_of = {i: d for d, intents in domains.items() for i in intents}
    if gold == "oos":
        intents = rng.sample(sorted(domains[rng.choice(sorted(domains))]), 9)
    else:
        intents = [*rng.sample(sorted(i for i in domains[domain_of[gold]] if i != gold), 8), gold]
    rng.shuffle(intents)
    criteria = {**dict.fromkeys(intents), OOS: "The request fits none of the listed intents"}
    return criteria, OOS if gold == "oos" else gold


def clinc_candidate(
    split: str, index: int, text: str, gold: str, domains: dict[str, list[str]], rng: random.Random
) -> Rec:
    criteria, label = clinc_options(gold, domains, rng)
    domain = "oos" if gold == "oos" else next(d for d, v in domains.items() if gold in v)
    q = q_choice("Which intent does this request express?", criteria, label, "clinc150")
    return candidate(
        "clinc150",
        f"{split}/{index}",
        text,
        {"intent": q},
        f"{split}/{index}",
        domain,
        split=split,
        provenance={"via": f"hf:{CLINC[0]}@{CLINC[1]}:{CLINC[2][split]}", "row": index},
    )


SGD_VARIANTS = ("original", "v1", "v2", "v3", "v4", "v5")
SGD_NONE = "NONE"


def sgd_variant_services(original: list[Rec], variant: list[Rec]) -> dict[str, Rec]:
    if len(original) != len(variant):
        raise ValueError("SGD-X variant lists a different number of services")
    out = {}
    for o, v in zip(original, variant, strict=True):
        if not v["service_name"].startswith(o["service_name"]) or len(o["intents"]) != len(v["intents"]):
            raise ValueError(f"SGD-X variant does not line up at {o['service_name']}")
        out[o["service_name"]] = v
    return out


def sgd_turns(dialogue: Rec) -> list[tuple[int, str, str, str, list[str]]]:
    history: list[str] = []
    previous: dict[str, str] = {}
    out = []
    for t, turn in enumerate(dialogue["turns"]):
        history.append(f"{'User' if turn['speaker'] == 'USER' else 'System'}: {turn['utterance']}")
        if turn["speaker"] != "USER":
            continue
        for frame in turn["frames"]:
            intent, service = frame["state"]["active_intent"], frame["service"]
            if len(turn["frames"]) == 1:
                stratum = "none" if intent == SGD_NONE else "changed" if previous.get(service) != intent else "same"
                out.append((t, service, intent, stratum, list(history)))
            previous[service] = intent
    return out


def sgd_candidate(
    split: str, dialogue: Rec, turn: tuple[int, str, str, str, list[str]], schemas: dict[str, dict[str, Rec]]
) -> Rec:
    t, service, intent, stratum, history = turn
    variant = SGD_VARIANTS[int(h(dialogue["dialogue_id"]), 16) % len(SGD_VARIANTS)]
    original = schemas["original"][service]
    shown = schemas[variant][service]
    names = {o["name"]: v["name"] for o, v in zip(original["intents"], shown["intents"], strict=True)}
    criteria = {v["name"]: v["description"] for v in shown["intents"]}
    if SGD_NONE in criteria:
        raise ValueError("intent named NONE")
    criteria[SGD_NONE] = "The user is not pursuing any of these intents right now"
    q = q_choice(
        "Which of this service's intents is the user pursuing right now?",
        criteria,
        SGD_NONE if intent == SGD_NONE else names[intent],
        "sgd",
    )
    state = {"service": shown["description"], "dialogue": history}
    return candidate(
        "sgd",
        f"{split}/{dialogue['dialogue_id']}/{t}",
        state,
        {"intent": q},
        f"{split}/{dialogue['dialogue_id']}",
        stratum,
        split=split,
        schema_variant=variant,
        provenance={
            "via": f"github:google-research-datasets/dstc8-schema-guided-dialogue@{SGD_COMMIT}:{split}",
            "dialogue_id": dialogue["dialogue_id"],
            "turn": t,
        },
    )


def bfcl_candidate(category: str, item: Rec, answers: list[Rec] | None) -> Rec | None:
    functions = item["function"]
    if not 1 <= len(functions) <= MAX_OPTIONS:
        return None
    names = [f["name"] for f in functions]
    if len(set(names)) != len(names):
        return None
    called = {name for call in answers for name in call} if answers is not None else set()
    if not called <= set(names):
        return None
    conversation = [f"{m['role']}: {m['content']}" for m in item["question"][0]]
    state = {
        "conversation": conversation,
        "available_functions": [
            {
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": json.dumps(f.get("parameters", {}), ensure_ascii=False, sort_keys=True),
            }
            for f in functions
        ],
    }
    qs = {
        f"call_{i + 1}": q_noul(
            f"Should the assistant call the function `{name}` for this request?", name in called, f"bfcl_{category}"
        )
        for i, name in enumerate(names)
    }
    return candidate(
        "bfcl",
        item["id"],
        state,
        qs,
        item["id"],
        category,
        provenance={"via": f"hf:{BFCL[0]}@{BFCL[1]}:BFCL_v3_{category}.json", "id": item["id"]},
    )


def tool_summary(documentation: object, limit: int = 400) -> tuple[str, str]:
    try:
        doc = json.loads(documentation)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        doc = {"description": str(documentation)}
    if not isinstance(doc, dict):
        doc = {"description": json.dumps(doc, ensure_ascii=False)}
    name = str(doc.get("name") or doc.get("api_name") or doc.get("tool_name") or doc.get("api_call") or "tool").strip()
    parts = [str(doc[k]) for k in ("functionality", "description", "api_description", "tool_description") if doc.get(k)]
    return name, truncate_words(" ".join(parts) or json.dumps(doc, ensure_ascii=False, sort_keys=True), limit)


def retrieval_choice(
    instr: str, gold_text: str, negative_texts: list[str], rng: random.Random, src: str, prefix: str
) -> Rec:
    texts = [(gold_text, True)] + [(t, False) for t in negative_texts]
    rng.shuffle(texts)
    keys = [f"{prefix}_{i + 1}" for i in range(len(texts))]
    criteria = dict(zip(keys, [t for t, _ in texts], strict=True))
    label = keys[[ok for _, ok in texts].index(True)]
    return q_choice(instr, criteria, label, src)


API_REQUEST = re.compile(r"API-Request:\s*\[(\w+)\(")


def apibank_parse(row: Rec) -> tuple[str, dict[str, str], str | None]:
    described = {}
    tail = row["instruction"].split("API descriptions:", 1)[1]
    for line in tail.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            api = json.loads(line)
            described[api["name"]] = api["description"]
    dialogue = re.sub(r"\s*Generate API Request:\s*$", "", row["input"].strip())
    m = API_REQUEST.search(row["expected_output"])
    return dialogue, described, m.group(1) if m else None


def router_options(
    scores: dict[str, int], rng: random.Random, limit: int = MAX_OPTIONS
) -> tuple[list[str], str] | None:
    right = sorted(m for m, s in scores.items() if s == 1)
    wrong = sorted(m for m, s in scores.items() if s == 0)
    if not right or not wrong:
        return None
    label = rng.choice(right)
    rng.shuffle(wrong)
    keys = [label, *wrong[: limit - 1]]
    rng.shuffle(keys)
    return keys, label


def router_family(eval_name: str) -> str:
    return "mmlu" if eval_name.startswith("mmlu-") else eval_name


def humicroedit_pair(row: Rec) -> tuple[str, str, str, str] | None:
    if row["label"] not in ("1", "2"):
        return None
    tag = re.compile(r"<([^>]*)/>")
    original = tag.sub(lambda m: m.group(1), row["original1"])
    one = tag.sub(row["edit1"], row["original1"], count=1)
    two = tag.sub(row["edit2"], row["original2"], count=1)
    if normalise_text(one) == normalise_text(two):
        return None
    return " ".join(original.split()), " ".join(one.split()), " ".join(two.split()), "a" if row["label"] == "1" else "b"


def palette_hex(rgb15: Sequence[float]) -> str:
    return " ".join("#" + "".join(f"{round(float(c) * 255):02x}" for c in rgb15[i : i + 3]) for i in range(0, 15, 3))


def cfcolor_record(
    user: int,
    train: list[tuple[int, int]],
    test: list[tuple[int, int]],
    rgb: dict[int, list[float]],
    rng: random.Random,
    history: int = 12,
) -> tuple[Rec, Rec, Rec] | None:
    test = sorted(test, key=lambda x: h(user, x[0]))
    pair = next(((a, b) for i, a in enumerate(test) for b in test[i + 1 :] if a[1] != b[1]), None)
    if pair is None:
        return None
    shown = [x for x in sorted(train, key=lambda x: h(user, "train", x[0])) if x[0] not in (pair[0][0], pair[1][0])]
    shown = shown[:history]
    if len(shown) < 8:
        return None
    a, b = pair if rng.random() < 0.5 else pair[::-1]
    state = {
        "note": "Ratings this person gave to five-colour palettes, from 1 (dislike) to 5 (like).",
        "earlier_ratings": [{"palette": palette_hex(rgb[t]), "rating": int(r)} for t, r in shown],
    }
    q = q_choice(
        "Which of these two palettes did this person rate higher?",
        {"a": palette_hex(rgb[a[0]]), "b": palette_hex(rgb[b[0]])},
        "a" if a[1] > b[1] else "b",
        "cfcolor",
    )
    return state, q, {"themes": [int(a[0]), int(b[0])], "ratings": [int(a[1]), int(b[1])]}


# ---------------------------------------------------------------- per-dataset candidates (pinned raw files)


def musr(raw: Path, report: Rec) -> list[Rec]:
    out = []
    csv.field_size_limit(sys.maxsize)
    for fname in MUSR[2]:
        text = (raw / "knowledge/musr" / fname).read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
        out += [musr_candidate(fname[:-4], i, r) for i, r in enumerate(rows)]
    report["musr"] = {"candidates": len(out)}
    return split_halves(out, "musr")


def sata(raw: Path, report: Rec) -> list[Rec]:
    rows = json.loads((raw / "knowledge/sata-bench" / SATA[2]).read_text(encoding="utf-8"))
    out, dropped = [], 0
    for i, r in enumerate(rows):
        x = sata_candidate(i, r, seeded(f"sata/{i}"))
        if x is None:
            dropped += 1
        else:
            out.append(x)
    report["sata_bench"] = {"rows": len(rows), "dropped_options": dropped, "candidates": len(out)}
    return split_halves(out, "sata_bench")


def chessbench(raw: Path, report: Rec) -> list[Rec]:
    positions: dict[str, dict[str, float]] = defaultdict(dict)
    for rec in bag_records(raw / "knowledge/chessbench/action_value_data.bag"):
        fen, move, p = decode_action_value(rec)
        positions[fen][move] = p
    fens = sorted(positions)
    pool = sorted(fens, key=lambda f: h(SEED, "chess", f))[:4000]  # a seeded 4,000-position sample
    out = [x for x in (chess_candidate(f, positions[f], seeded(f"chess/{f}")) for f in pool) if x is not None]
    report["chessbench"] = {"positions": len(fens), "pool": len(pool), "candidates": len(out)}
    return split_halves(out, "chessbench")


def contractnli(raw: Path, report: Rec) -> list[Rec]:
    out = []
    with zipfile.ZipFile(raw / "language/contractnli/contract-nli.zip") as z:
        for split, member in (("development", "contract-nli/dev.json"), ("test", "contract-nli/test.json")):
            data = json.loads(z.read(member))
            hyps = {k: v["hypothesis"] for k, v in data["labels"].items()}
            for doc in data["documents"]:
                qs = contractnli_questions(doc["annotation_sets"][0]["annotations"], hyps, seeded(f"cnli/{doc['id']}"))
                out.append(
                    candidate(
                        "contractnli",
                        f"{split}/{doc['id']}",
                        doc["text"].strip(),
                        qs,
                        f"doc/{doc['id']}",
                        doc["document_type"],
                        split=split,
                        provenance={
                            "via": f"{CONTRACTNLI_URL}:{member}",
                            "document_id": doc["id"],
                            "file_name": doc["file_name"],
                        },
                    )
                )
    report["contractnli"] = {"candidates": len(out)}
    return out


def hellaswag(raw: Path, report: Rec) -> list[Rec]:
    rows = parquet_rows(raw / "language/hellaswag" / HELLASWAG[2])
    out = [hellaswag_candidate(i, r) for i, r in enumerate(rows) if r["source_id"].startswith("activitynet~")]
    report["hellaswag"] = {"rows": len(rows), "activitynet": len(out), "wikihow_left_out": len(rows) - len(out)}
    return split_halves(out, "hellaswag")


def clinc150(raw: Path, report: Rec) -> list[Rec]:
    base = raw / "retrieval/clinc150"
    domains = json.loads((base / "domains.json").read_text(encoding="utf-8"))
    card = (base / "README.md").read_text(encoding="utf-8")
    block = card[card.index("config_name: plus") :]
    block = block[: block.index("splits:")]
    names = {int(k): v.strip("'\"") for k, v in re.findall(r"'(\d+)': (\S+)", block)}  # YAML quotes 'yes' / 'no'
    if sorted({i for v in domains.values() for i in v}) != sorted(n for n in names.values() if n != "oos"):
        raise ValueError("CLINC150 domains.json and the card's intents differ")
    out = []
    for split, fname in CLINC[2].items():
        for i, r in enumerate(parquet_rows(base / fname)):
            out.append(clinc_candidate(split, i, r["text"], names[r["intent"]], domains, seeded(f"clinc/{split}/{i}")))
    report["clinc150"] = {"candidates": len(out)}
    return out


def sgd(raw: Path, report: Rec) -> list[Rec]:
    files = {}
    with tarfile.open(raw / "retrieval/sgd/dstc8-schema-guided-dialogue.tar.gz") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".json") and ("/dev/" in m.name or "/test/" in m.name):
                stream = t.extractfile(m)
                assert stream is not None
                files[m.name.split("/", 1)[1]] = json.loads(stream.read())
    out: list[Rec] = []
    stats: Counter[tuple[str, str]] = Counter()
    for split, folder in (("development", "dev"), ("test", "test")):
        original = files[f"{folder}/schema.json"]
        schemas = {"original": {s["service_name"]: s for s in original}}
        for v in SGD_VARIANTS[1:]:
            schemas[v] = sgd_variant_services(original, files[f"sgd_x/data/{v}/{folder}/schema.json"])
        for name in sorted(f for f in files if f.startswith(f"{folder}/dialogues_")):
            for dialogue in files[name]:
                for turn in sgd_turns(dialogue):
                    out.append(sgd_candidate(split, dialogue, turn, schemas))
                    stats[split, turn[3]] += 1
    report["sgd"] = {"candidates": len(out)}
    return out


def bright(raw: Path, report: Rec) -> list[Rec]:
    out: list[Rec] = []
    base = raw / "retrieval/bright"
    for domain in BRIGHT_DOMAINS:
        docs = parquet_rows(base / f"documents/{domain}-00000-of-00001.parquet")
        ids = [d["id"] for d in docs]
        texts = [d["content"] for d in docs]
        index = {d: i for i, d in enumerate(ids)}
        ok_len = [150 <= len(t.strip()) <= 1500 for t in texts]
        bm25 = BM25(texts)
        for ex in parquet_rows(base / f"examples/{domain}-00000-of-00001.parquet"):
            gold = [g for g in sorted(ex["gold_ids"]) if g in index and ok_len[index[g]]]
            if not gold:
                continue
            banned = set(ex["gold_ids"]) | set(ex["excluded_ids"] or []) | set(ex["gold_ids_long"] or [])
            folders = {g.split("/")[0] for g in ex["gold_ids"]}

            def skip(
                i: int,
                banned: set[str] = banned,
                folders: set[str] = folders,
                ids: list[str] = ids,
                ok_len: list[bool] = ok_len,
            ) -> bool:
                return ids[i] in banned or ids[i].split("/")[0] in folders or not ok_len[i]

            neg = bm25.top(ex["query"], 4, skip)
            if len(neg) < 4:
                continue
            q = retrieval_choice(
                "Which document is most helpful for answering this post?",
                texts[index[gold[0]]].strip(),
                [texts[i].strip() for i in neg],
                seeded(f"bright/{domain}/{ex['id']}"),
                "bright",
                "doc",
            )
            out.append(
                candidate(
                    "bright",
                    f"{domain}/{ex['id']}",
                    ex["query"].strip(),
                    {"document": q},
                    f"{domain}/{ex['id']}",
                    domain,
                    gold_id=gold[0],
                    negative_ids=[ids[i] for i in neg],
                    provenance={"via": f"hf:{BRIGHT[0]}@{BRIGHT[1]}:examples/{domain}", "query_id": ex["id"]},
                )
            )
        print(f"  bright/{domain}: {len(out)} candidates so far", flush=True)
    report["bright"] = {"candidates": len(out)}
    return split_halves(out, "bright")


def bfcl(raw: Path, report: Rec) -> list[Rec]:
    out: list[Rec] = []
    base = raw / "tools/bfcl"
    for cat in BFCL_CATEGORIES:
        items = read_jsonl(base / f"BFCL_v3_{cat}.json")
        answers: dict[str, Any]
        if "irrelevance" in cat:
            answers = {it["id"]: None for it in items}
        else:
            answers = {a["id"]: a["ground_truth"] for a in read_jsonl(base / f"possible_answer/BFCL_v3_{cat}.json")}
        out += [x for x in (bfcl_candidate(cat, it, answers[it["id"]]) for it in items if it["id"] in answers) if x]
    report["bfcl"] = {"candidates": len(out)}
    return split_halves(out, "bfcl")


def toolret(raw: Path, report: Rec) -> list[Rec]:
    corpora: dict[str, Rec] = {}
    for cat in ("web", "code", "customized"):
        rows = parquet_rows(raw / f"tools/toolret-tools/{cat}/tools-00000-of-00001.parquet")
        summaries = [tool_summary(r["documentation"]) for r in rows]
        corpora[cat] = {
            "ids": [r["id"] for r in rows],
            "index": {r["id"]: i for i, r in enumerate(rows)},
            "text": [f"{n}: {d}" for n, d in summaries],
            "bm25": BM25([f"{n} {r['documentation']}" for (n, _), r in zip(summaries, rows, strict=True)]),
        }
    queries = raw / "tools/toolret-queries"
    files = sorted(p.relative_to(queries).as_posix() for p in queries.glob("*/*.parquet"))
    out: list[Rec] = []
    for f in files:
        subset = f.split("/")[0]
        if subset in TOOLRET_EXCLUDED:
            continue
        for r in parquet_rows(queries / f):
            corpus = corpora.get(r["category"])
            labels = [lab for lab in json.loads(r["labels"]) if (lab.get("relevance") or 0) > 0]
            gold = sorted(lab["id"] for lab in labels if corpus and lab["id"] in corpus["index"])
            if not gold or corpus is None:
                continue
            gold_text = corpus["text"][corpus["index"][gold[0]]]
            banned = {lab["id"] for lab in labels}
            seen_text = {normalise_text(gold_text)}

            def skip(i: int, corpus: Rec = corpus, banned: set[str] = banned, seen_text: set[str] = seen_text) -> bool:
                t = normalise_text(corpus["text"][i])
                if corpus["ids"][i] in banned or t in seen_text:
                    return True
                seen_text.add(t)
                return False

            neg = corpus["bm25"].top(r["query"], 4, skip)
            if len(neg) < 4:
                continue
            q = retrieval_choice(
                "Which tool would be most useful for handling this request?",
                gold_text,
                [corpus["text"][i] for i in neg],
                seeded(f"toolret/{r['id']}"),
                "toolret",
                "tool",
            )
            out.append(
                candidate(
                    "toolret",
                    r["id"],
                    r["query"].strip(),
                    {"tool": q},
                    r["id"],
                    subset,
                    gold_id=gold[0],
                    negative_ids=[corpus["ids"][i] for i in neg],
                    provenance={
                        "via": f"hf:{TOOLRET[0]}@{TOOLRET[1]}:{f}",
                        "tools": f"hf:{TOOLRET[2]}@{TOOLRET[3]}:{r['category']}",
                    },
                )
            )
    report["toolret"] = {"candidates": len(out)}
    return split_halves(out, "toolret")


def apibank(raw: Path, report: Rec) -> list[Rec]:
    rows = json.loads((raw / "tools/apibank" / APIBANK[2]).read_text(encoding="utf-8"))
    parsed = [apibank_parse(r) for r in rows]
    catalog: dict[str, str] = {}
    for _, described, _ in parsed:
        catalog.update(described)
    names = sorted(catalog)
    bm25 = BM25([f"{n} {catalog[n]}" for n in names])
    out, dropped = [], 0
    for i, (r, (dialogue, described, gold)) in enumerate(zip(rows, parsed, strict=True)):
        if gold is None or gold not in described:
            dropped += 1
            continue

        def known(j: int, described: dict[str, str] = described) -> bool:
            return names[j] in described

        fill = bm25.top(dialogue, 8 - len(described), known)
        keys = sorted(described) + [names[j] for j in fill]
        seeded(f"apibank/{i}").shuffle(keys)
        q = q_choice("Which API should the assistant call next?", {k: catalog[k] for k in keys}, gold, "apibank")
        out.append(
            candidate(
                "apibank",
                f"level-1/{i}",
                dialogue,
                {"api": q},
                f"file/{r['file']}",
                gold,
                provenance={"via": f"hf:{APIBANK[0]}@{APIBANK[1]}:{APIBANK[2]}", "row": i, "file": r["file"]},
            )
        )
    report["apibank"] = {"rows": len(rows), "dropped": dropped, "candidates": len(out)}
    return split_halves(out, "apibank")


def routerbench(raw: Path, report: Rec) -> list[Rec]:
    import pandas as pd  # type: ignore[import-untyped]

    # a pickle: unpickling runs code, so `main` matches it to its sha256 pin in locks/ before anything is read
    df = pd.read_pickle(raw / "tools/routerbench" / ROUTERBENCH[2])
    out: list[Rec] = []
    for r in df.to_dict("records"):
        fam = router_family(r["eval_name"])
        if fam not in ("mmlu", "winogrande", "grade-school-math", "mbpp"):
            continue
        scores = {m: r[m] for m in ROUTER_MODELS}
        if any(s not in (0, 1, 0.0, 1.0) for s in scores.values()):
            continue
        picked = router_options({m: int(s) for m, s in scores.items()}, seeded(f"router/{r['sample_id']}"))
        if picked is None:
            continue
        keys, label = picked
        try:
            prompt = "\n\n".join(str(x) for x in ast.literal_eval(r["prompt"]))
        except (ValueError, SyntaxError):
            prompt = str(r["prompt"])
        q = q_choice("Which of these models answered this prompt correctly?", dict.fromkeys(keys), label, "routerbench")
        out.append(
            candidate(
                "routerbench",
                r["sample_id"],
                prompt.strip(),
                {"model": q},
                r["sample_id"],
                fam,
                eval_name=r["eval_name"],
                correct_models=sorted(m for m, s in scores.items() if int(s) == 1),
                provenance={
                    "via": f"hf:{ROUTERBENCH[0]}@{ROUTERBENCH[1]}:{ROUTERBENCH[2]}",
                    "sample_id": r["sample_id"],
                },
            )
        )
    report["routerbench"] = {"rows": len(df), "candidates": len(out)}
    return split_halves(out, "routerbench")


def humicroedit(raw: Path, report: Rec) -> list[Rec]:
    out = []
    with zipfile.ZipFile(raw / "arts/humicroedit/semeval-2020-task-7-dataset.zip") as z:
        for split, member in (
            ("development", "semeval-2020-task-7-dataset/subtask-2/dev.csv"),
            ("test", "semeval-2020-task-7-dataset/subtask-2/test.csv"),
        ):
            for r in csv.DictReader(io.TextIOWrapper(z.open(member), encoding="utf-8")):
                pair = humicroedit_pair(r)
                if pair is None:
                    continue
                original, one, two, label = pair
                q = q_choice(
                    "Which edited headline did readers rate funnier?", {"a": one, "b": two}, label, "humicroedit"
                )
                out.append(
                    candidate(
                        "humicroedit",
                        f"{split}/{r['id']}",
                        {"original_headline": original},
                        {"funnier": q},
                        f"headline/{r['id'].split('-')[0]}",
                        label,
                        split=split,
                        grades=[r["meanGrade1"], r["meanGrade2"]],
                        provenance={"via": f"{HUMICROEDIT_URL}:{member}", "id": r["id"]},
                    )
                )
    report["humicroedit"] = {"candidates": len(out)}
    return out


def cfcolor(raw: Path, report: Rec) -> list[Rec]:
    import scipy.io as sio  # type: ignore[import-untyped]

    with zipfile.ZipFile(raw / "arts/cfcolor/cfcolor.zip") as z:
        ratings = sio.loadmat(io.BytesIO(z.read("release/allMTurkRatings.mat")), squeeze_me=True)
        themes = sio.loadmat(io.BytesIO(z.read("release/themeData.mat")), squeeze_me=True, struct_as_record=False)[
            "datapoints"
        ]
    rgb = {i + 1: [float(v) for v in row] for i, row in enumerate(themes.rgb)}  # rating files index themes from 1
    train: dict[int, list[tuple[int, int]]] = defaultdict(list)
    test: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, t, r in ratings["train_vec"].tolist():
        train[int(u)].append((int(t), int(r)))
    for u, t, r in ratings["test_vec"].tolist():
        test[int(u)].append((int(t), int(r)))
    out = []
    for user in sorted(set(train) & set(test)):
        rec = cfcolor_record(user, train[user], test[user], rgb, seeded(f"cfcolor/{user}"))
        if rec is None:
            continue
        state, q, meta = rec
        out.append(
            candidate(
                "cfcolor",
                f"user/{user}",
                state,
                {"higher": q},
                f"user/{user}",
                q["label"],
                **meta,
                provenance={
                    "via": f"{CFCOLOR_URL}:release/allMTurkRatings.mat (train_vec history, test_vec pair)",
                    "user": user,
                },
            )
        )
    report["cfcolor"] = {"candidates": len(out)}
    return split_halves(out, "cfcolor")


BUILDERS: dict[str, Callable[[Path, Rec], list[Rec]]] = {
    "musr": musr,
    "sata_bench": sata,
    "chessbench": chessbench,
    "contractnli": contractnli,
    "hellaswag": hellaswag,
    "clinc150": clinc150,
    "sgd": sgd,
    "bright": bright,
    "bfcl": bfcl,
    "toolret": toolret,
    "apibank": apibank,
    "routerbench": routerbench,
    "humicroedit": humicroedit,
    "cfcolor": cfcolor,
}
PER_GROUP = {"sgd": 1, "apibank": 2}  # records per group within a partition

# ---------------------------------------------------------------- selection


def target(dataset: str) -> int:
    return CONTRACTNLI_DOCS if dataset == "contractnli" else EVAL_RECORDS


def select(dataset: str, pool: list[Rec], n: int, admit: Callable[[Rec], bool], split: str) -> list[Rec]:
    """Round-robin over the pool's strata (sorted), each stratum in a seeded order."""
    rng = seeded(f"select/{dataset}/{split}")
    items = sorted(pool, key=lambda x: x["_meta"]["id"])
    rng.shuffle(items)
    classes: dict[str, list[Rec]] = defaultdict(list)
    for x in items:
        classes[x["_stratum"]].append(x)
    return round_robin(classes, n, admit)


def build(raw: Path, tokenizer: Tokenizer, only: set[str] | None = None) -> tuple[dict[str, list[Rec]], Rec]:
    report: Rec = {}
    parts: dict[str, list[Rec]] = {s: [] for s in PARTITIONS}
    seen: set[str] = set()
    rejected: Counter[tuple[str, str]] = Counter()
    fit_cache: dict[str, bool] = {}
    for dataset, fn in BUILDERS.items():
        if only and dataset not in only:
            continue
        cands = fn(raw, report)
        kept, keys = [], set()
        for x in sorted(cands, key=lambda x: x["_meta"]["id"]):  # one candidate per normalised state (first by id)
            if x["_meta"]["text_sha256"] in keys:
                continue
            keys.add(x["_meta"]["text_sha256"])
            kept.append(x)
        for split in ("test", "development"):  # test first: a state shared across partitions stays in test
            pool = [x for x in kept if x["_split"] == split]
            groups: Counter[str] = Counter()

            def admit(x: Rec, dataset: str = dataset, groups: Counter[str] = groups) -> bool:
                k = x["_meta"]["text_sha256"]
                if k in seen:
                    rejected[dataset, "duplicate_state"] += 1
                    return False
                if groups[x["_meta"]["group_id"]] >= PER_GROUP.get(dataset, 10**9):
                    rejected[dataset, "group_limit"] += 1
                    return False
                if not line_safe(x):
                    rejected[dataset, "line_separator"] += 1
                    return False
                rid = x["_meta"]["id"]
                if rid not in fit_cache:
                    fit_cache[rid] = fits(x, tokenizer, **ADMISSION)
                if not fit_cache[rid]:
                    rejected[dataset, "context"] += 1
                    return False
                seen.add(k)
                groups[x["_meta"]["group_id"]] += 1
                return True

            chosen = select(dataset, pool, target(dataset), admit, split)
            if len(chosen) < target(dataset):
                raise SystemExit(f"{dataset}/{split}: only {len(chosen)} of {target(dataset)} records")
            parts[split] += chosen
        report[dataset]["pool"] = {s: sum(1 for x in kept if x["_split"] == s) for s in PARTITIONS}
        print(f"{dataset}: {report[dataset]}", flush=True)
    for split, recs in parts.items():
        for x in recs:
            stratum = x.pop("_stratum")
            x.pop("_split", None)
            x["_meta"].update(stratum=stratum, variant="clean", split=split)
            body = json.dumps({"state": x["state"], "questions": x["questions"]}, ensure_ascii=False, sort_keys=True)
            x["_meta"]["row_sha256"] = hashlib.sha256(body.encode()).hexdigest()
        random.Random(f"{SEED}:shuffle:{split}").shuffle(recs)
    report["rejected"] = {f"{d}/{k}": n for (d, k), n in sorted(rejected.items())}
    return parts, report


def check_invariants(parts: dict[str, list[Rec]]) -> None:
    """Groups never span partitions; no normalised state or id twice; every choice has 2..MAX_OPTIONS options."""
    where: dict[str, str] = {}
    keys: set[str] = set()
    ids: set[str] = set()
    for split, recs in parts.items():
        for r in recs:
            g = r["_meta"]["group_id"]
            if where.setdefault(g, split) != split:
                raise AssertionError(f"group {g} spans {where[g]} and {split}")
            if r["_meta"]["text_sha256"] in keys:
                raise AssertionError(f"duplicate state in {r['_meta']['id']}")
            if r["_meta"]["id"] in ids:
                raise AssertionError(f"duplicate id {r['_meta']['id']}")
            keys.add(r["_meta"]["text_sha256"])
            ids.add(r["_meta"]["id"])
            for q in r["questions"].values():
                if q["type"] == "choice" and not 2 <= len(q["criteria"]) <= MAX_OPTIONS:
                    raise AssertionError(f"{r['_meta']['id']}: option count")


def jsonl(records: Sequence[Rec]) -> bytes:
    """Kev's `write_jsonl`: one `json.dumps(ensure_ascii=False)` per line, UTF-8, LF."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


def verify_raw(raw: Path, lock: Path = Path("locks/raw-eval.json")) -> None:
    """Every pinned breadth raw file present and matching its sha256 in the lock, before any is read (one of them is a
    pickle, which runs code when loaded)."""
    from den.fetch import digest, read_lock

    entries = read_lock(lock)
    if entries is None:
        raise SystemExit(f"{lock} is missing: make data-download")
    root = Path(entries["root"])
    files = {root / rel: f["sha256"] for e in entries["entries"].values() for rel, f in e["files"].items()}
    pinned = {path: sha for path, sha in files.items() if raw in path.parents}
    if not pinned:
        raise SystemExit(f"no breadth raw files are pinned under {raw} in {lock}")
    if bad := [str(p) for p, sha in sorted(pinned.items()) if not p.is_file() or digest(p) != sha]:
        raise SystemExit(f"breadth raw files missing or not matching locks/ (make data-download): {bad[:5]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.build_breadth")
    p.add_argument("--raw", type=Path, default=RAW, help="the pinned breadth raw files (`make data-download`)")
    p.add_argument("--only", help="comma-separated datasets: a dry run that prints counts and writes nothing")
    p.add_argument("--staging", type=Path, default=Path("data/.breadth"), help="where a mismatching build is left")
    args = p.parse_args(argv)
    if not TOKENIZER.is_file():
        raise SystemExit(f"{TOKENIZER} is missing: make data-download fetches it (Kev's admission tokenizer)")
    verify_raw(args.raw)
    only = set(args.only.split(",")) if args.only else None
    parts, report = build(args.raw, Tokenizer.from_file(str(TOKENIZER)), only)
    check_invariants(parts)
    if only:
        print(json.dumps({s: dict(Counter(r["_meta"]["source"] for r in recs)) for s, recs in parts.items()}))
        return 0
    built = {split: jsonl(parts[split]) for split in PARTITIONS}
    got = {split: hashlib.sha256(data).hexdigest() for split, data in built.items()}
    if got != EXPECTED:
        args.staging.mkdir(parents=True, exist_ok=True)
        for split, data in built.items():
            (args.staging / f"{split}.jsonl").write_bytes(data)
        (args.staging / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        bad = [s for s in PARTITIONS if got[s] != EXPECTED[s]]
        raise SystemExit(f"breadth-v1 {bad} differ from Kev's sha256: left in {args.staging}, not written to data/")
    for split, data in built.items():
        OUT[split].parent.mkdir(parents=True, exist_ok=True)
        OUT[split].write_bytes(data)
        records = len(parts[split])
        print(f"wrote {OUT[split]}  {records} records, sha256 {got[split]} (Kev's breadth-v1 {split}, byte for byte)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
