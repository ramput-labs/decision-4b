"""The breadth-v1 port on small inputs: cases from Kev's tests/test_breadth_v1.py, plus Kev's admission count. The
whole build is checked by `scripts.breadth` itself, which writes nothing unless both partitions match Kev's sha256."""

from __future__ import annotations

import hashlib
import random
import struct
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from den.pins import BREADTH_SHA256 as EXPECTED
from scripts.breadth import (
    BM25,
    MAX_OPTIONS,
    OOS,
    OUT,
    ascii_board,
    bag_records,
    chess_options,
    clinc_options,
    contractnli_questions,
    deal_groups,
    decode_action_value,
    describe_move,
    fen_board,
    fits,
    humicroedit_pair,
    jsonl,
    round_robin,
    sgd_turns,
)

START = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def test_fen_board_ascii_and_moves() -> None:
    board = fen_board(START)
    assert board["e4"] == "P" and board["e8"] == "k" and "e2" not in board and len(board) == 32
    rows = ascii_board(START).split("\n")
    assert rows[0] == "8 r n b q k b n r" and rows[4] == "4 . . . . P . . ." and rows[-1].strip() == "a b c d e f g h"
    fen = "r3k2r/8/8/3p4/4P3/8/8/R3K2R w KQkq - 0 1"
    assert describe_move(START, "e7e5") == "black pawn e7-e5"
    assert describe_move(fen, "e4d5") == "white pawn e4-d5 capturing the pawn"
    assert describe_move(fen, "e1g1") == "white king e1-g1 (castling)"
    assert describe_move("8/4P3/8/8/8/8/8/k6K w - - 0 1", "e7e8q") == "white pawn e7-e8, promoting to queen"


def test_chess_options_keep_best_drop_near_ties_and_cap() -> None:
    moves = {f"a{i}b{i}": 0.1 + 0.01 * i for i in range(1, 9)} | {"e2e4": 0.9, "d2d4": 0.895, "g1f3": 0.5}
    moves |= {f"h{i}h{i + 1}": 0.2 for i in range(1, 8)}
    picked = chess_options(moves, random.Random(0))
    assert picked is not None
    keys, best = picked
    assert best == "e2e4" and "d2d4" not in keys and len(keys) == MAX_OPTIONS == len(set(keys))
    assert chess_options(moves, random.Random(0)) == (keys, best)
    assert chess_options({"e2e4": 0.5, "d2d4": 0.495}, random.Random(0)) is None


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def test_action_value_decoding_and_bag_reader(tmp_path: Path) -> None:
    rows = [(START, "e7e5", 0.51), ("8/8/8/8/8/8/8/k6K w - - 0 1", "h1g1", 0.5)]
    recs = [_varint(len(f)) + f.encode() + _varint(len(m)) + m.encode() + struct.pack(">d", p) for f, m, p in rows]
    ends, pos = [], 0
    for r in recs:
        pos += len(r)
        ends.append(pos)
    (tmp_path / "x.bag").write_bytes(b"".join(recs) + struct.pack(f"<{len(ends)}q", *ends))
    assert [decode_action_value(r) for r in bag_records(tmp_path / "x.bag")] == rows


def test_clinc_options_in_and_out_of_scope() -> None:
    domains = {"banking": [f"bank_{i}" for i in range(15)], "travel": [f"travel_{i}" for i in range(15)]}
    criteria, label = clinc_options("bank_3", domains, random.Random(1))
    keys = list(criteria)
    assert label == "bank_3" and len(keys) == 10 and keys[-1] == OOS and all(k.startswith("bank_") for k in keys[:-1])
    criteria, label = clinc_options("oos", domains, random.Random(2))
    assert label == OOS and len({k.split("_")[0] for k in list(criteria)[:-1]}) == 1


def test_contractnli_questions_round_robin_over_labels() -> None:
    ann = {
        "nda-1": {"choice": "Entailment"},
        "nda-2": {"choice": "Entailment"},
        "nda-3": {"choice": "NotMentioned"},
        "nda-4": {"choice": "Contradiction"},
        "nda-5": {"choice": "Entailment"},
        "nda-7": {"choice": "NotMentioned"},
    }
    qs = contractnli_questions(ann, {k: f"hypothesis {k}" for k in ann}, random.Random(0), k=4)
    assert sorted(q["label"] for q in qs.values()) == ["contradiction", "entailment", "entailment", "not_mentioned"]
    assert "hypothesis nda-4" in qs["nda-4"]["instructions"]


def test_sgd_turns_strata_and_history() -> None:
    def turn(speaker: str, text: str, intent: str | None = None) -> dict[str, object]:
        frames = [{"service": "Alarm_1", "state": {"active_intent": intent}}] if intent else []
        return {"speaker": speaker, "utterance": text, "frames": frames}

    dialogue = {"dialogue_id": "1_00001", "turns": [
        turn("USER", "hi", "NONE"), turn("SYSTEM", "hello"), turn("USER", "show my alarms", "GetAlarms"),
        turn("SYSTEM", "you have one"), turn("USER", "and the same again", "GetAlarms"), turn("SYSTEM", "ok"),
        turn("USER", "add one at 7", "AddAlarm")]}  # fmt: skip
    turns = sgd_turns(dialogue)
    assert [t[3] for t in turns] == ["none", "changed", "same", "changed"]
    assert turns[1][4] == ["User: hi", "System: hello", "User: show my alarms"]


def test_humicroedit_pairs_and_ties() -> None:
    row = {"label": "2", "original1": "Trump <visits/> Paris", "edit1": "bakes", "original2": "Trump <visits/> Paris",
           "edit2": "eats"}  # fmt: skip
    assert humicroedit_pair(row) == ("Trump visits Paris", "Trump bakes Paris", "Trump eats Paris", "b")
    assert humicroedit_pair({**row, "label": "0"}) is None


def test_bm25_ranks_by_overlap_with_index_ties_and_skips() -> None:
    bm25 = BM25(["red apple pie", "green apple", "blue sky ocean", "green apple"])
    assert bm25.top("green apple", 3) == [1, 3, 0]  # equal documents tie by index; no overlap is never returned
    assert bm25.top("green apple", 3, lambda i: i == 1) == [3, 0]


def test_selection_helpers_are_deterministic() -> None:
    classes = {"b": [1, 2, 3], "a": [10, 11]}
    assert round_robin(classes, 4, lambda x: x != 2) == [10, 1, 11, 3]
    where = deal_groups({"g1": 3, "g2": 1, "g3": 2, "g4": 2}, {"test": 0.5, "development": 0.5}, "seed")
    assert where == deal_groups({"g4": 2, "g3": 2, "g2": 1, "g1": 3}, {"test": 0.5, "development": 0.5}, "seed")
    assert set(where.values()) == {"test", "development"}


def test_fits_counts_kevs_packed_layout() -> None:
    """`<state>` + state, then per question `<q>` + instructions + (`<opt>` option `</opt>`)... + `<decide>`."""
    vocab = {w: i for i, w in enumerate(["[UNK]", "a", "b", "c", "d", "no", "yes", "x", "y"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    record = {
        "state": "a b c",
        "questions": {"q": {"type": "choice", "instructions": "d", "criteria": {"x": None, "y": "a"}}},
    }
    state, branch = 1 + 3, (1 + 1) + (2 + 1) + (2 + 3) + 1  # "y: a" is 3 tokens: y, :, a
    assert fits(record, tokenizer, max_state=state, max_branch=state + branch, max_packed=state + branch)
    assert not fits(record, tokenizer, max_state=state - 1, max_branch=10**6, max_packed=10**6)
    assert not fits(record, tokenizer, max_state=state, max_branch=state + branch - 1, max_packed=10**6)
    noul = {"state": "a", "questions": {"q": {"type": "noul", "instructions": "d"}}}
    assert fits(noul, tokenizer, max_state=2, max_branch=2 + 2 + 3 + 3 + 1, max_packed=2 + 9)


def test_jsonl_is_kevs_write_jsonl() -> None:
    assert jsonl([{"state": "é\u2028", "q": 1}]) == '{"state": "é\u2028", "q": 1}\n'.encode()


@pytest.mark.parametrize("split", ["development", "test"])
def test_built_breadth_files_are_kevs(split: str) -> None:
    if not OUT[split].is_file():
        pytest.skip("breadth-v1 not built here (make data-breadth)")
    assert hashlib.sha256(OUT[split].read_bytes()).hexdigest() == EXPECTED[split]


def test_raw_files_are_checked_against_the_lock_before_reading(tmp_path: Path) -> None:
    import json

    from scripts.breadth import verify_raw

    raw = tmp_path / "data" / "sources" / "eval" / "breadth"
    (raw / "tools").mkdir(parents=True)
    (raw / "tools" / "x.pkl").write_bytes(b"pinned")
    sha = hashlib.sha256(b"pinned").hexdigest()
    lock = tmp_path / "raw-eval.json"
    entry = {
        "use": "eval_only",
        "source": "x",
        "files": {"sources/eval/breadth/tools/x.pkl": {"bytes": 6, "sha256": sha}},
    }
    lock.write_text(json.dumps({"root": str(tmp_path / "data"), "entries": {"x": entry}}))
    verify_raw(raw, lock)
    (raw / "tools" / "x.pkl").write_bytes(b"swapped")
    with pytest.raises(SystemExit, match="not matching"):
        verify_raw(raw, lock)
