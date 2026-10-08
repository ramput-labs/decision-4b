"""Instructions follow Kev's builders; augmentation (shuffling, wrappers, none-of-the-above) belongs to training."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from .api import Json, squash
from .text import clean

type Split = Literal["train", "dev", "test"]
type Row = Mapping[str, object]
type Labels = Mapping[str, Sequence[str]]
type Converter = Callable[[Row, Labels], Json | None]

SPLITS: tuple[Split, ...] = ("train", "dev", "test")


@dataclass(frozen=True, slots=True)
class Spec:
    name: str
    native: Mapping[str, tuple[str, ...]]
    plan: Mapping[Split, tuple[str, ...]]
    convert: Converter
    folder: str | None = None

    @property
    def trainable(self) -> bool:
        return "train" in self.plan

    @property
    def raw(self) -> str:
        return f"{'train' if self.trainable else 'eval'}/{self.folder or self.name}"


def class_names(path: Path) -> dict[str, list[str]]:
    """Hugging Face embeds ClassLabel names in the parquet schema metadata."""
    if path.suffix != ".parquet":
        return {}
    metadata = pq.read_schema(path).metadata or {}
    features = json.loads(metadata.get(b"huggingface", b"{}")).get("info", {}).get("features", {})
    return {k: list(v["names"]) for k, v in features.items() if isinstance(v, dict) and v.get("_type") == "ClassLabel"}


def _str(row: Row, key: str) -> str | None:
    value = row.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _int(row: Row, key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return int(value) if isinstance(value, str) and value.lstrip("-").isdigit() else None


def _label(row: Row, key: str, names: Sequence[str]) -> str | None:
    index = _int(row, key)
    return names[index] if index is not None and 0 <= index < len(names) else None


def _request(state: Json, **questions: Json) -> Json:
    return {"state": state, "questions": questions}


def _choice(instructions: str, criteria: Mapping[str, Json], label: str) -> Json:
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria), "label": label}


def _score(instructions: str, levels: Sequence[str], label: int) -> Json:
    return {"type": "score", "instructions": instructions, "criteria": list(levels), "label": label}


def _noul(instructions: str, label: bool, criteria: Mapping[str, str] | None = None) -> Json:
    question: dict[str, Json] = {"type": "noul", "instructions": instructions, "label": label}
    if criteria:
        question["criteria"] = dict(criteria)
    return question


def _slug(name: str) -> str:
    """'MeanOfTransportation' -> 'mean_of_transportation'; 'PII/Privacy' -> 'pii_privacy'."""
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()).strip("_")


def _mcq(state: Json, options: Sequence[str], answer: int | None) -> Json | None:
    """Neutral keys in published order. Repeated wrong options collapse to one; a repeated answer is ambiguous."""
    if answer is None or not 0 <= answer < len(options) or options.count(options[answer]) > 1:
        return None
    unique = list(dict.fromkeys(options))
    keys = [f"opt_{i + 1}" for i in range(len(unique))]
    label = keys[unique.index(options[answer])]
    criteria = dict(zip(keys, unique, strict=True))
    return _request(state, answer=_choice("Which option correctly answers the question?", criteria, label))


def _choices(row: Row, answer_key: str) -> tuple[list[str], int | None] | None:
    choices = row.get("choices")
    if not isinstance(choices, dict):
        return None
    texts, labels = choices.get("text"), choices.get("label")
    if not (isinstance(texts, list) and isinstance(labels, list)) or len(texts) != len(labels):
        return None
    options = [clean(t) for t in texts if isinstance(t, str)]
    answer = row.get(answer_key)
    return options, labels.index(answer) if answer in labels else None


def _seeded_order(seed: str, items: Sequence[str]) -> list[int]:
    return sorted(range(len(items)), key=lambda i: hashlib.sha256(f"{seed}\x1f{items[i]}".encode()).hexdigest())


AG_NEWS = {
    "world": "World news: politics, international affairs, conflicts",
    "sports": "Sports: games, athletes, teams, results",
    "business": "Business: companies, markets, economy, finance",
    "scitech": "Science and technology: research, gadgets, software, space",
}
MNLI = {
    "entailment": "The hypothesis follows from the premise",
    "neutral": "The hypothesis may or may not be true given the premise",
    "contradiction": "The hypothesis contradicts the premise",
}
TREC = {
    "abbreviation": "Asks what an abbreviation stands for",
    "entity": "Asks about a thing, object, animal, product, or creative work",
    "description": "Asks for a definition, description, reason, or manner",
    "human": "Asks about a person, group, or organisation",
    "location": "Asks about a place",
    "number": "Asks for a number, date, count, or other numeric value",
}
SST5 = ("very negative", "negative", "neutral", "positive", "very positive")
YELP = ("1 star: terrible experience", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent")
AMAZON = ("1 star: very negative", "2 stars: negative", "3 stars: mixed", "4 stars: positive", "5 stars: very positive")
AEGIS_CATEGORIES = (
    "Criminal Planning/Confessions", "Needs Caution", "Violence", "Hate/Identity Hate", "Harassment", "Profanity",
    "Controlled/Regulated Substances", "Sexual", "PII/Privacy", "Guns and Illegal Weapons", "Suicide and Self Harm",
    "Unauthorized Advice", "Political/Misinformation/Conspiracy", "Fraud/Deception", "Other", "Immoral/Unethical",
    "Illegal Activity", "Sexual (minor)", "Threat", "Malware", "Copyright/Trademark/Plagiarism",
    "High Risk Gov Decision Making", "Manipulation",
)  # fmt: skip
WHEN2CALL = {
    "tool_call": "Call one of the available tools",
    "request_for_info": "Ask the user for missing information",
    "cannot_answer": "Say it cannot be done with the available tools",
    "direct": "Answer directly without a tool",
}


def banking77(row: Row, labels: Labels) -> Json | None:
    text, intent = _str(row, "text"), _label(row, "label", labels["label"])
    if text is None or intent is None:
        return None
    criteria = {name: None for name in labels["label"]}
    return _request(
        clean(text), intent=_choice("Which banking intent best describes this customer message?", criteria, intent)
    )


def massive(row: Row, labels: Labels) -> Json | None:
    text = _str(row, "utt")
    intent, scenario = _label(row, "intent", labels["intent"]), _label(row, "scenario", labels["scenario"])
    if text is None or intent is None or scenario is None:
        return None
    return _request(
        clean(text),
        intent=_choice(
            "Which intent does this request to a voice assistant express?", dict.fromkeys(labels["intent"]), intent
        ),
        scenario=_choice("Which scenario does this request belong to?", dict.fromkeys(labels["scenario"]), scenario),
    )


def ag_news(row: Row, labels: Labels) -> Json | None:
    text, index = _str(row, "text"), _int(row, "label")
    if text is None or index is None or not 0 <= index < len(AG_NEWS):
        return None
    topic = list(AG_NEWS)[index]
    return _request(
        clean(text, escapes=True, markup=True), topic=_choice("What is the topic of this article?", AG_NEWS, topic)
    )


def trec(row: Row, labels: Labels) -> Json | None:
    text, index = _str(row, "text"), _int(row, "coarse_label")
    if text is None or index is None or not 0 <= index < len(TREC):
        return None
    kind = list(TREC)[index]
    return _request(clean(text), answer_type=_choice("What kind of answer does this question ask for?", TREC, kind))


def dbpedia14(row: Row, labels: Labels) -> Json | None:
    text, category = _str(row, "content"), _label(row, "label", labels["label"])
    if text is None or category is None:
        return None
    criteria = {_slug(name): None for name in labels["label"]}
    instructions = "Which category does the subject of this encyclopedia text belong to?"
    return _request(clean(text), category=_choice(instructions, criteria, _slug(category)))


def boolq(row: Row, labels: Labels) -> Json | None:
    passage, question, answer = _str(row, "passage"), _str(row, "question"), row.get("answer")
    if passage is None or question is None or not isinstance(answer, bool):
        return None
    instructions = clean(question).rstrip("?").capitalize() + "?"
    return _request(clean(passage), answer=_noul(instructions, answer))


def multinli(row: Row, labels: Labels) -> Json | None:
    premise, hypothesis = _str(row, "premise"), _str(row, "hypothesis")
    relation = _label(row, "label", labels["label"])
    if premise is None or hypothesis is None or relation is None:
        return None
    instructions = f'Hypothesis: "{clean(hypothesis, markup=True)}" How does it relate to the premise?'
    return _request(clean(premise, markup=True), relation=_choice(instructions, MNLI, relation))


def _stars(levels: Sequence[str], instructions: str, *, escapes: bool = False) -> Converter:
    def convert(row: Row, labels: Labels) -> Json | None:
        text, level = _str(row, "text"), _int(row, "label")
        if text is None or level is None or not 0 <= level < len(levels):
            return None
        return _request(clean(text, escapes=escapes, markup=escapes), rating=_score(instructions, levels, level))

    return convert


def imdb(row: Row, labels: Labels) -> Json | None:
    text, label = _str(row, "text"), _int(row, "label")
    if text is None or label not in (0, 1):
        return None
    criteria = {"true": "The reviewer liked the film overall", "false": "The reviewer disliked the film overall"}
    return _request(clean(text, markup=True), positive=_noul("Is this movie review positive?", label == 1, criteria))


@cache
def _aart_prompts(root: Path) -> frozenset[str]:
    table = pq.read_table(
        root / "sources/train/safety/aart-filter/data/train-00000-of-00001.parquet", columns=["prompt"]
    )
    return frozenset(squash(p) for p in table.column("prompt").to_pylist() if isinstance(p, str))


def aegis(root: Path) -> Converter:
    def convert(row: Row, labels: Labels) -> Json | None:
        prompt, verdict = _str(row, "prompt"), row.get("prompt_label")
        if prompt is None or prompt == "REDACTED" or verdict not in ("safe", "unsafe"):
            return None
        if squash(prompt) in _aart_prompts(root):
            return None
        criteria = {"true": "Asks for or promotes harmful content", "false": "A safe request"}
        questions: dict[str, Json] = {"unsafe": _noul("Is this prompt unsafe?", verdict == "unsafe", criteria)}
        categories = [c.strip() for c in str(row.get("violated_categories") or "").split(",") if c.strip()]
        if verdict == "unsafe" and row.get("response") is None and len(categories) == 1:
            if categories[0] not in AEGIS_CATEGORIES:
                return None
            options = {_slug(c): c for c in AEGIS_CATEGORIES}
            questions["category"] = _choice(
                "Which safety category does this prompt violate?", options, _slug(categories[0])
            )
        return _request(clean(prompt, markup=True), **questions)

    return convert


def _question_mcq(field: str) -> Converter:
    """ARC, OpenBookQA and CommonsenseQA: a question plus labelled `choices`."""

    def convert(row: Row, labels: Labels) -> Json | None:
        question, parsed = _str(row, field), _choices(row, "answerKey")
        return _mcq({"question": clean(question)}, *parsed) if question and parsed else None

    return convert


def qasc(row: Row, labels: Labels) -> Json | None:
    question, parsed = _str(row, "question"), _choices(row, "answerKey")
    facts = [clean(f) for key in ("fact1", "fact2") if (f := _str(row, key))]
    if question is None or parsed is None or len(facts) != 2:
        return None
    return _mcq({"facts": list(facts), "question": clean(question)}, *parsed)


def _functions(system: str) -> list[dict[str, Json]] | None:
    decoder, found, i = json.JSONDecoder(), [], system.find("{")
    while i != -1:
        try:
            spec, end = decoder.raw_decode(system, i)
        except json.JSONDecodeError:
            return None
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            return None
        found.append(spec)
        i = system.find("{", end)
    return found or None


_TURN = re.compile(r"(?:^|\n)\s*(USER|ASSISTANT|FUNCTION RESPONSE):")
_CALL = re.compile(r'<functioncall>\s*\{\s*"name"\s*:\s*"([^"]+)"')
_REFUSAL = re.compile(
    r"\b(?:sorry|unable|can't|cannot|can not|not able|don't have the (?:ability|capability)|beyond my|limited to)\b",
    re.I,
)


def _glaive_decision(chat: str) -> tuple[str, str | None, bool | None] | None:
    """(request, function called for it, whether the request already held its arguments), or None if ambiguous.

    Glaive's assistant often asks for missing arguments first and calls a turn later; that request still fits the
    function, so only a conversation that never calls one is a "no". A call any later, or after a refusal (the user
    then asks for something else), can't be tied to the first request and is left out."""
    parts = _TURN.split(chat)
    if parts[0].strip():
        return None
    turns = [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
    if len(turns) < 2 or turns[0][0] != "USER" or turns[1][0] != "ASSISTANT":
        return None
    request, reply = turns[0][1], turns[1][1]
    if called := _CALL.match(reply):
        return request, called.group(1), True
    if "<functioncall>" not in chat:
        return request, None, None
    if len(turns) < 4 or turns[2][0] != "USER" or turns[3][0] != "ASSISTANT" or _REFUSAL.search(reply):
        return None
    called = _CALL.match(turns[3][1])
    return (request, called.group(1), False) if called else None


GLAIVE_SWAP = 0.25  # share of answered requests re-asked against an unrelated conversation's functions, as a "no"
GLAIVE_RAW = "sources/train/tools/glaive-function-calling/glaive-function-calling-v2.json"
_GENERIC = frozenset(
    (
        "about", "amount", "based", "calculate", "certain", "check", "convert", "create", "current", "data", "details",
        "find", "from", "generate", "given", "information", "into", "list", "name", "number", "provide", "random",
        "search", "send", "specific", "specified", "that", "the", "this", "user", "using", "value", "which", "with",
    )
)  # fmt: skip


def _fraction(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 2**64


def _words(f: Mapping[str, Json]) -> frozenset[str]:
    text = f"{f.get('name', '')} {f.get('description', '')}".lower()
    return frozenset(w for w in re.findall(r"[a-z]+", text) if len(w) > 2) - _GENERIC


@cache
def _glaive_catalogs(root: Path) -> tuple[tuple[dict[str, Json], ...], ...]:
    """Every distinct function list in Glaive, in a fixed order: the pool unrelated catalogs are drawn from."""
    found: dict[str, tuple[dict[str, Json], ...]] = {}
    for row in json.loads((root / GLAIVE_RAW).read_text(encoding="utf-8")):
        if isinstance(system := row.get("system"), str) and (functions := _functions(system)):
            found.setdefault(json.dumps(functions, sort_keys=True), tuple(functions))
    return tuple(found[key] for key in sorted(found))


def _unrelated(
    pool: Sequence[Sequence[dict[str, Json]]], seed: str, request: str, called: Mapping[str, Json]
) -> list[dict[str, Json]] | None:
    """A catalog none of whose functions shares a content word with the request or the function that answered it."""
    words, start = _words(called), int(_fraction(seed) * len(pool))
    if not words:
        return None
    words |= _words({"description": request})
    for k in range(min(len(pool), 64)):
        catalog = pool[(start + k) % len(pool)]
        if not any(words & _words(f) for f in catalog):
            return list(catalog)
    return None


def glaive(root: Path) -> Converter:
    """Glaive's first request against its functions. Of the answered ones, a seeded GLAIVE_SWAP share is asked
    against another conversation's unrelated functions instead, so "no function fits" isn't only pizza and flights."""

    def convert(row: Row, labels: Labels) -> Json | None:
        system, chat = _str(row, "system"), _str(row, "chat")
        if system is None or chat is None or (decision := _glaive_decision(chat)) is None:
            return None
        functions = _functions(system)
        if functions is None:
            return None
        request, called, ready = decision
        request_text = clean(request)
        seed = f"{system}\x1f{request_text}"
        if called is not None and _fraction(f"glaive-swap:{seed}") < GLAIVE_SWAP:
            spec = next((f for f in functions if f["name"] == called), None)
            swapped = _unrelated(_glaive_catalogs(root), seed, request_text, spec) if spec else None
            if swapped is not None:
                functions, called, ready = swapped, None, None
        functions = [functions[i] for i in _seeded_order(request_text, [str(f["name"]) for f in functions])]
        names = [str(f["name"]) for f in functions]
        if len(set(names)) != len(names) or (called and called not in names):
            return None
        catalog: list[Json] = []
        for f in functions:
            params = f.get("parameters")
            props = params.get("properties") if isinstance(params, dict) else None
            entry: dict[str, Json] = {"name": str(f["name"]), "description": clean(str(f.get("description", "")))}
            if isinstance(props, dict) and props:
                entry["parameters"] = ", ".join(props)
            catalog.append(entry)
        questions: dict[str, Json] = {
            "call": _noul(
                "Should the assistant call one of the available functions for this request?", called is not None
            )
        }
        if ready is not None:
            criteria = {"true": "Call the function now", "false": "Ask the user for the missing details first"}
            instructions = "Does the request already give everything the function needs, so it can be called now?"
            questions["ready"] = _noul(instructions, ready, criteria)
        if len(names) > 1:
            options: dict[str, Json] = {**dict.fromkeys(names), "none": "No function fits this request"}
            questions["function"] = _choice("Which function should handle this request?", options, called or "none")
        return _request({"functions": catalog, "request": request_text}, **questions)

    return convert


def qnli(row: Row, labels: Labels) -> Json | None:
    question, sentence, label = _str(row, "question"), _str(row, "sentence"), _int(row, "label")
    if question is None or sentence is None or label not in (0, 1):
        return None
    instructions = f'Does the sentence contain the answer to this question: "{clean(question)}"'
    return _request(clean(sentence), answers=_noul(instructions, label == 0))


def paws(row: Row, labels: Labels) -> Json | None:
    first, second, label = _str(row, "sentence1"), _str(row, "sentence2"), _int(row, "label")
    if first is None or second is None or label not in (0, 1):
        return None
    criteria = {"true": "Same meaning, possibly reworded", "false": "Different meaning, even if most words match"}
    return _request(
        clean(first),
        paraphrase=_noul(f'Does this sentence mean the same thing: "{clean(second)}"', label == 1, criteria),
    )


def sciq(row: Row, labels: Labels) -> Json | None:
    question, answer = _str(row, "question"), _str(row, "correct_answer")
    distractors = [d for k in ("distractor1", "distractor2", "distractor3") if (d := _str(row, k))]
    if question is None or answer is None or len(distractors) != 3:
        return None
    options = [clean(answer), *(clean(d) for d in distractors)]
    order = _seeded_order(question, options)
    support = _str(row, "support")
    state: dict[str, Json] = (
        {"passage": clean(support), "question": clean(question)} if support else {"question": clean(question)}
    )
    return _mcq(state, [options[i] for i in order], order.index(0))


def mmlu(row: Row, labels: Labels) -> Json | None:
    question, subject, choices = _str(row, "question"), _str(row, "subject"), row.get("choices")
    if question is None or subject is None or not isinstance(choices, list):
        return None
    options = [clean(str(c)) for c in choices]
    return _mcq({"subject": subject.replace("_", " "), "question": clean(question)}, options, _int(row, "answer"))


def mmlu_pro(row: Row, labels: Labels) -> Json | None:
    question, category, choices = _str(row, "question"), _str(row, "category"), row.get("options")
    if question is None or category is None or not isinstance(choices, list):
        return None
    options = [clean(str(c)) for c in choices]
    return _mcq({"subject": category, "question": clean(question)}, options, _int(row, "answer_index"))


def emotion(row: Row, labels: Labels) -> Json | None:
    text, feeling = _str(row, "text"), _label(row, "label", labels["label"])
    if text is None or feeling is None:
        return None
    criteria = dict.fromkeys(labels["label"])
    return _request(clean(text), emotion=_choice("Which emotion does the writer express?", criteria, feeling))


def tweeteval_offensive(row: Row, labels: Labels) -> Json | None:
    text, label = _str(row, "text"), _int(row, "label")
    if text is None or label not in (0, 1):
        return None
    criteria = {
        "true": "Contains insults, threats, profanity directed at someone, or hateful content",
        "false": "Not offensive",
    }
    return _request(clean(text, markup=True), offensive=_noul("Is this post offensive?", label == 1, criteria))


def when2call(row: Row, labels: Labels) -> Json | None:
    question, action, tools = _str(row, "question"), row.get("correct_answer"), row.get("tools")
    if question is None or action not in WHEN2CALL or not isinstance(tools, list):
        return None
    catalog: list[Json] = []
    for raw in tools:
        tool = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            return None
        catalog.append({"name": tool["name"], "description": clean(str(tool.get("description", "")))})
    instructions = "What should the assistant do with this request?"
    return _request(
        {"tools": catalog, "request": clean(question)}, action=_choice(instructions, WHEN2CALL, str(action))
    )


def prompt_injection(row: Row, labels: Labels) -> Json | None:
    text = _str(row, "text")
    label = _int(row, "label") if "label" in row else 1
    if text is None or label not in (0, 1):
        return None
    criteria = {"true": "Tries to override or extract the assistant's instructions", "false": "An ordinary request"}
    return _request(clean(text), injection=_noul("Is this a prompt injection?", label == 1, criteria))


STANDARD: dict[Split, tuple[str, ...]] = {"train": ("train",), "dev": ("validation",), "test": ("test",)}
DEV_FROM_TRAIN: dict[Split, tuple[str, ...]] = {"train": ("train",), "dev": (), "test": ("test",)}
VALIDATION_AS_TEST: dict[Split, tuple[str, ...]] = {"train": ("train",), "dev": (), "test": ("validation",)}
EVAL_ONLY: dict[Split, tuple[str, ...]] = {"dev": ("validation",), "test": ("test",)}


def _each(pattern: str, *splits: str) -> dict[str, tuple[str, ...]]:
    return {split: (pattern.format(split),) for split in splits}


def specs(root: Path) -> tuple[Spec, ...]:
    tvt = ("train", "validation", "test")
    return (
        Spec("intent/banking77", _each("data/{}-*", "train", "test"), DEV_FROM_TRAIN, banking77),
        Spec("intent/massive-en", _each("en-US/{}/*", *tvt), STANDARD, massive),
        Spec("topic/ag-news", _each("data/{}-*", "train", "test"), DEV_FROM_TRAIN, ag_news),
        Spec("topic/trec", _each("default/{}/*", "train", "test"), DEV_FROM_TRAIN, trec),
        Spec(
            "topic/dbpedia14",
            _each("dbpedia_14/{}-*", "train", "test"),
            DEV_FROM_TRAIN,
            dbpedia14,
        ),
        Spec("reading/boolq", _each("data/{}-*", "train", "validation"), VALIDATION_AS_TEST, boolq),
        Spec(
            "reading/multinli",
            {
                "train": ("data/train-*",),
                "matched": ("data/validation_matched-*",),
                "mismatched": ("data/validation_mismatched-*",),
            },
            {"train": ("train",), "dev": ("matched",), "test": ("mismatched",)},
            multinli,
        ),
        Spec(
            "sentiment/sst5",
            _each("{}.jsonl", "train", "dev", "test"),
            {"train": ("train",), "dev": ("dev",), "test": ("test",)},
            _stars(SST5, "What is the sentiment of this review sentence?"),
        ),
        Spec(
            "sentiment/yelp-full",
            _each("yelp_review_full/{}-*", "train", "test"),
            DEV_FROM_TRAIN,
            _stars(YELP, "How many stars did this reviewer give?", escapes=True),
        ),
        Spec(
            "sentiment/amazon-reviews",
            _each("{}.jsonl", *tvt),
            STANDARD,
            _stars(AMAZON, "How many stars did this product reviewer give?"),
        ),
        Spec("sentiment/imdb", _each("plain_text/{}-*", "train", "test"), DEV_FROM_TRAIN, imdb),
        Spec("safety/aegis", _each("{}.json", *tvt), STANDARD, aegis(root)),
        Spec("knowledge/arc", _each("ARC-*/{}-*", *tvt), STANDARD, _question_mcq("question")),
        Spec("knowledge/openbookqa", _each("main/{}-*", *tvt), STANDARD, _question_mcq("question_stem")),
        Spec(
            "knowledge/commonsenseqa",
            _each("data/{}-*", "train", "validation"),
            VALIDATION_AS_TEST,
            _question_mcq("question"),
        ),
        Spec(
            "knowledge/qasc",
            _each("data/{}-*", "train", "validation"),
            VALIDATION_AS_TEST,
            qasc,
        ),
        Spec(
            "tools/glaive-function-calling",
            {"train": ("glaive-function-calling-v2.json",)},
            {"train": ("train",), "dev": (), "test": ()},
            glaive(root),
        ),
        Spec(
            "transfer/qnli",
            _each("qnli/{}-*", "train", "validation"),
            {"dev": ("train",), "test": ("validation",)},
            qnli,
        ),
        Spec("transfer/paws", _each("labeled_final/{}-*", "validation", "test"), EVAL_ONLY, paws),
        Spec("transfer/sciq", _each("data/{}-*", "validation", "test"), EVAL_ONLY, sciq),
        Spec("transfer/mmlu", _each("all/{}-*", "validation", "test"), EVAL_ONLY, mmlu),
        Spec("transfer/mmlu-pro", _each("data/{}-*", "validation", "test"), EVAL_ONLY, mmlu_pro),
        Spec("transfer/emotion", _each("split/{}-*", "validation", "test"), EVAL_ONLY, emotion),
        Spec(
            "transfer/tweeteval-offensive",
            _each("offensive/{}-*", "validation", "test"),
            EVAL_ONLY,
            tweeteval_offensive,
        ),
        Spec(
            "devtools/when2call",
            {"test": ("test/when2call_test_mcq.jsonl",)},
            {"dev": (), "test": ("test",)},
            when2call,
        ),
        Spec(
            "devtools/prompt-injection",
            {
                "dev": (
                    "prompt-injection-deepset/data/train-*",
                    "prompt-injection-gandalf/data/train-*",
                    "prompt-injection-gandalf/data/validation-*",
                ),
                "test": ("prompt-injection-deepset/data/test-*", "prompt-injection-gandalf/data/test-*"),
            },
            {"dev": ("dev",), "test": ("test",)},
            prompt_injection,
            folder="devtools",
        ),
    )
