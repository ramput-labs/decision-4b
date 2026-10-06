# den

**System One decision models** on any Qwen backbone, from 0.6B to 35B. A model reads a document (the *state*)
and typed questions about it, and returns a calibrated probability over the options of each question. It does this in
one forward pass and generates no text. The recipe is Kev's ([jaredpalmer/kev](https://github.com/jaredpalmer/kev)),
the open reconstruction of TypeSafe's Jev: a base model with a LoRA adapter and a **pointer head**, served behind
TypeSafe's `POST /v1/systemone`. The goal is to match each Kev model at its size, on Kev 1.0's own data, then beat it.

## Models

`make models` lists every backbone. Each one is pinned to a commit, with the sha256 of every weight shard:

| Family | Sizes (`MODEL=` key) |
|---|---|
| Qwen3 | `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-30b-a3b` (MoE) |
| Qwen3.5 | `qwen3.5-0.8b`, `qwen3.5-2b`, `qwen3.5-4b` (default), `qwen3.5-9b`, `qwen3.5-35b-a3b` (MoE) |
| Qwen3.8 | `qwen3.8-27b` |
| Kev 1.0 (reference) | `kev-0.8b`, `kev-4b`, `kev-9b`, `kev-27b`, the baselines to beat at each size |

Any other Hub model works too: `make model MODEL=hf:<org>/<name>@<40-hex commit>`. Its digests are frozen in the
lock on first download. Choose the model with `MODEL=` on make targets, or set `DEN_MODEL` for every command.
The backends load any size the device can hold, and refuse a model that won't fit before loading it.

## System One rules

1. **One pass, no generation.** The backbone prefills `state + question + options`. The pointer head scores each
   option's closing token against the question's final token, and a softmax turns the scores into probabilities.
2. **Typed questions.** `choice` (1–255 named options), `score` (ordered levels), `noul` (yes/no). The request shape
   is the API, and training records use the same bytes the server sends to the model.
3. **Calibrated.** One temperature `T` is fitted on a held-out calibration split. It is never fitted on dev or test.
4. **Honest evaluation.** Eval-only sources are never trained on. Locked test partitions are read once per release
   candidate. The catalog enforces both (`den/catalog.py`, `tests/`).
5. **Pinned.** Every download names a commit and a sha256. `locks/*.json` records every byte that was placed.

## Steps

You need [uv](https://docs.astral.sh/uv/) and disk for your chosen model (1.2 GB for `qwen3-0.6b`, 9.3 GB for
`qwen3.5-4b`, 56 GB for `qwen3.8-27b`) plus about 100 MB for the default data.
Optionally run `uvx hf auth login` to avoid Hub rate limits.

```bash
make setup          # Python 3.12 + this platform's runtime (MLX on Apple Silicon, PyTorch + CUDA on Linux)
make check          # ruff, mypy --strict, unit tests
make model          # the default backbone, qwen3.5-4b; or MODEL=qwen3-0.6b, MODEL=qwen3.5-9b, ...
make data           # the suites: train / calibration / dev / test (~100 MB, sha256-verified)
make verify         # re-hash everything against locks/
make normalize      # raw sources -> canonical, leakage-free train / dev / test (needs data-raw-train, -new, -eval)
make clean-data     # cleaned, deduplicated copies of every suite and source under data/clean/ (pinned files untouched)
make audit          # check every record is fit to train and evaluate on
make smoke          # run MODEL once on this machine's backend
```

## Fine-tuning (Unsloth + LoRA)

For a rented H100, follow [`instructions.md`](instructions.md): a phase-by-phase runbook with a gate after each phase,
from an empty machine to an uploaded model checked over HTTP.

Training runs in two rounds, each a chain of stages where every stage continues from the last (`--init-from`):

- **Round 1, `make train-kev`:** Kev-4B's own recipe, from its model card and code. It has four stages: `core` ×2 at
  5e-5 with 25% none-of-the-above minimal pairs, then dates, then documents, then skills+devtools, at 2e-5 with
  2k/2k/4k `core` records replayed.
- **Round 2, `make train-round2`:** continues from round 1 on 17 public sources Kev-4B never trained on (`CAP`
  records each) with `CAP` records replayed from each of Kev's suites. `den compare` ships round 2 only if it beats
  round 1 on dev without losing more than 1 point on any file.

Every stage: one row per question (the state plus one question), Kev's option augmentation, LoRA rank 16 (alpha 32,
dropout 0) on all 248 text-decoder projections (the attention, Gated DeltaNet and MLP projections, none in the
vision tower), with fp32 adapters over a bf16 base. Validation on a dev sample runs every `--eval-steps` steps, and
T is fitted on `calibration/core`. Unsloth needs Linux and an NVIDIA GPU; on a Mac, `--engine peft` runs the same
adapters to check the wiring. This follows [Unsloth's Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune),
which advises against QLoRA on Qwen3.5.

```bash
make clean-data && make train-check          # data/clean/, then tokenize and report sizes (no GPU)
make train-setup                             # on the GPU box: Unsloth, keeping the locked torch/transformers
make train-kev                               # round 1 -> runs/kev-recipe/4-skills
make train-round2 CAP=1500                   # round 2 -> runs/round2 (needs data-raw-* + normalize)
uv run den evaluate --run runs/round2 dev/core.jsonl dev/documents.jsonl        # acc, NLL, ECE; test needs --final
uv run den compare runs/kev-recipe/4-skills runs/round2                         # which one to ship, judged on dev
uv run den publish --run runs/round2 --repo <org>/<name> --logs runs-train.log  # private Hub model, stages, data, logs
make train ARGS="--lora 0 --head-lr 1e-3"    # baseline: frozen Qwen, pointer head only
```

### Experiments and gates

| Mode | Command |
|---|---|
| A zero-shot Qwen (its own answer-letter probabilities) | `den train --head-kind letters --lora 0 --epochs 0` |
| B Qwen + LoRA, answering with the letter | `den train --head-kind letters` |
| C frozen Qwen + pointer head | `den train --lora 0 --head-lr 1e-3` |
| D Qwen + LoRA + pointer head (the release recipe) | `make train-kev`, `make train-round2` |

Ablations: `--lora-targets all|attention-mlp|attention`, `--option-rep end|marker|mean|attn`, `--head-proj
linear|mlp`, `--head-kind set|pointer`, `--ordinal-weight`. Gates: `den overfit` (the head must fit 100 real
examples; then determinism, option-permutation and option-replacement checks) and `den train --overfit 100` (the
same through LoRA and the real training loop). `den probe` compares modes A and C quickly on frozen-backbone features (also on a Mac). `den evaluate --augment pairs|none-replace|none-add|distract` measures
Kev's augmentations, with `pair_accuracy` for minimal pairs.

## Use a trained model

`den serve` puts a run behind Kev's (and TypeSafe's) `POST /v1/systemone`, with the same request and response, so
existing clients and the TypeSafe SDK work unchanged. `den predict` prints the same responses for JSONL requests. A
run is a local directory or `hf:<org>/<name>` (downloaded once into the Hub cache).

```bash
make serve RUN=hf:<org>/<name>               # http://127.0.0.1:8009; DEN_API_KEY=... requires a bearer key
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery, delays", "billing": "Charges"}},
    "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}}}'
echo '{"state": ..., "questions": {...}}' | uv run den predict --run hf:<org>/<name>
```

From Python, the same response without a server:

```python
import den

model = den.load("hf:<org>/<name>")      # or a local run directory
model.predict({"state": "...", "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}})
```

The response holds, per question, `choice` (most likely option), `score` (expected level) or `noul` (probability of
true), with `confidence` = (p_max − 1/K)/(1 − 1/K) and every option's probability, plus `usage` and `latency_ms`.
Each question is read with the state alone.

The prompt layout (`den/prompt.py`) is this repo's own, not Kev's serving template. The pointer head
(`den/model.py`) starts as Kev's, q(question's final token) · k(option's closing token), and adds span pooling, one
cross-option attention block and a per-option prior, all starting at zero (`--head-layers 0` drops the block;
`--head-kind pointer` is Kev's head exactly). Backbone and head are independent: `MODEL=` picks any Qwen backbone,
`--head-*` the head. A run exports Hugging Face-style files only: `merged/` (a standard checkpoint), the PEFT
adapter, and the head as `head.safetensors` + `head.json` (its `hidden_size` must match the backbone's). For score
questions, `--ordinal-weight` adds Kev's ranked probability score, so near-miss levels cost less than far ones.
`den evaluate` reports accuracy, NLL, Brier, ECE, coverage at 5% error, accuracy per question type, and the
expected level's error on score questions.

Optional:

```bash
make model MODEL=kev-4b   # the Kev checkpoint to beat at your size (kev-0.8b, kev-9b, kev-27b)
make data-raw-eval   # eval-only raw sources (rebuilds Kev's private breadth-v1 panel)
make data-raw-new    # new trainable sources aimed at Kev's weak spots
make data-raw-train  # raw sources Kev trained from, at Kev's pins (to regenerate or scale the suites)
make data-raw-bulk   # CFPB, CommitPackFT, CodeReviewer, FlakeFlagger raws (~5 GB)
make models          # print every backbone and reference checkpoint
make list            # print every dataset
make env             # show which backend this machine uses
make test-model      # MLX vs PyTorch parity on the downloaded qwen3.5-4b
```

Tools that aren't part of the library live in `scripts/` (`uv run python -m scripts.<name>`): the breadth-v1
builder, the data mirror, and the runbook's environment and timing checks.

Eval-only extras beside Kev's suites: `make breadth` rebuilds Kev's breadth-v1 panel (14 public datasets) from the
pinned raws, byte for byte against Kev's published sha256, into `data/{dev,test}/breadth.jsonl`; `binding` and
`semif` (Kev's role-binding diagnostic and SemIf's authored decisions) come with `make data`. `den evaluate` reports
Kev's robustness checks (option-order flips, contrastive pair flips, confidence on unknowable items) where a file
carries them, and `--augment permute` measures option-order sensitivity on any file.

Models are versioned: `make release VERSION=v1 RUN=<run> REPO=<org>/<name>` publishes a run, tags the Hub commit `v1`
and records it in `releases/v1.json`. The next version continues from it on new data
(`make train-next FROM=v1 OUT=runs/v2 DATA="train/<new>.jsonl"`, replaying Kev's suites so nothing is forgotten) and is
released with `PARENT=v1`, which refuses it if any dev file fell more than a point below v1. `release:v1` works
wherever a run does (`--init-from`, `--run`).

Every evaluation is kept in git: `den evaluate` writes `reports/runs/<run>/` (each file's metrics, every question's
probabilities, and where they came from), and `den compare --paired A B` says whether B beats A beyond noise, question
by question with 95% intervals. Numbers printed in the docs are listed in `reports/claims.json` and checked against
that evidence by `make check`.

Every source's licence is recorded with its evidence in `den/licences.py`. `make licences` writes them into `data/`
(`README.md` as the dataset card, `LICENSES.md` with each file's sources, `LICENSES/` with per-source terms, licence
texts and the licence files shipped with the data), and a `SOURCE-LICENSE.md` beside every dataset saying what each
of its files holds, under which licence, and whether the uploaded copy has it. `make upload-data` never uploads sources whose terms forbid
redistribution (Yelp, Amazon reviews, the raw HellaSwag and RouterBench files), and keeps unlicensed ones out of
public copies; Kev's `core` suite holds Yelp and Amazon reviews, so `make download-data` rebuilds it, and every
other left-out file, from its pinned original (`NO_REBUILD=1` skips that).

`data/` and `models/` are gitignored. `make data` (Kev's suites and manifests), `make data-raw-*`, `make normalize` and
`make clean-data` rebuild `data/` from `locks/`, which is committed with `reports/`. To skip the rebuild, `make
upload-data DATA_REPO=<org>/<name>` puts the whole `data/` (about 6 GB) in a private Hub dataset with a sha256 manifest,
and `make download-data DATA_REPO=<org>/<name>[@commit]` restores it, checking every file against that manifest and
every pinned file against `locks/`.

## Backends

| Machine | Backend | Precision | Runtime |
|---|---|---|---|
| Apple Silicon | `mlx` | bf16 | mlx-lm, Metal |
| Linux + NVIDIA GPU | `cuda` | bf16 | PyTorch + transformers, flash-linear-attention kernels |
| anything else | `cpu` | fp32 | PyTorch + transformers |

`uv sync` installs only the runtime for the platform it runs on. Set `DEN_BACKEND=mlx|cuda|cpu` to override the
choice. Both runtimes load only the model's text tower, found on the loaded model, so dense, MoE and vision-language
checkpoints all work. The hidden size is read from the config, and both return the same final hidden states.
On Qwen3.5-4B, MLX in bf16 and PyTorch in fp32 agree to a per-token cosine similarity of at least 0.999
(`make test-model`).
The Linux wheels are built for CUDA 13, which needs NVIDIA driver 580 or newer. `causal-conv1d` is optional and
speeds up the DeltaNet convolutions.

## Data audit

`make audit` parses every record against the `/v1/systemone` schema, then checks it for training. It cross-checks
the counts against Kev's manifests, then looks for duplicates, conflicting labels and leakage across train,
calibration, dev and test. It also measures label balance and token lengths, and opens every raw source file.
The full report is in `reports/data-audit.json`. It covers the Kev suites and all 69 normalized files.
Current result: **0 errors**. None of the normalized training files overlaps any calibration, dev or test file.

- **Clean:**
  - All 42,866 suite records are valid, and every count matches Kev's manifests.
  - No training question or state appears in any calibration, dev or test file.
  - Every raw source opens and has rows.
  - The longest training state is 7,329 tokens, under Kev's 7,552 limit.
- **Exact duplicates:** `train/core` has 420, mostly from Kev's rule-structure generator (its "relevant" and
  "irrelevant" variants sometimes render to the same text). They are kept, so the data stays byte-identical to
  Kev 1.0.
- **Unknowable probes:** `probes` and `calibration/heldout` repeat some questions with different labels. These are
  Kev's *unknowable* items, where the deciding fact is missing, so they are scored on confidence, not accuracy.
- **Not fully held out:** 18 of the 22 unknowable families in `probes` and `calibration/heldout` come from generators
  that stage 2 (`train/dates-unknowable`) trains on. Leave them out when fitting `T` on `calibration/heldout`.
  The `transfer` test is fully out of domain: none of its synthetic families is trained on.

## Normalized sources

`make normalize` turns every trainable and eval-only raw source (17 trainable, 9 eval-only) into the same canonical
record format as Kev's suites, in about 2 minutes. The output is deterministic: the same inputs always give
byte-identical files. Each output's counts, label distribution and sha256 go to `reports/normalize.json`, which
`make audit` checks.

- **Text:** NFC Unicode and tidy whitespace. Literal `\n` and `\"` escapes are undone (Yelp, AG News), HTML
  entities are decoded, including AG News' broken `#39;` and `quot;`, and `<br />` tags become newlines (IMDB,
  Aegis). Datasets that are pre-tokenized, like SST-5, are left as written.
- **Questions:** each source keeps its native label set. For the sources Kev used, the instructions are Kev's
  wording. Multiple-choice options get neutral keys (`opt_1`…). SciQ's options and Glaive's functions are put in a
  seeded order, so the answer's position carries no signal.
- **Cleaning:**
  - Unlabeled or invalid rows are rejected.
  - Aegis prompts that are REDACTED or that match AART are rejected, as Kev does.
  - Records whose identical question carries different labels are dropped. This mostly affects Glaive and Aegis.
  - A repeated wrong option collapses to one copy. If the correct answer itself is repeated, the item is dropped.
  - Every duplicate question is kept once, in the most held-out split it appears in.
- **Splits:** native splits are used where they exist, e.g. MultiNLI uses matched for dev and mismatched for test.
  BoolQ, CommonsenseQA and QASC have unlabeled tests, so their validation split becomes the test. A missing dev
  split is carved from train by a hash of the source text, so every question about one document lands in the same
  split. Dev and test are capped at 5,000 records each.
- **No leakage:** matches use both the rendered state and the raw-text provenance (`_meta.text_sha256`, the same
  hash Kev records).
  - Test drops anything in Kev's train, calibration or dev.
  - Dev drops anything in Kev's train or calibration, or in any test.
  - Train drops anything in any calibration, dev or test.
- **Not yet normalized:** the 14 breadth-v1 sources stay raw in `sources/eval/breadth/`. Rebuilding Kev's panel
  needs its own per-dataset mapping.

## Data layout

Files are organized by **role**, so the path says how a file may be used. The catalog refuses any entry whose path
doesn't match its role. Training code reads only `train/` and `sources/train/`.

```
data/
├── train/                  frozen training records, one file per Kev 1.0 stage
│   ├── core.jsonl              stage 1  12,576  public classification + policy pairs + rule structures
│   ├── dates-unknowable.jsonl  stage 2   1,425  date policies + deciding sentence removed
│   ├── documents.jsonl         stage 3   5,219  CFPB complaint narratives (up to ~7k tokens)
│   ├── skills.jsonl            stage 4   6,000  long policies, trade-offs, probability, multi-hop, dates, judging, abstention
│   └── devtools.jsonl          stage 4   5,320  code review, commit type, flaky tests, content safety
├── calibration/            fit the temperature only
│   ├── core.jsonl              in-distribution
│   └── heldout.jsonl           held-out sources (the honest choice)
│   └── sources/<skill>/<ds>.jsonl  normalized from sources/train (make normalize)
├── dev/                    model selection: core, documents, skills, devtools, transfer, probes
│   └── sources/…               normalized dev splits, trainable and eval-only sources
├── test/                   locked, read once per release candidate: same suites as dev/
│   └── sources/…               normalized test splits
├── manifests/              Kev's manifest for each suite (sources, revisions, counts)
└── sources/
    ├── train/              raw, may become training records
    │   ├── intent/             banking77 · massive-en
    │   ├── topic/              ag-news · trec · dbpedia14
    │   ├── reading/            boolq · multinli
    │   ├── sentiment/          sst5 · yelp-full · amazon-reviews · imdb
    │   ├── safety/             aegis · aart-filter
    │   ├── knowledge/          arc · openbookqa · commonsenseqa · qasc
    │   ├── tools/              glaive-function-calling
    │   ├── documents/          cfpb-complaints                          (raw-bulk)
    │   └── devtools/           commitpackft · codereviewer · flakeflagger (raw-bulk)
    └── eval/               raw, NEVER trained on
        ├── transfer/           qnli · paws · sciq · mmlu · mmlu-pro · emotion · tweeteval-offensive
        ├── devtools/           when2call · prompt-injection-deepset · prompt-injection-gandalf
        └── breadth/            Decision Index areas
            ├── knowledge/          musr · sata-bench · chessbench
            ├── language/           contractnli · hellaswag
            ├── retrieval/          clinc150 · sgd · bright
            ├── tools/              bfcl · toolret-queries · toolret-tools · apibank · routerbench
            └── arts/               humicroedit · cfcolor
```

### Download sets

| Set | Target | Size | Lands in |
|---|---|---|---|
| `suites` | `make data` | ~100 MB | `train/ calibration/ dev/ test/ manifests/` |
| `raw-train` | `make data-raw-train` | ~0.8 GB | `sources/train/{intent,topic,reading,sentiment,safety}` |
| `raw-new` | `make data-raw-new` | ~0.3 GB | `sources/train/{knowledge,intent,tools}` |
| `raw-eval` | `make data-raw-eval` | ~0.6 GB | `sources/eval/` |
| `raw-bulk` | `make data-raw-bulk` | ~5 GB | `sources/train/{documents,devtools}` |

### Names → Kev's names

| Ours | Kev |
|---|---|
| `core` | `decision-v7` |
| `dates-unknowable` | `night2/dates_unknowable` |
| `documents` | `documents-v1` |
| `skills` | `hard-v1` |
| `devtools` | `devtools-v1` |
| `transfer` | `transfer-v4` (its test is the headline number) |
| `probes` | `transfer-v9` (date deadlines, MMLU, MMLU-Pro) |
| `heldout` | `round3/transfer-r3` |

Kev's recipe, for reference: LoRA r16/α32 on attention, MLP and DeltaNet projections. LR 5e-5 one-cycle for stage 1,
then 2e-5. Effective batch 8, bf16 autocast, states of at most 7,552 tokens. Every later stage replays `core` records
(2,000, 2,000 and 4,000). Option order is shuffled, and none-of-the-above options, distractors and minimal pairs are
mixed in.

Some repositories can never be trained on, and the catalog refuses to mark any of them trainable:
`tasksource/*` (backs Kev's private tasksource-heldout-v1), CUAD (longdoc-v1), SNLI and Rotten Tomatoes (Kev excluded
them), and MMLU and MMLU-Pro. Kev's breadth-v1 suite is private, but `sources/eval/breadth/` holds every pinned source
needed to rebuild it.

## Targets: Kev-4B (Kev 1.0), the default size

| Panel | Kev-4B | Jev |
|---|---|---|
| `test/transfer`, accuracy / Brier (headline) | **0.838** / 0.224 | – |
| `dev/core` / `test/core` | 0.873 / 0.865 | 0.845 / – |
| `test/skills` | 0.803 | – |
| `test/devtools` | 0.756 | – |
| `test/documents` | 0.903 | – |
| breadth-v1 test (14 held-out datasets) | 0.690 | 0.757 |
| MMLU-Pro (`dev/probes`) | 0.565 | 0.840 |
| Dates, `deadline` policy (`dev/probes`) | 0.65 | 0.95 |

Its weak spots are knowledge, dates, held-out breadth (tools, retrieval) and long states beyond 8k tokens. The
`raw-new` set targets the first three, together with the date and skill generators that come later.

## Layout

```
den/                  the package, one module per concern (flat, like kev/)
├── api.py                    the record format: Kev's /v1/systemone request + labels, `render`
├── pins.py                   every model and dataset, pinned to a commit and a sha256
├── catalog.py                catalog types and rules (pins, roles, contamination)
├── fetch.py                  verified downloads and lock files
├── sources.py                per-source mappings from raw rows to records
├── text.py                   text repair shared by normalize and clean (NFC, escapes, entities)
├── normalize.py              raw sources -> canonical, leakage-free records (`make normalize`)
├── clean.py                  cleaned, deduplicated training copies in data/clean/ (`make clean-data`)
├── audit.py                  dataset checks (`make audit`)
├── prompt.py                 records -> token ids and pointer positions, option shuffling
├── model.py                  LoRA backbone (Unsloth, or PEFT for checks) and the pointer head
├── calibrate.py              temperature fit on the calibration split
├── train.py                  training: data, Trainer, run.json (`make train`)
├── evaluate.py               load a run; `den evaluate` and `den predict`
├── publish.py                `den publish`: model card, stages, data, logs -> private Hub upload
├── serve.py                  `den serve`: POST /v1/systemone (Kev's API)
├── device.py                 backend choice (mlx | cuda | cpu) and the backbone interface
├── mlx_model.py              MLX backbone (Apple Silicon)
├── torch_model.py            PyTorch backbone (CUDA, CPU)
└── cli.py                    `den models | model | list | data | verify | normalize | clean | audit | train | env | smoke`
tests/                      one test file per module it covers
data/                       suites and normalized sources, by role (see Data layout)
locks/                      sha256 of every placed file
reports/                    audit, normalize and clean results
runs/                       training outputs (gitignored)
```

## Next

Encoder (`state + question + options` → token ids and option positions) → pointer head + LoRA on both backends → stage-1
training on `train/core.jsonl` → temperature fit → evaluation harness on the locked suites → `/v1/systemone` server.
