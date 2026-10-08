# CLAUDE.md

den trains **System One decision models**: a Qwen backbone with a LoRA adapter and a **pointer head**. The model
reads a state plus typed questions and returns a calibrated probability over each question's options. It runs in one
forward pass and generates no text. The goal is to match, then beat, Kev 1.0 at each model size. README.md has the
full rules and data layout. This file covers what you need before changing anything.

## Commands

GPU work happens in two places, both driven from the `Makefile` (`make help` groups the targets):
- **Local RTX 3070 (8 GB):** `docs/local-gpu.md`. `make local-setup`, then `make local`: the whole pipeline on
  qwen3.5-0.8b (4B doesn't fit 8 GB in bf16), at most `LOCAL_N`=1000 records a file (`den train --limit`), runs in
  `runs/local/`. It must pass before any H100 time is rented.
- **Cloud H100:** `docs/h100-runbook.md` (phases with gates; read it before touching a GPU box), `docs/h100-guide.md`
  for a person. One target per phase: `h100-setup` (ends with `h100-doctor`), `h100-gates`, `h100-timing`,
  `h100-round1`, `h100-round2`, `h100-eval`; then `eval-test`, `release`. Both boxes' targets wrap the generic
  `train-round1`/`train-round2`/`eval-dev`, so a recipe change is made once.
- Once `.unsloth-pins.txt` exists (`make setup-gpu`), the Makefile exports `UV_NO_SYNC=1` for every target.

```bash
make check          # ruff check + ruff format --check + mypy --strict + pytest. Run before saying work is done.
make help           # every target, grouped; named <area>-<action> (data-*, train-*, eval-*, local-*, h100-*)
make data-download  # the only way to get data/ (DATA_REPO, default a1i6ek/den-datasets); rebuilds what licences left out
make data-upload DATA_REPO=<org>/<name>     # data/ (gitignored) -> Hub dataset + sha256 manifest; the repo must be named
make data-check     # tokenize data/clean, print shape (no GPU; works on the Mac)
make data-clean     # rebuild data/clean/ (gitignored) from suites + normalized sources
make data-audit     # full data audit -> reports/data-audit.json
uv run pytest -q tests/test_train.py   # one file
```

- Use `uv run` for everything. Never use bare `python` or `pip`. Python 3.12, line length 120.
- `mypy --strict` covers `src` and `tests`, and ruff selects `E,F,I,B,UP,SIM,RUF,ANN`. New code needs full type annotations.
- Tests marked `model` load the real 4B weights and are deselected by default. `make test-model` runs them.
- Training (`make train`) needs Linux + an NVIDIA GPU + Unsloth (`make setup-gpu`, then `export UV_NO_SYNC=1`). On the Mac, verify with
  `make check` and `make data-check`, and say plainly that GPU training was not run.
- LoRA is Unsloth (`--engine unsloth`, the default). `--engine peft` loads the same adapters with plain PEFT on
  any device and is for checks only. `--lora 0` is the frozen-backbone, head-only baseline.
- `uv run pytest -q -m model tests/test_train.py` (about 3 min, CPU) checks LoRA wiring on the real weights: 248
  adapted modules, none in the vision tower, tower is `Qwen3_5TextModel`, gradients reach the adapters and the head.

## Layout

One flat package, `den/`, at the repo root (plus `scripts/` for tools, below), like kev's `kev/`. One module per concern, no subpackages, and
`tests/test_<module>.py` beside it. Keep it flat: add a module rather than a subpackage.

- `api.py`: the record schema (`/v1/systemone` request plus labels). `render` must match Kev's `api.render` byte for
  byte.
- `pins.py`, `catalog.py`, `fetch.py`: every dataset/model pinned to a commit and a sha256, and the role of every path.
- `sources.py`, `text.py`, `normalize.py`: raw sources -> canonical, leakage-free train/dev/test under
  `data/*/sources/`.
- `clean.py`: writes cleaned, deduplicated copies to `data/clean/`. Training reads only `data/clean/`.
- `audit.py`: validity, duplicate, conflict and leakage checks.
- `prompt.py`: prompt layout and token positions (option `starts`/`closes`, question `finals`), and `shuffle_options`.
- `model.py`: `load_backbone` (Unsloth or PEFT LoRA), `text_tower`, `PointerHead` (Kev's q·k pointer plus span
  pooling, cross-option attention and a per-option prior), `question_loss`.
- `serve.py`: `den serve`, Kev's `POST /v1/systemone` over the standard library's HTTP server (`DEN_API_KEY` for a
  bearer key). Keep the response shape identical to Kev's: `answer()` is tested against Kev's README example.
- `metrics.py`: every evaluation number (`summarize`). `overfit.py`: `den overfit`.
- `calibrate.py`: `fit_temperature`. `train.py`: the data (`Shuffled`), a `transformers.Trainer` subclass, and the CLI
  flags (`den train ...`).
- `evaluate.py`: loads a trained run (`merged/` + `head.safetensors`/`head.json`) through `device.load`, so a run is scored by the same
  backbone code that serves it. `den evaluate` (acc/NLL/ECE, `--final` for test) and `den predict`.
- `evidence.py`: every `den evaluate` writes `reports/runs/<run>/<file>/report.json` + `rows.json` (each question's
  probabilities) and `provenance.json`, committed like Kev's runs/ (no weights). `den compare --paired A B` compares two
  runs question by question (`metrics.paired_bootstrap`, records resampled, 95% intervals). `reports/claims.json` ties
  printed numbers to that evidence; `scripts/verify_claims.py` (in `make check`) fails on any that drifts.
- `release.py`: `den release create|list|show` (`make release`, `make release-list`). Versions v1, v2, ...: a release
  publishes a finished run, tags the Hub commit with the version and writes `releases/<version>.json` (lineage, data
  hashes, dev/test results, file hashes, data repo commit). `release:<version>` resolves to `hf:<repo>@<version>` for
  `--init-from`, `--run`. A release refuses: an existing version, a failed integrity check, no locked-test results, a
  different base than its parent, or any dev file more than 1 point below its parent unless `--accept-regression`
  says why (recorded). The record names its evidence folder and holds the paired comparison with its parent.
- `licences.py`: `den licences` (`make data-licences`). Every source's licence, class (open / share-alike / non-commercial
  / unspecified / restricted) and the primary-source evidence for it; each file in `data/` takes its most restrictive
  source (`_meta.source`, or the raw item it sits under). Writes `data/README.md` (Hub card), `data/LICENSES.md`,
  `data/LICENSES/` (per-source pages, SPDX texts from `licences/`, upstream licence files) and a `SOURCE-LICENSE.md`
  beside every dataset (each raw source, each folder of suite/normalized/clean files): every file there, its sources,
  licence, class and whether the copy has it. A new pin or `_meta.source` must get an entry: `tests/test_licences.py` fails otherwise.
- `integrity.py`: `den check-run`, run after `--merge` and before every upload: head/adapter finite, `merged/` has
  the base's layout and files with exactly the adapted weights changed, sha256 of every model file -> `integrity.json`.
- `publish.py`: `den publish` bundles `stages/`, `data/` (exact training data + every scored file, sha256
  manifest) and `logs/` into the run, writes the card, uploads it as a private Hub model (resumable), checks the listing. `predict`/`evaluate` accept `--run hf:<org>/<name>`.
- `device.py`, `mlx_model.py`, `torch_model.py`: MLX (Apple Silicon) and PyTorch (CUDA/CPU) inference backbones,
  which must return the same hidden states.

`scripts/` (top level, beside `den/`) holds one-off and operational tools that use `den` but aren't the library.
Run them as `uv run python -m scripts.<name>`, test them in `tests/test_<name>.py` (or `test_scripts.py`), and never
import `scripts` from `den`. ruff and mypy cover it like `den/`.

- `breadth.py` (`make data-breadth`): Kev's breadth-v1 builder (his scripts/build_breadth_v1.py) ported to read our pinned
  raws. Writes `data/{dev,test}/breadth.jsonl` only if both match `pins.BREADTH_SHA256`; keep it byte-faithful
  (seeds, sorts, key order).
- `mirror.py` (`make data-upload` / `make data-download`): `data/` as a Hub dataset with a sha256 manifest (left-out
  files' hashes included, as `excluded_files`); pinned files are checked against `locks/` both ways. The download
  rebuilds the files the licences leave out, running only the steps (fetch by item, breadth, normalize, clean) whose
  files are missing or wrong, and checks the result against the uploader's hashes. No target fetches single sets:
  `uv run den data <set>` is for work on the data pipeline itself.
- `check_env.py`, `check_merged.py`, `estimate_time.py`: the runbook's phase 1 and phase 3 checks (docs/h100-runbook.md).

## Hard rules (do not break)

1. **Never train or tune on test.** `data/test/**` is read once per release candidate. The loader refuses `test/`
   paths. Keep that guard.
2. **Eval-only sources are never trained on** (`data/sources/eval/**` and anything they produce). The role of each
   path is enforced in `catalog.py` and its tests.
3. **Temperature `T` is fitted only on calibration files**, never on dev or test. By default, leave
   `calibration/heldout` out: 18 of its 22 unknowable families also appear in `train/dates-unknowable`.
4. **Never upload data the licences exclude.** `scripts/mirror.py` leaves out restricted sources (Yelp, Amazon
   reviews, raw files holding HellaSwag's wikiHow items) always and unlicensed ones from public copies; a repo that is
   already public always gets the public rules. Change a
   source's class in `den/licences.py` only with evidence from its own terms, and record it there.
5. **Do not edit pinned files.** Kev's suites in `data/{train,calibration,dev,test}/*.jsonl`, `data/manifests/`, and
   `locks/*.json` are sha256-pinned. Fix data by changing `clean.py`/`normalize.py`/`sources.py` and regenerating, not by hand.
6. **Determinism.** Normalize and clean outputs must be byte-identical across runs. Use seeded `random.Random`, not
   global randomness.
7. **One forward pass, no generation.** Scores come from hidden states at known token positions, never from LM logits.

## Pointer head and prompt invariants

- The prompt is plain text (`prompt.text(record)`), tokenized once like any string. Pointer positions come from the
  tokenizer's character offsets, so a server only needs `tokenizer(text(record))`. Don't go back to tokenizing segments
  separately: that produced ids a plain tokenizer call doesn't reproduce.
- Each option's span runs from the `-` to its closing `\n`. The question's final token is the `:` of `Answer:`. If you
  change the layout, update `tests/test_train.py`, which checks these positions.
- The head must stay **permutation-equivariant** over options: no position embeddings in the option-mixing block.
  `test_scores_follow_the_options_not_their_order` guards this.
- New head components should start as no-ops (zeroed residual or output weights), so training starts from Kev's
  scoring (`test_head_starts_as_kevs_pointer`).
- Shuffle only `choice` options. `score` levels are ordered, and `noul` is always `(no, yes)`.
- Keep `lora_dropout=0` and bf16 (no 4-bit) for Unsloth on Qwen3.5. `text_tower()` must return
  `Qwen3_5TextModel`, the module the serving backends run, not the vision-language wrapper.
- The head is exported Hugging Face-style: `head.safetensors` (weights) + `head.json` (format `den-pointer-head`,
  kind, dim, layers, heads, `hidden_size`, base, lora, engine, temperature). No pickle anywhere. `model.save_head` /
  `model.load_head` are the only reader and writer; `load_head` refuses a head whose `hidden_size` differs from the
  backbone's. Head kinds: `set` (ours, default) and `pointer` (Kev's plain q·k), chosen with `--head-kind`.

## Known data facts

- `data/clean` passes validation: 0 parse errors, and no state or question overlap from train into calibration,
  dev or test.
- `documents` (CFPB) lists options in a fixed order that tracks the label: the answer is option 1 in 92%/64%/53% of
  its 3/4/5-option questions, in dev and test too. Option shuffling during training handles this. Don't trust
  documents dev/test accuracy as-is.
- Weaker shortcuts: in OpenBookQA the longest option is correct 40% of the time (chance is 25%), and in `when2call`
  option 1 is correct 35% of the time.
- Glaive's assistant often asks for missing arguments and calls a turn later: that request is `call` true, `ready`
  false, not a "no". Only a conversation that never calls is a "no", plus a seeded `GLAIVE_SWAP` (25%) of answered
  requests asked against an unrelated catalog (no shared content word), so negatives aren't only pizza and flights.
- `train/core` has 420 exact duplicates from Kev's generator. `clean.py` drops them in `data/clean` only.
- Identical questions with different labels in `probes`, `calibration/heldout` and `train/dates-unknowable` are
  deliberate *unknowable* items with uniform soft targets. Don't "fix" them.

## Experiment modes, ablations and gates

- Modes (all through `den train`, all evaluated by `den evaluate` on the same files):
  A zero-shot Qwen `--head-kind letters --lora 0 --epochs 0`; B LoRA + letters `--head-kind letters`;
  C frozen Qwen + head `--lora 0`; D LoRA + head (default). The letter scorer (`model.LetterHead`) is Qwen's own
  next-token logits for " A".." Z" (tied embedding rows stored in the head), over the `A) option` prompt: at most
  26 options. `tests/test_wiring.py` proves the four modes train different parameter sets.
- Ablations: `--lora-targets all|attention-mlp|attention` (248/128/32 modules), `--option-rep end|marker|mean|attn`,
  `--head-proj linear|mlp`, `--head-kind set|pointer`, `--ordinal-weight`. Defaults reproduce the main run.
- Gates before any long run: `den overfit` (head fits 100 real rows on cached hidden states; then determinism,
  option-permutation content agreement, option-replacement sensitivity) and `den train --overfit N` (LoRA + head
  through the real loop, dev = the same N rows). Never weaken a gate to make it pass.
- `den train --limit N` caps every file at N records (`--data` sampled seeded like replay, `--replay` and
  `--sources-cap` lowered to N; the lines land in `data_used`). It is the local rehearsal's size, not a recipe.
- `den probe` runs modes A and C on cached frozen-backbone features (MLX/CUDA/CPU): the fast way to compare them on a
  Mac, where `den train`'s torch path is far too slow (DeltaNet's PyTorch fallback). Same dev rows, same
  `fit_temperature`, same `metrics.summarize`; `--max-options 26` keeps the letter scorer's rows identical.
- `den serve` is single-threaded on purpose: MLX's GPU stream belongs to the thread that loaded the model.
- `den evaluate --augment pairs|none-replace|none-add|distract` scores Kev's augmentations; `pairs` reports
  `pair_accuracy`. `eval.json` also has `accuracy_by_source` (skill families), `ece_by_type`, `uncalibrated`.
- All metric math lives in `den/metrics.py` (`summarize`); tests check it against hand-computed values.

## Training: use the library, not a hand-written loop

- `den/trainer.py` holds `PointerTrainer`: length-grouped sampling, `param_groups` optimizer, and checkpoints of only
  the trainable parameters (`trainable.safetensors`) beside the Trainer's own optimizer/scheduler/RNG state.
  `--save-steps N` / `--resume latest|DIR`. `tests/test_wiring.py` proves a resumed run equals an uninterrupted one.
- The head runs under BF16 autocast in training: any constant written into an activation (fill values, masks) must
  use that tensor's own dtype (`torch.finfo(x.dtype)`). `test_heads_run_under_bf16_autocast` guards it.
- `den doctor [--require cuda|mlx]` is the readiness gate; `make test-cuda` (`pytest -m cuda`) runs the CUDA checks
  and one real Unsloth LoRA + head step. `den overfit --path auto` is LoRA through Qwen on CUDA, head-only elsewhere.

- Training is two rounds: round 1 `make train-round1` (below), round 2 `make train-round2` (from round 1: public
  sources capped at `CAP` each + `CAP` replayed from each Kev suite). `den compare` picks the round to ship on dev
  only (mean accuracy, no file down more than 1 point); the test set is read once, for the shipped run.
- `--eval-steps N` validates on a fixed `--eval-max` dev sample during training; `run.json` keeps `dev_history`,
  `train_history` (the Trainer's loss log), `calibration_report` (calibration split and dev, before/after T) and
  `data_used` (every file, sha256, sampled lines), which `den publish` uses to upload the exact training data.
- The lowest-dev-NLL checkpoint goes to `best/` (adapter + head with its own T; `run.json` `best`). It is mirrored to
  disk while training so `--resume` keeps it. The final weights still ship and merge: `best/` is for `--init-from`.
- `den evaluate` adds `robustness` (`metrics.robustness`, Kev's benchmark checks) from each file's `_meta`: clean-only
  numbers, permuted-variant flips, contrastive `paired_flip`, `unknowable` confidence; `--augment permute` rotates
  options on any file. Eval-only suites: `breadth` (dev+test, built by `make data-breadth` before `make data-normalize`, so
  normalize keeps its texts out of training), `binding` and `semif` (dev only, pinned from Kev's repo).
- `den evaluate --final` refuses to re-read a test file a run already has in `eval.json`. `den baselines D=.. A=..`
  writes the A/B/C/D table to the shipped run's `baselines.json`; the card shows it. `make eval-dev RUN=..` and
  `make eval-test RUN=..` run the after-training steps.
- Versions: v1 is the shipped run of the two rounds; each later version continues from the last release on new data
  with replay of Kev's suites, `make train-next FROM=v1 OUT=runs/v2 DATA="..."` (round 2's settings), then
  `make eval-dev`, `make eval-test`, `make release VERSION=v2 RUN=runs/v2 REPO=... PARENT=v1`. Tag the data copy it
  trained from too (`make data-upload ... TAG=data-v2`). Never edit a record in `releases/`.
- The release recipe is Kev-4B's, `make train-round1`: four stages (core ×2 at 5e-5 with 25% none minimal pairs →
  dates → documents → skills+devtools, at 2e-5 with 2k/2k/4k `core` replay), each `--init-from` the last. It is
  sourced from Kev's model card and `kev/train.py`/`kev/data.py`; change it only with evidence, and say so.
- `prompt.augment`/`prompt.none_pair` port Kev's augmentation (none-of-the-above 10%/12%, distractor 15%, minimal
  pairs). Soft-target questions are only shuffled, and score/noul are never touched. `--augment shuffle|none` for
  ablations.
- `--init-from` (`model.warm_start`) refuses a different base, LoRA rank or head shape.
- `--ordinal-weight` adds Kev's ranked probability score on score questions (default 0, as Kev-4B trained).
- `den evaluate` reports accuracy, NLL, Brier, ECE, coverage at 5% error, accuracy per question type, and score-level
  MAE/RPS. `den.load(run).predict(request)` is the Python API; it returns exactly what `den serve` returns.

- The `Trainer` (which Unsloth patches) owns bf16, gradient accumulation, clipping, the cosine schedule with warmup and
  length-grouped sampling. Only override what is specific to this model: `_get_train_sampler` (precomputed lengths)
  and `create_optimizer` (the head's own learning rate). Don't reintroduce manual autocast or step loops.
- `model.SystemOne.forward` returns `{"loss": ...}`. `save_strategy="no"`: the run saves the adapter, the head,
  `run.json` (setup, timings, versions, commit) and then, with `--merge`, merged weights. `model.save_merged` is
  ours on both engines: W + scale × B·A for each adapted module, written into a copy of the base checkpoint's own
  files, so `merged/` has the base's layout by construction. It refuses unless all 248 weights are replaced. Verified
  on real weights: merged (MLX) vs adapter (torch) hidden states, min cosine 0.9999.
- Unsloth (2026.9) loads Qwen3.5 through `FastModel` as the vision-language class and returns a processor, not a
  tokenizer. We never use it: prompts are tokenized by `prompt.encode`, and padding is masked (pad id 0). Keep
  `get_peft_model`'s `finetune_*` flags at their defaults: setting any to False routes our explicit target list through
  Unsloth's name-pattern filter, which can drop the DeltaNet projections.
- `transformers` is 5.x: `train_sampling_strategy="group_by_length"` and fractional `warmup_steps`. `group_by_length`
  and `warmup_ratio` no longer exist.
- If `uv run` hangs at 0% CPU, another uv process holds the lock (often an install running in a terminal). Use
  `.venv/bin/...` directly.

## Working style

- Match the surrounding code: short module docstrings that explain *why*, few comments, dense idiomatic Python, and
  `dataclass(frozen=True, slots=True)`.
- Add or update a test with every behavior change. Data-validity claims should come from running code over the files,
  not from reading the README.
- Commit only when asked. Branch first if you are on `main`.
