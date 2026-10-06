# CLAUDE.md

den trains **System One decision models**: a Qwen backbone with a LoRA adapter and a **pointer head**. The model
reads a state plus typed questions and returns a calibrated probability over each question's options. It runs in one
forward pass and generates no text. The goal is to match, then beat, Kev 1.0 at each model size. README.md has the
full rules and data layout. This file covers what you need before changing anything.

## Commands

Cloud H100 fine-tuning runbook: `instructions.md` (phases with gates; read it before touching a GPU box).

```bash
make check          # ruff check + ruff format --check + mypy --strict + pytest. Run before saying work is done.
make train-check    # tokenize data/clean, print shape (no GPU; works on the Mac)
make clean-data     # rebuild data/clean/ (gitignored) from suites + normalized sources
make audit          # full data audit -> reports/data-audit.json
uv run pytest -q tests/test_train.py   # one file
```

- Use `uv run` for everything. Never use bare `python` or `pip`. Python 3.12, line length 120.
- `mypy --strict` covers `src` and `tests`, and ruff selects `E,F,I,B,UP,SIM,RUF,ANN`. New code needs full type annotations.
- Tests marked `model` load the real 4B weights and are deselected by default. `make test-model` runs them.
- Training (`make train`) needs Linux + an NVIDIA GPU + Unsloth (`make train-setup`). On the Mac, verify with
  `make check` and `make train-check`, and say plainly that GPU training was not run.
- LoRA is Unsloth (`--engine unsloth`, the default). `--engine peft` loads the same adapters with plain PEFT on
  any device and is for checks only. `--lora 0` is the frozen-backbone, head-only baseline.
- `uv run pytest -q -m model tests/test_train.py` (about 3 min, CPU) checks LoRA wiring on the real weights: 248
  adapted modules, none in the vision tower, tower is `Qwen3_5TextModel`, gradients reach the adapters and the head.

## Layout

One flat package, `den/`, at the repo root, like kev's `kev/`. One module per concern, no subpackages, and
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
- `publish.py`: `den publish` bundles `stages/`, `data/` (exact training data + every scored file, sha256
  manifest) and `logs/` into the run, writes the card, uploads it as a private Hub model (resumable), checks the listing. `predict`/`evaluate` accept `--run hf:<org>/<name>`.
- `device.py`, `mlx_model.py`, `torch_model.py`: MLX (Apple Silicon) and PyTorch (CUDA/CPU) inference backbones,
  which must return the same hidden states.

## Hard rules (do not break)

1. **Never train or tune on test.** `data/test/**` is read once per release candidate. The loader refuses `test/`
   paths. Keep that guard.
2. **Eval-only sources are never trained on** (`data/sources/eval/**` and anything they produce). The role of each
   path is enforced in `catalog.py` and its tests.
3. **Temperature `T` is fitted only on calibration files**, never on dev or test. By default, leave
   `calibration/heldout` out: 18 of its 22 unknowable families also appear in `train/dates-unknowable`.
4. **Do not edit pinned files.** Kev's suites in `data/{train,calibration,dev,test}/*.jsonl`, `data/manifests/`, and
   `locks/*.json` are sha256-pinned. Fix data by changing `clean.py`/`normalize.py`/`sources.py` and regenerating, not by hand.
5. **Determinism.** Normalize and clean outputs must be byte-identical across runs. Use seeded `random.Random`, not
   global randomness.
6. **One forward pass, no generation.** Scores come from hidden states at known token positions, never from LM logits.

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

- Training is two rounds: round 1 `make train-kev` (below), round 2 `make train-round2` (from round 1: public
  sources capped at `CAP` each + `CAP` replayed from each Kev suite). `den compare` picks the round to ship on dev
  only (mean accuracy, no file down more than 1 point); the test set is read once, for the shipped run.
- `--eval-steps N` validates on a fixed `--eval-max` dev sample during training; `run.json` keeps `dev_history` and
  `data_used` (every file, sha256, sampled lines), which `den publish` uses to upload the exact training data.
- The release recipe is Kev-4B's, `make train-kev`: four stages (core ×2 at 5e-5 with 25% none minimal pairs →
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
