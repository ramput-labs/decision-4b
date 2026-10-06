# Fine-tuning den on a cloud H100: runbook for the cloud agent

You are the cloud agent on a rented H100. The GPU is billed by the hour, so the goal is a clean run with no wasted
time. You will **validate the setup, train Qwen3.5-4B in two rounds, pick the better round on dev, test it once,
upload the model with its adapters, data and logs to the Hugging Face Hub, and prove that the uploaded model answers
over HTTP**. The person then downloads it, tests it against Kev, and decides whether it goes to production.

- **Round 1** is Kev-4B's own four-stage recipe (`make train-kev`).
- **Round 2** continues from round 1 on 17 public datasets Kev-4B never trained on, replaying all of Kev's suites so
  nothing is forgotten (`make train-round2`). It aims past Kev on breadth.

Follow the phases in order. Every phase ends in a **gate**. If a gate fails, stop, fix only what the gate names, and do
not move on until it passes. Don't improvise new hyperparameters or "quick experiments". They cost money, and the
recipe is fixed.

Read `CLAUDE.md` in the repo before starting. Its hard rules apply here too: never train or tune on `test/`, fit the
temperature only on `calibration/`, choose between runs on `dev/` only, and never edit pinned data or `locks/`.

## Inputs the person gives you

| Variable | Meaning | Example |
|---|---|---|
| `REPO_URL` + `BRANCH` | the code, already pushed | `https://github.com/ramput-labs/den.git`, `h100-finetune` |
| `GITHUB_TOKEN` | read access to the repo, if it is private | |
| `HF_TOKEN` | Hugging Face token with **write** access | |
| `HF_REPO` | where the model goes (created **private**) | `ramput-labs/den-qwen3.5-4b` |
| `BUDGET_HOURS` | hard ceiling for the whole session | `5` |

If any input is missing, ask for it before renting time.

Write a one-line progress note to `~/progress.log` at the end of each phase (time, phase, gate result, key numbers).
If the session dies, the next agent starts from that log.

## Time plan

| Phase | What | Expected |
|---|---|---|
| 0 | machine checks | 2 min |
| 1 | code, environment, Unsloth | 10 min |
| 2 | model, Kev's suites, public sources (download and normalize) | 15–20 min |
| 3 | overfit gates, then the timing run: both rounds, 20 steps per stage | 25 min |
| 4 | round 1: Kev-4B's four stages (23.3M tokens) | measured in phase 3; expect 1.6–2.5 h |
| 5 | upload round 1 (safety copy) | 5–10 min |
| 6 | round 2: public sources + replay (8.8M tokens) | measured in phase 3; expect 50–70 min |
| 7 | evaluate both rounds on dev, choose one | 20 min |
| 8 | test the chosen run once, publish everything | 15 min |
| 9 | prove the uploaded model works (CLI and HTTP) | 5 min |
| 10 | baselines on the same protocol (A always; C and B with budget) | 5 min – 2 h |

Round 2 is the part that gets skipped if time is short (phase 3 decides). Round 1 alone is a complete model.

---

## Phase 0: is this machine usable? (2 min)

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h ~ | tail -1
nproc; free -g | head -2
```

**Gate 0:**
- GPU name contains `H100`, with about 80 GB of memory.
- **Driver ≥ 580.** The locked PyTorch is the CUDA 13 build, which does not run on older drivers. If the driver is
  older, stop and tell the person to pick another image or host. Do not try to install drivers.
- At least **80 GB free disk**: base 9.3 GB, merged weights 9 GB, venv about 15 GB, plus uv cache and headroom.

## Phase 1: code and environment (10 min)

Run everything inside `tmux`, so an SSH drop never kills a run:

```bash
tmux new -s s1          # reattach later with: tmux attach -t s1
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone --branch "$BRANCH" "$REPO_URL" den && cd den   # private repo: https://$GITHUB_TOKEN@github.com/...
make setup                      # uv sync: Python 3.12, torch 2.14.1 (CUDA 13), transformers 5.x, flash-linear-attention
uv run python -m scripts.check_env      # torch, CUDA and the GPU; exits 1 without CUDA
make check                      # ruff, mypy, unit tests: must pass before any GPU time is spent
```

Install Unsloth **without letting it replace torch or transformers**. `make train-setup` freezes what the lockfile
installed into `.unsloth-pins.txt` and installs Unsloth under that constraint:

```bash
make train-setup
cat .unsloth-pins.txt
```

- If the resolver refuses (for example, Unsloth caps `transformers` below what we need), install it without its
  dependency resolution, then add whatever imports fail, one by one, still under the same constraint:
  `uv pip install --no-deps unsloth unsloth_zoo && uv pip install -c .unsloth-pins.txt <missing packages>`.
- **Do not install `vllm`.** This model never generates text, and vllm is only for generation. Don't install
  `bitsandbytes` either: 4-bit QLoRA is not recommended for Qwen3.5, and we train in bf16.
- From here on, **never let uv re-sync**, because a plain `uv run` puts the lockfile back and can undo the install:

```bash
export UV_NO_SYNC=1             # every `uv run` and `make` target now uses the venv as is
echo 'export UV_NO_SYNC=1' >> ~/.bashrc
```

```bash
uv run python -m scripts.check_env --unsloth   # + unsloth, transformers (>= 5.17), peft, BF16; exits 1 on a problem
```

```bash
make doctor ARGS="--require cuda"        # versions, GPU, BF16, driver, Unsloth import, kernels, model files, disk
```

**Gate 1:**
- `make doctor ARGS="--require cuda"` exits 0: every check prints `ok`. Save its output to `~/progress.log`.
- `make check` passes.
- torch is still `2.14.1` with CUDA `13.x`, and `cuda.is_available()` is True.
- transformers is **≥ 5.17** (Qwen3.5 needs transformers v5).
- `import unsloth` works, and `bf16 True` is printed.

If transformers was downgraded, restore it with `uv pip install "transformers==<version in .unsloth-pins.txt>"` and
rerun the gate. If Unsloth then refuses to import, stop and report both versions. Do not try more combinations.

## Phase 2: model and data (15–20 min)

```bash
uvx hf auth login --token "$HF_TOKEN"     # faster, rate-limit-free downloads; also used by `den publish`
make model MODEL=qwen3.5-4b               # 9.3 GB, every shard sha256-checked against locks/
make data                                 # Kev's suites (data/ is not in git; ~100 MB, sha256-checked)
make data-raw-train data-raw-new data-raw-eval   # public sources for round 2 (~1.7 GB, sha256-checked)
make breadth                              # Kev's breadth-v1 eval panel, rebuilt byte for byte (~10 min); before normalize
make normalize                            # raw sources -> leakage-free train/dev/test records (~2 min, deterministic)
make clean-data                           # data/clean/: Kev's suites and the normalized sources
make train-check                          # tokenizes round 1's data; no GPU
uv run den train --dry-run --data --sources-cap 1500 --replay 1500 \
  --replay-from train/core.jsonl train/dates-unknowable.jsonl train/documents.jsonl train/skills.jsonl train/devtools.jsonl
```

If the person gave a `DATA_REPO` (a `make upload-data` copy), replace the `make data` through `make clean-data` lines
with `make download-data DATA_REPO="$DATA_REPO"` (after `make model`: breadth needs its tokenizer): the same `data/`,
every file checked against the copy's manifest and the pinned ones against `locks/`. The files the licences keep out
of the copy (Kev's `core` suite, Yelp, Amazon, ...) are rebuilt from their pinned originals in the same command
(~20 min). It must end with `ok, every file matches`.

**Gate 2:** `make train-check` prints exactly:

```
train      40352 records    40352 questions  tokens median 176 p99 4455 max 7390 total 17.6M  skipped 0
dev         1468 records     1468 questions  tokens median 111 p99 788 max 858 total 0.3M  skipped 0
calib       1148 records     1148 questions  tokens median 120 p99 774 max 815 total 0.2M  skipped 0
```

and the round-2 dry run's first line is
`train      37459 records    37459 questions  tokens median 118 p99 2481 max 6058 total 8.8M  skipped 0`.
Any other count means the code or data differs from what was validated. Stop and report the difference.

## The recipe

**Round 1: Kev-4B's four stages** (`make train-kev`, from Kev-4B's model card and Kev's `kev/train.py` and
`kev/data.py`). Each stage continues from the previous one's adapter and head (`--init-from`) and replays `core`:

| Stage | Data | Epochs | lr | Batch | Replay | Tokens |
|---|---|---|---|---|---|---|
| 1-base | `train/core` (Kev's `decision-v7`), 25% with none-of-the-above minimal pairs | 2 | 5e-5 | 4 × 2 | | 6.4M |
| 2-dates | `train/dates-unknowable` | 1 | 2e-5 | 4 × 2 | 2,000 core | 0.6M |
| 3-documents | `train/documents` (CFPB, states up to 7.4k tokens) | 1 | 2e-5 | 2 × 4 | 2,000 core | 6.3M |
| 4-skills | `train/skills` + `train/devtools` | 1 | 2e-5 | 2 × 4 | 4,000 core | 9.9M |

**Round 2** (`make train-round2`) starts from round 1's `4-skills`: 1,500 records from each of the 17 normalized
public sources (intent, topic, reading, sentiment, knowledge, safety, tools; records whose text is already in Kev's
suites are skipped), plus 1,500 replayed from each of Kev's five suites. 1 epoch, lr 2e-5, batch 4 × 2, 8.8M tokens.

Both rounds:
- **One row per question:** the state plus that one question, never the record's other questions (Kev's "one row
  per question"), so an answer doesn't depend on what else was asked.
- **Kev's augmentation:** options shuffled; a 10% chance the true option becomes "none of the above", 12% a none
  option added as wrong, 15% an irrelevant distractor.
- **LoRA:** rank 16 (alpha 32, dropout 0) on all 248 text-decoder projections, adapters in fp32 over a bf16 base, TF32
  matmuls, the head at the same learning rate, 10% warmup then cosine.
- **Validation while training:** a fixed 600-question dev sample every 400 steps (`step N dev(600) nll .. acc ..`),
  all of the stage's dev files at each epoch's end (`epoch dev ...`). The curve is saved in each stage's `run.json`
  (`dev_history`). The test set is never touched while training.
- **Calibration:** each stage fits T on `calibration/core`.
- **Merge:** the last stage of each round writes `merged/`, the base checkpoint's own files with all 248 adapted
  weights replaced by W + scale × B·A.

Runs land in `runs/kev-recipe/{1-base,2-dates,3-documents,4-skills}` and `runs/round2`.

## Phase 3: overfit gates and timing run (25 min)

Run both rounds for 20 optimizer steps per stage. This tests every failure point before the paid hours: memory
(stages 3 and 4 hold the longest states; the length-grouped sampler puts each stage's longest batch first), speed,
validation, calibration, the stage handoffs, the source sampling and both merges.

First the **CUDA smoke test**, the real training path for one step, in about 2 minutes:

```bash
make test-cuda 2>&1 | tee test-cuda.log
```

It must show **3 passed**:
1. variable-K padding on CUDA under BF16 autocast;
2. a toy LoRA + head step on CUDA;
3. `test_unsloth_qwen_lora_pointer_step`: Unsloth loads Qwen3.5-4B with the base in bf16, LoRA on exactly 248 text
   projections in fp32, no vision modules. One forward/backward gives finite logits and loss for K = 2, 3 and 5,
   nonzero LoRA-B and head gradients, no gradient on the base, and an optimizer step that moves both.

A failure here is the cheapest one to have: stop and report it.

Then the two **overfit gates**. A model that can't fit 100 examples has a bug (positions, labels, masking, loss,
gradients), and more training won't fix it. They take about 5–10 minutes:

```bash
uv run den overfit --n 100 --steps 300 --out overfit-lora.json      # on CUDA: Qwen + LoRA + head, backward through Qwen
uv run den train --data train/core.jsonl --overfit 100 --epochs 25 --lr 2e-4 --head-lr 1e-3 --batch 4 --accum 1 \
  --out runs/overfit 2>&1 | tee overfit-train.log          # the same through the real Trainer loop: ~3 min
```

- `den overfit` on CUDA runs `path: lora` (Unsloth) and first prints `lora   engine unsloth ... modules 248` and
  `step 1 gradients: finite, LoRA-B and head nonzero, frozen base untouched`. It must end with `PASS`: train
  accuracy ≥ 0.95, `deterministic: true`, and a final loss under half the first. Also report its
  `content_agreement`, the share of choice rows where reversing the options (re-read by Qwen) keeps the same option,
  and `replacement_lowers_probability`. Near 1.0 means the head reads option content. Near chance means it scores
  positions, so report it and stop.
- `den train --overfit 100`: the last `epoch dev` line (dev here *is* the 100 training rows) must show acc ≥ 0.95,
  and `loss` must fall steadily. If not, stop and report the log. Then `rm -rf runs/overfit`.

Then the timing run:

```bash
make train-kev RUNS=runs/timing ARGS="--max-steps 20" 2>&1 | tee runs-timing.log
make train-round2 RUNS=runs/timing ROUND2=runs/timing/round2 ARGS="--max-steps 20" 2>&1 | tee -a runs-timing.log
nvidia-smi --query-gpu=memory.used,memory.total --format=csv    # from a second tmux pane, during stages 3 and 4
```

**Gate 3:**
- Every stage prints a `setup` line: `engine unsloth  dtype bf16  device NVIDIA H100 ...  lora_modules 248 ...
  trainable_dtype float32`. A `bfloat16` there means the adapter updates are being rounded away: stop and report.
  Every stage after `1-base` also prints `init   from runs/timing/...`.
- Validation lines appear: with only 20 steps there is no `step 400` line, but each stage ends with `epoch dev ...`
  or, when `--max-steps` ends it first, with `dev at T ...`.
- No CUDA out-of-memory error. If a stage runs out of memory, halve its `--batch` and double its `--accum` on that
  stage's line in the `Makefile` (the effective batch stays 8), rerun, and keep the edit for the real run. Report it.
- The logged `loss` is finite (not `nan`) in every stage.
- Both rounds end with `saved .../merged  (248 merged weights)`, and
  `uv run python -m scripts.check_merged runs/timing/4-skills runs/timing/round2` prints `ok` for both (`qwen3_5`).
- `uv run den predict --run runs/timing/round2 < /dev/null` exits without error.
- Estimate both rounds from the measured `tokens_per_second` in each stage's `run.json`:

```bash
uv run python -m scripts.estimate_time --budget-hours "$BUDGET_HOURS" --spent-minutes <minutes spent so far>
```

  It prints both rounds' minutes (2 per stage for loading, validation and saving included) and the decision below.

  The estimate runs high, because 20 steps include kernel compilation and the longest batch. Then decide:
  - round 1 + round 2 + about 80 min (phases 0–3 already spent, 5, 7, 8, 9) **within** `BUDGET_HOURS`: run both.
  - only round 1 + 80 min fits: run round 1, skip phase 6, and say so in the report.
  - not even that: **stop and report the estimate**. Don't drop stages or data on your own.

```bash
rm -rf runs/timing
```

## Phase 4: round 1

```bash
make train-kev ARGS="--max-minutes 100 --save-steps 500" 2>&1 | tee runs-train.log
```

`--save-steps 500` writes `checkpoint-*` folders in each stage's directory. Each holds the adapter and head
(`trainable.safetensors`) and the optimizer, scheduler, RNG and step state, about 100 MB, with the last 2 kept. If
the instance dies mid-stage, rerun **that stage's line** from the `Makefile` with `--resume latest` added. It continues
from the last checkpoint: same weights, optimizer state and data position (resume is tested to give results
identical to an uninterrupted run). Augmentation draws restart, which is statistically equivalent. Finished stages
need nothing.

`--max-minutes 100` is a per-stage safety net; no stage should get close. If a cap is reached, that stage stops,
calibrates and saves, and the next continues from it; `run.json` records `stopped_by_max_minutes`. Report it.

Check on it every 15 minutes, not more often: `grep -E "^setup|^init|dev|calibration|saved|stopping" runs-train.log`.
The `step N dev(600)` accuracy should rise within each stage, or at least not fall by more than about 2 points.

**Gate 4:** each stage ends with `epoch dev ...`, `calibration  T ...`, `dev at T ...` and `saved runs/kev-recipe/<stage>`;
the last also prints `saved runs/kev-recipe/4-skills/merged  (248 merged weights)`. Check:
- **1-base** on `dev/core`: chance is acc 0.31, nll 1.48. Need nll < 1.0 and acc ≥ 0.63.
- **3-documents** and **4-skills**: accuracy on their own dev files is well above chance (`dev/documents` 0.16,
  `dev/skills` 0.28, `dev/devtools` 0.41), and `dev/core` hasn't dropped more than about 3 points from 1-base.
- Every T is between 0.5 and 3. T at 0.05 or 20 means the fit hit its limit: report it.

If any loss goes to `nan`, or 1-base misses its limits, stop and report the log. Don't retrain with new settings. A
finished stage can be restarted from: to rerun only stages 3–4, run their two lines from the `Makefile` by hand.

## Phase 5: upload round 1 now (5–10 min)

A safety copy, so a lost instance cannot lose the trained model. `publish` creates the repo **private**, writes the
model card, bundles the stages, data and logs into the run, uploads resumably, and checks the listing:

```bash
uv run den publish --run runs/kev-recipe/4-skills --repo "$HF_REPO" --logs runs-timing.log runs-train.log
```

**Gate 5:** `publish` prints the repo URL with a file count, and no error. If interrupted, rerun the same command: it
resumes. If phase 8 later ships round 2, its upload replaces these files on `main`. Round 1 stays in the repo's
history and under round 2's `stages/`.

## Phase 6: round 2 (skip if phase 3 said so)

```bash
make train-round2 ARGS="--max-minutes 90 --save-steps 500" 2>&1 | tee runs-round2.log
```

**Gate 6:** it ends with `epoch dev ...`, `calibration  T ...` (0.5 < T < 3), `dev at T ...`,
`saved runs/round2` and `saved runs/round2/merged  (248 merged weights)`. Its `dev/core`, `dev/documents`,
`dev/skills` and `dev/devtools` accuracy (in `epoch dev`) should be close to round 1's. Phase 7 judges that properly.

## Phase 7: evaluate both rounds on dev, choose one (20 min)

This loads each run's `merged/` through the same backbone code that serving uses, question by question:

```bash
DEV="dev/core.jsonl dev/documents.jsonl dev/skills.jsonl dev/devtools.jsonl dev/transfer.jsonl dev/probes.jsonl dev/breadth.jsonl dev/binding.jsonl dev/semif.jsonl"
SRC_DEV=$(cd data/clean && ls dev/sources/*/*.jsonl | tr '\n' ' ')
for RUN in runs/kev-recipe/4-skills runs/round2; do          # drop runs/round2 if round 2 was skipped
  uv run den evaluate --run $RUN $DEV
  uv run den evaluate --run $RUN --limit 500 $SRC_DEV       # 500 sampled records per source file, the same for both
done
uv run den compare runs/kev-recipe/4-skills runs/round2 --files dev/core.jsonl dev/documents.jsonl \
  dev/skills.jsonl dev/devtools.jsonl dev/transfer.jsonl $SRC_DEV
```

Then the shipped candidates' behavior on Kev's augmentations, on the same dev files. This shows whether they read
option content (minimal pairs) and handle "none of the above", not just whether they're accurate:

```bash
for RUN in runs/kev-recipe/4-skills runs/round2; do
  for AUG in pairs permute none-replace none-add distract; do
    uv run den evaluate --run $RUN --augment $AUG dev/core.jsonl dev/documents.jsonl dev/skills.jsonl
  done
done
```

`eval.json` then holds, per file, `accuracy_by_source` (per skill family: `hard_*` for the skills suite, per dataset
elsewhere), `accuracy_by_type`, `ece_by_type`, the `uncalibrated` numbers, and for `+pairs` the `pair_accuracy`
(both halves of a minimal pair right). Files that carry Kev's structure also get `robustness`, Kev's benchmark checks:
`clean` (headline numbers without the variants, Kev's `clean`), `permutation` (`flip_rate`: how often a permuted copy
changes the answer), `paired_flip` (contrastive pairs: `flip_rate` where the answer should change, `invariance_rate`
where it shouldn't) and `unknowable` (`share_at_0_9`: answered at >= 0.9 with the deciding evidence removed; lower is
better). `+permute` does the option-order check on any file by rotating every choice question's options. Question
isolation needs no check: every question is read with the state alone.

`breadth` (14 public datasets, Kev's breadth-v1, rebuilt byte for byte), `binding` (role binding and date arithmetic)
and `semif` (SemIf's 144 authored decisions + 108 perturbations) are eval-only and not part of the ship decision.
Kev-4B's published numbers: binding 0.943; semif 0.847 on the 144 clean rows (`robustness.clean`); breadth per dataset
in Kev's `runs/breadth-v1-report/report.md` (compare `accuracy_by_source` on single-question datasets; Kev scores
`sata_bench` and `bfcl` record by record).

`compare` prints accuracy per file for both runs and ends with `ship: <run>`. Then the same two runs question by
question, which says whether round 2's gain is beyond noise (95% intervals from resampling records):

```bash
uv run den compare --paired runs/kev-recipe/4-skills runs/round2
```

Every `den evaluate` above also wrote `reports/runs/<run>/` (each file's report and every question's probabilities).
Commit `reports/runs/` with the release record in phase 8: it is the evidence behind every number you report.

`compare` (the plain one) picks the shipped run. It picks round 2 only if its mean
accuracy is higher **and** it doesn't lose more than 1 point on any single file, so a breadth gain can't hide a
regression on Kev's own tasks. `dev/transfer` and the eval-only sources (`dev/sources/transfer/*`,
`dev/sources/devtools/*`) are never trained on by either round, so they measure out-of-domain transfer honestly; the
other `dev/sources` files are in-distribution for round 2 only. `probes` is left out of the choice: it is mostly
unknowable items. Call the shipped run `$SHIP`.

`make post-train RUN=<run>` runs `den check-run` and the dev and augmentation evaluations above in one go.

**Gate 7:** both runs have `eval.json` and an `integrity.json` with `"ok": true`, `compare` printed `ship: ...`, and you wrote the table to `~/progress.log`.

## Phase 8: test once, publish everything (15 min)

Read the locked test set **once**, for the shipped run only. Never repeat it, and never test the other run:

```bash
TEST="test/core.jsonl test/documents.jsonl test/skills.jsonl test/devtools.jsonl test/transfer.jsonl test/probes.jsonl test/breadth.jsonl"
SRC_TEST=$(cd data/clean && ls test/sources/*/*.jsonl | tr '\n' ' ')
uv run den evaluate --final --run "$SHIP" $TEST                       # or: make final-test RUN=$SHIP
uv run den evaluate --final --run "$SHIP" --limit 1000 $SRC_TEST     # 1,000 sampled records per source file
make release VERSION=v1 RUN="$SHIP" REPO="$HF_REPO" ARGS="--logs runs-timing.log runs-train.log runs-round2.log"
```

`make release` runs `den publish`, tags that Hub commit `v1`, and writes `releases/v1.json` (lineage, data hashes,
dev and test results, file hashes, and the paired comparison with its parent). Commit it with `reports/runs/`
(every evaluation's report and per-question rows) and push: the next version continues from `release:v1`, and its
release is compared with v1 question by question from that evidence.
If the data was uploaded with `make upload-data`, add `DATA_REPO=<org>/<name>@<commit or tag>` so v1 records it.

What the Hub repo then holds (`den publish` bundles it):

| Path | What |
|---|---|
| `merged/` | the shipped model: LoRA folded into bf16 weights, loads like the base checkpoint |
| `adapter_model.safetensors`, `adapter_config.json` | the LoRA alone, for use on top of `Qwen/Qwen3.5-4B-Base` |
| `head.safetensors`, `head.json` | the pointer head's weights, and its config (kind, size, backbone hidden size) and T |
| `run.json`, `training_config.json`, `eval.json` | how it was trained (loss log `train_history`, validation curve `dev_history`, `calibration_report`, `best`, GPU and peak memory) and measured (every eval entry has `read_at`) |
| `best/` | the lowest-dev-NLL checkpoint (adapter + head, its own T) when it isn't the final step; not merged, not served |
| `integrity.json` | the merged run's checks (head/adapter finite, merged = base layout with exactly 248 weights changed) and every model file's sha256; `publish` reruns it and refuses a failure |
| `baselines.json` | phase 10's A/B/C/D table (`den baselines`), shown on the card |
| `stages/<name>/` | every earlier stage: adapter, head, run.json, eval.json |
| `data/` | the exact training data of every stage (sampled files hold only the lines used), every dev, calibration and test file it was scored on, `data/MANIFEST.json` with sha256s |
| `logs/` | the training logs |
| `README.md` | model card: results table, stage table, how to use it |

`den evaluate --final` refuses a second read of a test file the run already has in `eval.json`: if a read fails
midway, the files already read are recorded, so rerun only the missing ones.

**Gate 8:** every test file prints `acc`, `nll`, `brier`, `ece` and `cov@5%`. `publish` prints the URL and file count, and the Hub page
shows the card with the results and stage tables. If `publish` stops on a missing file, read its message. If
interrupted, rerun the same command.

## Phase 9: prove the uploaded model works (5 min)

Start from nothing but the Hub. `hf:` downloads the run into the Hub cache, not into `runs/`. First the CLI:

```bash
REQ='{"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.", "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}}, "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"}, "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}}}'
echo "$REQ" | uv run den predict --run "hf:$HF_REPO"
```

Then over HTTP, the API the person will test with:

```bash
uv run den serve --run "hf:$HF_REPO" --port 8009 > serve.log 2>&1 &
SERVER=$!
until grep -q "den serving" serve.log || ! kill -0 $SERVER 2>/dev/null; do sleep 2; done
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d "$REQ"; echo
curl -s localhost:8009/v1/models; echo
kill $SERVER
```

**Gate 9:** both print a response with Kev's shape:
`{"model": ..., "answers": {"department": {"type": "choice", "choice": ..., "confidence": ..., "probabilities": {...}},
"escalate": {"type": "noul", "noul": ...}, "frustration": {"type": "score", "score": ..., ...}}, "usage": ..., "latency_ms": ...}`,
with each question's probabilities summing to 1.

## Phase 10: baselines (same protocol; after the ship decision)

The comparison that shows whether the pointer head and LoRA earn their keep. Each baseline is trained and evaluated
exactly like the shipped run, on the same files. Run them only now that `$SHIP` is chosen, so their test numbers can't
influence any choice:

| Mode | What | Command | Cost |
|---|---|---|---|
| A | Qwen3.5-4B zero-shot: its own answer-letter probabilities, T fitted on calibration | `uv run den train --head-kind letters --lora 0 --epochs 0 --out runs/base-a` | ~5 min |
| C | frozen Qwen + pointer head | `uv run den train --lora 0 --head-lr 1e-3 --batch 8 --accum 1 --out runs/base-c` | ~40 min |
| B | Qwen + LoRA answering with the letter (text-generation baseline) | `uv run den train --head-kind letters --batch 4 --accum 2 --out runs/base-b` | ~1.5 h |

Mode D is `$SHIP`. Run A always. Run C if 1 hour of budget remains, and B only if 2 hours remain. The letter
scorer can't address questions with more than 26 options (`banking77` in `core`), so those are skipped for A and B.
Their `eval.json` reports how many questions were scored. Evaluate each on the same files as `$SHIP`, test
included:

```bash
for RUN in runs/base-a runs/base-c runs/base-b; do
  [ -d $RUN ] && uv run den evaluate --final --run $RUN $TEST
done
```

Then put them side by side, into `$SHIP/baselines.json`, and republish so the card shows the table:

```bash
uv run den baselines D=$SHIP A=runs/base-a $( [ -d runs/base-c ] && echo C=runs/base-c ) \
  $( [ -d runs/base-b ] && echo B=runs/base-b ) --files $TEST
uv run den publish --run "$SHIP" --repo "$HF_REPO" --logs runs-timing.log runs-train.log runs-round2.log
```

Upload them under `baselines/`:
`uvx hf upload "$HF_REPO" runs/base-a baselines/base-a`, and the same for the others (no merged weights: `--merge` is off).

## Final report to the person

Put the shipped run's test numbers next to Kev-4B's published ones (its model card, test partitions read once, at its
temperature). Like-for-like rows have the same question counts as Kev's:

| Our file (questions) | Kev-4B | Comparable? |
|---|---|---|
| `test/documents` (936) | 0.903 | yes: documents-v1 test, same 936 |
| `test/skills` (1,088) | 0.803 | yes: hard-v1 test, same 1,088 |
| `test/devtools` (1,073) | 0.756 | yes: devtools-v1 test (Kev drops 2 duplicate-id questions) |
| `test/core` (1,440) | 0.865 | roughly: Kev reports 1,200 decision-v7 locked-test questions |
| `test/transfer` (764) | 0.838 | roughly: Kev's transfer-v4 locked test has 656 |

Add `test/breadth` (1,990 records; Kev-4B's breadth-v1 report has its per-dataset numbers) and the dev-only
`binding` and `semif` rows. Report each difference in percentage points, with the `ece` next to it. Don't claim parity from one number: say which
rows are within about 2 pp, and which are not. Add the `test/sources/*` rows as our breadth results (Kev publishes no
numbers on those files).

Then the baselines table, same test files: A (zero-shot Qwen), C (frozen Qwen + head), B (LoRA + letters, if run),
D (`$SHIP`) and Kev-4B's published numbers: accuracy, NLL, Brier, ECE, accuracy per type, score MAE/RPS, and
`ms_per_question`. Say plainly if D doesn't beat C, or C doesn't beat A: that is a finding, not a failure to hide.

Send one message containing:
1. the Hub link, which round shipped and why (`den compare`'s table), and that phase 9 passed
2. the overfit gates' results (train accuracy, `content_agreement`, `replacement_lowers_probability`)
3. the test table above, the baselines table, T, and the calibration report (`run.json`: dev before/after T)
4. the augmentation results (`pair_accuracy`, `none-replace`, `none-add`, `distract`) and per-skill accuracy
5. wall time per phase and total GPU hours, from `~/progress.log`
6. anything unusual: gate retries, memory fallbacks, version workarounds, a skipped round 2
7. how the person can test it themselves:

```bash
git clone https://github.com/ramput-labs/den && cd den && make setup
uvx hf auth login                                         # the repo is private
make serve RUN=hf:$HF_REPO                                # POST http://127.0.0.1:8009/v1/systemone
echo '{"state": ..., "questions": {...}}' | uv run den predict --run hf:$HF_REPO
```

Then remind the person to **shut the instance down**. Don't leave it idle.

## Troubleshooting (only what has been seen or is likely)

| Symptom | Fix |
|---|---|
| `uv run` hangs at 0% CPU | another uv process holds the lock; use `.venv/bin/den ...` |
| a package disappeared after `uv run` | `UV_NO_SYNC=1` was not set; reinstall Unsloth as in phase 1 and export it |
| `... falling back to its reference PyTorch implementation` | flash-linear-attention is missing; `uv sync` installs it on Linux. Without it, training is many times slower: fix this before phase 3 |
| CUDA OOM at a stage's first step | halve that stage's `--batch` and double its `--accum` in the `Makefile` (the effective batch stays 8) |
| `train/sources is missing` from round 2 | phase 2's `make data-raw-* normalize clean-data` didn't run or failed |
| `merge: replaced N of 248` | the adapter is fine; `uvx hf upload "$HF_REPO" <run>` without `merged/`, and report it. Don't retrain |
| upload interrupted | rerun the same `den publish` command; it resumes |
