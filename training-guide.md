# Training guide: den v1 on a cloud H100, start to finish

A step-by-step path for you, the person at the keyboard. Each step says what to run, what you should see, and what to
do if you don't. Tick the boxes as you go. `instructions.md` is the same plan written for an agent, with more detail
on each gate; this guide follows it.

**When you're stuck,** send me: the step number (e.g. "Step 4.3"), the command you ran, and the last ~40 lines of its
output (`tail -40 <log>`). That's usually enough to tell what went wrong.

---

## The plan at a glance

| Part | What | Time | GPU billed? |
|---|---|---|---|
| A | Prepare on your Mac: commit, push, choose names | 10 min | no |
| B | Rent the H100 and check it | 5 min | yes, from here on |
| C | Install the code and Unsloth | 10–15 min | yes |
| D | Download the model and build the data | 30–40 min | yes |
| E | Safety checks and a 20-step timing run | 25 min | yes |
| F | Round 1: Kev-4B's four stages | 1.6–2.5 h | yes |
| G | Safety copy to the Hub | 5–10 min | yes |
| H | Round 2: public sources + replay (optional) | 50–70 min | yes |
| I | Evaluate, choose the round, test once | 30–40 min | yes |
| J | Release v1, prove it works, save the evidence | 15 min | yes |
| K | Shut the machine down | 1 min | stops billing |

Budget about **5–6 GPU hours** for everything, **3.5–4** if you skip round 2. Part E tells you which fits.

---

## Part A: on your Mac, before renting anything

### A.1 Commit and push your work
The GPU box gets the code from GitHub (`ramput-labs/den`), so everything must be pushed first.

```bash
cd ~/ramput-labs/den
make check                     # must end with "... passed" and "claims: 0 of 0 trace to their evidence"
git status                     # data/ is gitignored, so its files show as deleted from git: that's expected
git checkout -b h100-v1
git add -A
git commit -m "Prepare v1 training: releases, evidence, licences, breadth"
git push -u origin h100-v1
```

- [ ] `make check` passed
- [ ] pushed branch `h100-v1`

### A.2 Get your tokens and pick names
- [ ] **Hugging Face token** with *write* access: <https://huggingface.co/settings/tokens>
- [ ] **GitHub token** (read is enough to clone; *write* if you want to push results back from the box)
- [ ] Model repo name. It is created **private** by the first upload. Suggested: `a1i6ek/den-qwen3.5-4b`
- [ ] Your **budget** in hours, e.g. 6

Write these down; you'll paste them in Part C.

---

## Part B: rent the H100 and check it

Rent **1× H100 80 GB** with Ubuntu 22.04/24.04, **NVIDIA driver ≥ 580** and **≥ 120 GB disk**. SSH in.

### B.1 Is the machine right?
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h ~ | tail -1
```
You should see `NVIDIA H100 80GB HBM3, 81559 MiB, 580.xx` (or newer) and at least 80 GB free.

- **Driver below 580?** Stop here: the locked PyTorch needs CUDA 13. Destroy the machine and rent one with a newer image. Don't
  try to install drivers.

### B.2 Work inside tmux (always)
An SSH drop then can't kill a training run.
```bash
tmux new -s den           # later, after reconnecting: tmux attach -t den
```

- [ ] H100, 80 GB, driver ≥ 580, disk OK, inside tmux

---

## Part C: code and environment

### C.1 Your variables (paste, with your values)
```bash
export GITHUB_TOKEN=ghp_...
export HF_TOKEN=hf_...
export HF_REPO=a1i6ek/den-qwen3.5-4b
export BUDGET_HOURS=6
echo "export HF_REPO=$HF_REPO BUDGET_HOURS=$BUDGET_HOURS" >> ~/.bashrc
```

### C.2 Install uv and the code
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone --branch h100-v1 https://$GITHUB_TOKEN@github.com/ramput-labs/den.git && cd den
make setup
uv run python -m scripts.check_env      # torch 2.14.1, cuda 13.x, available True, NVIDIA H100
make check                              # must pass
```

### C.3 Install Unsloth without letting it change torch
```bash
make train-setup
export UV_NO_SYNC=1 && echo 'export UV_NO_SYNC=1' >> ~/.bashrc    # from now on uv never re-syncs
uv run python -m scripts.check_env --unsloth                      # must end without a BAD line, bf16 True
make doctor ARGS="--require cuda" | tee ~/doctor.log              # every line ok
```

- **`make train-setup` fails to resolve?** Run `uv pip install --no-deps unsloth unsloth_zoo`, then install whatever import
  fails, one at a time, with `uv pip install -c .unsloth-pins.txt <package>`.
- **`check_env --unsloth` says transformers < 5.17?** Run `uv pip install "transformers==$(grep -i '^transformers==' .unsloth-pins.txt | cut -d= -f3)"`.
- **Never install `vllm` or `bitsandbytes`.**

- [ ] `make check` passed, `check_env --unsloth` clean, `make doctor` all ok

Start a progress log, one line per finished part:
```bash
echo "$(date -u +%H:%M) C done: env ok" >> ~/progress.log
```

---

## Part D: model and data

```bash
uvx hf auth login --token "$HF_TOKEN"
make model MODEL=qwen3.5-4b                         # 9.3 GB, every shard sha256-checked
make data                                           # Kev's suites (~100 MB)
make data-raw-train data-raw-new data-raw-eval      # public sources (~1.7 GB)
make breadth                                        # Kev's breadth-v1, rebuilt byte for byte (~10 min)
make normalize                                      # ~2 min
make clean-data
make licences                                       # licence notices into data/ (nothing is uploaded)
```

Your public dataset (`a1i6ek/den-datasets`) leaves out files whose licences forbid redistribution, so building from
the pinned originals, as above, is the complete path. Every download is checked against `locks/`.

### D.1 The data gate: the numbers must match exactly
```bash
make train-check
```
must print:
```
train      40352 records    40352 questions  tokens median 176 p99 4455 max 7390 total 17.6M  skipped 0
dev         1468 records     1468 questions  tokens median 111 p99 788 max 858 total 0.3M  skipped 0
calib       1148 records     1148 questions  tokens median 120 p99 774 max 815 total 0.2M  skipped 0
```
and
```bash
uv run den train --dry-run --data --sources-cap 1500 --replay 1500 \
  --replay-from train/core.jsonl train/dates-unknowable.jsonl train/documents.jsonl train/skills.jsonl train/devtools.jsonl | head -1
```
must print `train      37459 records    37459 questions  tokens median 118 p99 2481 max 6058 total 8.8M  skipped 0`.

- **Different numbers?** Stop and send me both outputs. Something in the code or data differs from what was checked.
- **`make breadth` says the sha256 differs?** Send me its last lines. Don't skip it: normalize relies on it.

- [ ] both counts match

---

## Part E: safety checks and timing (≈25 min)

### E.1 One real training step on the GPU (~2 min)
```bash
make test-cuda 2>&1 | tee test-cuda.log            # must show: 3 passed
```

### E.2 Can the model learn at all? (~10 min)
```bash
uv run den overfit --n 100 --steps 300 --out overfit-lora.json
uv run den train --data train/core.jsonl --overfit 100 --epochs 25 --lr 2e-4 --head-lr 1e-3 --batch 4 --accum 1 \
  --out runs/overfit 2>&1 | tee overfit-train.log
rm -rf runs/overfit
```
- `den overfit` must end with **PASS**. Note its `content_agreement` (near 1.0 is good).
- In `overfit-train.log` the last `epoch dev` line must show **acc ≥ 0.95**.

### E.3 Timing run: 20 steps of every stage
```bash
make train-kev RUNS=runs/timing ARGS="--max-steps 20" 2>&1 | tee runs-timing.log
make train-round2 RUNS=runs/timing ROUND2=runs/timing/round2 ARGS="--max-steps 20" 2>&1 | tee -a runs-timing.log
uv run python -m scripts.check_merged runs/timing/4-skills runs/timing/round2       # ok for both
uv run den predict --run runs/timing/round2 < /dev/null                            # no error
uv run python -m scripts.estimate_time --budget-hours "$BUDGET_HOURS" --spent-minutes 75
rm -rf runs/timing
```
While stages 3–4 run, open a second tmux pane (`Ctrl-b %`) and watch memory: `watch -n 5 nvidia-smi`.

Check in `runs-timing.log`:
- every stage has a `setup` line with `engine unsloth  dtype bf16 ... lora_modules 248 ... trainable_dtype float32`
- no `nan` loss, no `CUDA out of memory`

`estimate_time` ends with a decision:
- **run both rounds:** do F, G, H.
- **run round 1 only:** do F and G, skip H.
- **stop and report:** send me the output before spending more.

- **Out of memory in a stage?** Halve that stage's `--batch` and double its `--accum` in the `Makefile` (same effective batch),
  rerun the timing run, and tell me.

- [ ] test-cuda 3 passed, overfit PASS, timing clean, decision noted

```bash
echo "$(date -u +%H:%M) E done: <decision>" >> ~/progress.log
```

---

## Part F: round 1, Kev-4B's four stages (1.6–2.5 h)

```bash
make train-kev ARGS="--max-minutes 100 --save-steps 500" 2>&1 | tee runs-train.log
```

Check every ~15 minutes (not more often):
```bash
grep -E "^setup|^init|dev|calibration|saved|stopping|nan|Error" runs-train.log | tail -20
```

What good looks like:
- each stage ends with `epoch dev ...`, `calibration  T ...` (T between 0.5 and 3), `dev at T ...` and `saved runs/kev-recipe/<stage>`
- **1-base:** dev nll < 1.0 and acc ≥ 0.63 (chance: 0.31)
- the last stage also prints `saved runs/kev-recipe/4-skills/merged  (248 merged weights)` and the integrity lines all `ok`

**If the machine or SSH dies mid-stage:** reconnect, `tmux attach -t den` (or a new tmux), and rerun **only that stage's
line** from the `Makefile` with `--resume latest` added. Finished stages are kept.

- [ ] all four stages saved, merge ok

---

## Part G: safety copy to the Hub (5–10 min)

If the machine is lost after this, the trained model isn't.
```bash
uv run den publish --run runs/kev-recipe/4-skills --repo "$HF_REPO" --logs runs-timing.log runs-train.log
```
It ends with the repo URL and a file count. If it's interrupted, run the same command again: it resumes. This is
not a release yet: that's Part J.

- [ ] safety copy uploaded

---

## Part H: round 2 (skip if Part E said so)

```bash
make train-round2 ARGS="--max-minutes 90 --save-steps 500" 2>&1 | tee runs-round2.log
```
Good: `epoch dev ...`, `calibration  T ...` (0.5–3), `saved runs/round2` and `saved runs/round2/merged  (248 merged weights)`.

- [ ] round 2 saved (or skipped)

---

## Part I: evaluate, choose, test once

### I.1 Evaluate each round on dev (integrity, metrics, augmentations, option-order checks)
```bash
make post-train RUN=runs/kev-recipe/4-skills 2>&1 | tee eval-round1.log
make post-train RUN=runs/round2 2>&1 | tee eval-round2.log          # skip if no round 2
```

### I.2 Choose the round to ship (dev only)
```bash
uv run den compare runs/kev-recipe/4-skills runs/round2 \
  --files dev/core.jsonl dev/documents.jsonl dev/skills.jsonl dev/devtools.jsonl dev/transfer.jsonl
uv run den compare --paired runs/kev-recipe/4-skills runs/round2     # is the difference beyond noise?
```
`compare` ends with `ship: <run>`. Set it:
```bash
export SHIP=runs/round2          # or runs/kev-recipe/4-skills, whatever compare printed
```
Without round 2, `SHIP=runs/kev-recipe/4-skills`.

### I.3 The locked test: once, for the shipped run only
```bash
make final-test RUN=$SHIP 2>&1 | tee test.log
```
**Read the test set once.** The command refuses a second read. Never run it for the other round.

- [ ] SHIP chosen, test read once

```bash
echo "$(date -u +%H:%M) I done: ship $SHIP" >> ~/progress.log
```

---

## Part J: release v1, prove it, keep the evidence

### J.1 Release
```bash
make release VERSION=v1 RUN=$SHIP REPO="$HF_REPO" ARGS="--logs runs-timing.log runs-train.log runs-round2.log"
```
This publishes the run, tags that Hub commit `v1` and writes `releases/v1.json` (with the sha256 of every file each
stage trained on, so the exact data is recorded even though it was built here rather than downloaded). (Drop `runs-round2.log` if you
skipped round 2.) Try `ARGS="--dry-run"` first if you want to see every check pass before uploading.

### J.2 Prove the published model answers
```bash
REQ='{"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.", "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}}, "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"}}}'
echo "$REQ" | uv run den predict --run release:v1
```
You should get JSON with `answers.department.probabilities` summing to 1.

### J.3 Save the evidence to git
`releases/v1.json` and `reports/runs/` (every evaluation's numbers and per-question answers) are what v2 will be
compared against. Push them:
```bash
git add releases/ reports/runs/
git commit -m "Release v1: record and evaluation evidence"
git push
```
No write token on the box? Copy them to your Mac instead (run this **on your Mac**):
```bash
scp -r <user>@<gpu-host>:den/releases <user>@<gpu-host>:den/reports/runs ~/ramput-labs/den/
```

- [ ] v1 released, predict works, evidence pushed or copied

---

## Part K: shut it down

- [ ] `releases/v1.json` and `reports/runs/` are safe (pushed or copied)
- [ ] the model is on the Hub (`https://huggingface.co/$HF_REPO`, tag `v1`)
- [ ] **destroy the instance** in your cloud console. Stopping isn't always enough: some providers bill stopped disks

```bash
echo "$(date -u +%H:%M) done: v1 released" >> ~/progress.log && cat ~/progress.log
```
Send me `~/progress.log`, `test.log` and the compare output, and I'll help you read the results against Kev-4B.

---

## Later: v2

Once v1 is released and you've added new data (new pins plus `make normalize clean-data`):
```bash
make train-next FROM=v1 OUT=runs/v2 DATA="train/<new>.jsonl"
make post-train RUN=runs/v2 && make final-test RUN=runs/v2
make release VERSION=v2 RUN=runs/v2 REPO="$HF_REPO" PARENT=v1
```
The release refuses v2 if any dev file falls more than a point below v1, and records the question-by-question
comparison.

---

## Quick fixes

| Symptom | Fix |
|---|---|
| `uv run` hangs at 0% CPU | another uv process holds the lock: use `.venv/bin/den ...` |
| a package vanished after a `uv run` | `UV_NO_SYNC=1` wasn't set: redo C.3 |
| `... falling back to its reference PyTorch implementation` | flash-linear-attention is missing (many times slower): send me the log before training |
| `CUDA out of memory` | halve that stage's `--batch`, double its `--accum` in the `Makefile` |
| SSH dropped | `tmux attach -t den`; training kept running |
| instance died mid-stage | rerun that stage's `Makefile` line with `--resume latest` |
| `merge: replaced N of 248` | the adapter is fine; send me the log, don't retrain |
| an upload stopped | run the same command again: uploads resume |
| `has already read [...]: the locked test set is read once` | working as intended: don't try to re-read test |
