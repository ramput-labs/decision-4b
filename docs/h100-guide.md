# Cloud H100: full fine-tuning of den v1, start to finish

A step-by-step path for you, the person at the keyboard. Each step says what to run, what you should see, and what to
do if you don't. Tick the boxes as you go. `h100-runbook.md` is the same plan written for an agent, with more detail
on each gate; this guide follows it. Every step is one `make h100-*` target, so there is little to type.

**Before this:** the local rehearsal (`local-gpu.md`, `make local` on your RTX 3070) has passed. It runs the same
pipeline at small scale, so the H100 hours go to training, not to debugging.

**When you're stuck,** send me: the step (e.g. "E.2"), the command you ran, and the last ~40 lines of its output
(`tail -40 <log>`). That's usually enough to tell what went wrong.

---

## The plan at a glance

| Part | What | Command | Time | GPU billed? |
|---|---|---|---|---|
| A | Prepare on your Mac: check, commit, push, tokens | `make check` | 10 min | no |
| B | Rent the H100 and check it | `nvidia-smi` | 5 min | yes, from here on |
| C | Code, environment, model, data, doctor | `make h100-setup` | 40–50 min | yes |
| D | Gates: CUDA tests and overfit tests | `make h100-gates` | 15 min | yes |
| E | Timing run and the go/no-go decision | `make h100-timing` | 15 min | yes |
| F | Round 1: Kev-4B's four stages | `make h100-round1` | 1.6–2.5 h | yes |
| G | Safety copy to the Hub | `den publish` | 5–10 min | yes |
| H | Round 2: public sources + replay (optional) | `make h100-round2` | 50–70 min | yes |
| I | Evaluate both rounds, choose, test once | `make h100-eval`, `make eval-test` | 30–40 min | yes |
| J | Release v1, prove it works, keep the evidence | `make release` | 15 min | yes |
| K | Shut the machine down | | 1 min | stops billing |

Budget about **5–6 GPU hours** for everything, **3.5–4** if round 2 is skipped. Part E tells you which fits.

---

## Part A: on your Mac, before renting anything

### A.1 Check, commit and push
The GPU box gets the code from GitHub (`ramput-labs/den`), so everything must be pushed first.

```bash
cd ~/ramput-labs/den
git checkout main && git pull   # the box must get every merged fix (e.g. the merged/ integrity check)
make check                     # must end with "... passed" and "claims: 0 of 0 trace to their evidence"
git checkout -b h100-v1
git add -A && git commit -m "Prepare v1 training"
git push -u origin h100-v1
```

If `data/` was regenerated since the last upload (a change to `sources.py`, `normalize.py` or `clean.py`), upload it
before renting the box. Otherwise the box downloads the old copy and trains on it:

```bash
make data-upload DATA_REPO=a1i6ek/den-datasets TAG=data-v1
```

### A.2 Tokens and names
- [ ] **Hugging Face token** with *write* access: <https://huggingface.co/settings/tokens>
- [ ] **GitHub token** if the repo is private (read is enough to clone; *write* to push results back from the box)
- [ ] **Model repo** name, created **private** by the first upload. Suggested: `a1i6ek/den-qwen3.5-4b`
- [ ] **Data copy** to train from: `a1i6ek/den-datasets` (the default), or a pinned `<repo>@<tag>`
- [ ] Your **budget** in hours, e.g. 6

- [ ] `make check` passed, branch pushed, tokens and names written down

---

## Part B: rent the H100 and check it

Rent **1× H100 80 GB** with Ubuntu 22.04/24.04, **NVIDIA driver ≥ 580** and **≥ 120 GB disk**. SSH in.

### B.1 Is the machine right?
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h ~ | tail -1
```
You should see `NVIDIA H100 80GB HBM3, 81559 MiB, 580.xx` (or newer) and at least 80 GB free.

- **Driver below 580?** Stop here: the locked PyTorch needs CUDA 13. Destroy the machine and rent one with a newer
  image. Don't try to install drivers.

### B.2 Packages, and tmux for everything
```bash
sudo apt update && sudo apt install -y git build-essential tmux curl    # often preinstalled; Triton needs gcc
tmux new -s den           # an SSH drop can't kill a run; reconnect with: tmux attach -t den
```

- [ ] H100, 80 GB, driver ≥ 580, ≥ 80 GB free, inside tmux

---

## Part C: code, environment, model and data (40–50 min)

### C.1 Your variables (paste, with your values)
```bash
export GITHUB_TOKEN=ghp_...
export HF_TOKEN=hf_...
export HF_REPO=a1i6ek/den-qwen3.5-4b
export DATA_REPO=a1i6ek/den-datasets
export BUDGET_HOURS=6
echo "export HF_REPO=$HF_REPO DATA_REPO=$DATA_REPO BUDGET_HOURS=$BUDGET_HOURS" >> ~/.bashrc
```

### C.2 uv, the code and the Hub login
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone --branch h100-v1 https://$GITHUB_TOKEN@github.com/ramput-labs/den.git && cd den
uvx hf auth login --token "$HF_TOKEN"
```

### C.3 Everything else in one command
```bash
make h100-setup 2>&1 | tee ~/setup.log
export UV_NO_SYNC=1 && echo 'export UV_NO_SYNC=1' >> ~/.bashrc    # your own `uv run`s never undo Unsloth
make check                                                         # must pass
```
`make h100-setup` runs, stopping at the first failure:
1. `setup-gpu`: `uv sync` (torch 2.14.1 for CUDA 13, transformers 5.x, flash-linear-attention), a check that torch
   sees the H100, then Unsloth on top without replacing torch or transformers, and a second check (`bf16 True`, no
   `BAD` line).
2. `model`: `qwen3.5-4b` (9.3 GB), every shard checked against `locks/`.
3. `data-download`: `data/` from `$DATA_REPO` (~3.6 GB), then a rebuild of the files the licences keep out of the
   copy (~20 min). It first prints `data commit: <repo>@<commit>` (the exact copy, recorded by the release in Part J)
   and ends with `ok, every file matches`.
4. `data-check` twice: round 1's and round 2's data shapes (the data gate below).
5. `h100-doctor`: the readiness gate, saved to `doctor.log`.

If `make h100-setup` stops, fix what it names and run it again: finished downloads are verified, not redone.

### C.4 The gates: the numbers must match exactly
In `~/setup.log`, the first `data-check` (round 1) must print:
```
train      40352 records    40352 questions  tokens median 176 p99 4455 max 7390 total 17.6M  skipped 0
dev         1468 records     1468 questions  tokens median 111 p99 788 max 858 total 0.3M  skipped 0
calib       1148 records     1148 questions  tokens median 120 p99 774 max 815 total 0.2M  skipped 0
```
and the second (round 2) must start with
`train      37459 records    37459 questions  tokens median 118 p99 2481 max 6058 total 8.8M  skipped 0`.
Then `h100-doctor` must print `ok` on every line, including `GPU memory fits qwen3.5-4b in bf16`.

- **Different numbers?** Stop and send me both outputs. The code or data differs from what was checked.
- **Unsloth fails to install?** `uv pip install --no-deps unsloth unsloth_zoo`, then install whatever import fails,
  one at a time, with `uv pip install -c .unsloth-pins.txt <package>`.
- **`transformers < 5.17`?** `uv pip install "transformers==$(grep -i '^transformers==' .unsloth-pins.txt | cut -d= -f3)"`.
- **Never install `vllm` or `bitsandbytes`.**

- [ ] data shapes match, every doctor line `ok`, `make check` passed

Start a progress log, one line per finished part:
```bash
echo "$(date -u +%H:%M) C done: setup ok" >> ~/progress.log
```

---

## Part D: gates: can it train at all? (~15 min)

```bash
make h100-gates
```
- `test-cuda.log` shows **3 passed**. The third test loads Qwen3.5-4B through Unsloth with LoRA on exactly 248 text
  projections in fp32 and takes one real step.
- `den overfit` ends with **PASS**. Note `content_agreement` in `overfit-lora.json` (near 1.0: it reads option
  content).
- In `overfit-train.log` the last `epoch dev` line shows **acc ≥ 0.95**.

A failure here is the cheapest one to have: stop and send me the log.

- [ ] 3 passed, PASS, overfit acc ≥ 0.95

---

## Part E: timing run and the decision (~15 min)

Count the minutes spent since the machine started (≈ 75 if all went well), then:
```bash
make h100-timing BUDGET_HOURS="$BUDGET_HOURS" SPENT=75
```
It trains both rounds for 20 steps per stage, checks both merges, runs `den predict` on the merged model, prints the
time estimate with a decision, and deletes `runs/timing`. While stages 3–4 run, open a second tmux pane (`Ctrl-b %`)
and watch memory: `watch -n 5 nvidia-smi`.

Check `runs-timing.log`:
- every stage has a `setup` line with `engine unsloth  dtype bf16 ... lora_modules 248 ... trainable_dtype float32`
- every stage after the first has `init   from runs/timing/...`
- no `nan` loss, no `CUDA out of memory`
- both merges print `saved .../merged  (248 merged weights)` and `check_merged` prints `ok` twice

The decision at the end:
- **run both rounds:** do F, G, H.
- **run round 1 only:** do F and G, skip H.
- **stop and report:** send me the output before spending more.

- **Out of memory in a stage?** Halve that stage's `--batch` and double its `--accum` on its line in the
  `Makefile` (`train-round1`; same effective batch), rerun `make h100-timing`, and tell me.

- [ ] timing clean, decision noted

```bash
echo "$(date -u +%H:%M) E done: <decision>" >> ~/progress.log
```

---

## Part F: round 1, Kev-4B's four stages (1.6–2.5 h)

```bash
make h100-round1          # -> runs/kev-recipe/{1-base,2-dates,3-documents,4-skills}; log: runs-train.log
```
Checkpoints every 500 steps; each stage is capped at 100 minutes (it then calibrates, saves and hands over).

Check every ~15 minutes (not more often):
```bash
grep -E "^setup|^init|dev|calibration|saved|stopping|nan|Error" runs-train.log | tail -20
```
What good looks like:
- each stage ends with `epoch dev ...`, `calibration  T ...` (T between 0.5 and 3), `dev at T ...` and
  `saved runs/kev-recipe/<stage>`
- **1-base:** dev nll < 1.0 and acc ≥ 0.63 (chance: 0.31)
- **3-documents, 4-skills:** `dev/core` no more than ~3 points below 1-base
- the last stage also prints `saved runs/kev-recipe/4-skills/merged  (248 merged weights)` and `integrity ok`

**If the machine or SSH dies mid-stage:** reconnect and `tmux attach -t den` (or start a new tmux). Finished stages
are kept. Print the round's commands with `make -n h100-round1`, and rerun **only the unfinished stage's line and the
ones after it**, adding `--resume latest` to the unfinished one. It continues from its last checkpoint.

- [ ] all four stages saved, merge and integrity ok

---

## Part G: safety copy to the Hub (5–10 min)

If the machine is lost after this, the trained model isn't:
```bash
uv run den publish --run runs/kev-recipe/4-skills --repo "$HF_REPO" --logs runs-timing.log runs-train.log
```
It ends with the repo URL and a file count. If it's interrupted, run the same command again: it resumes. This is not
a release yet: that's Part J.

- [ ] safety copy uploaded

---

## Part H: round 2 (skip if Part E said so)

```bash
make h100-round2          # -> runs/round2; log: runs-round2.log
```
Good: `epoch dev ...`, `calibration  T ...` (0.5–3), `saved runs/round2` and
`saved runs/round2/merged  (248 merged weights)`. Its dev accuracy should be close to round 1's; Part I judges it.

- [ ] round 2 saved (or skipped)

---

## Part I: evaluate, choose, test once

### I.1 Evaluate both rounds on dev and choose (dev only)
```bash
make h100-eval            # log: eval-rounds.log
```
For each round: integrity, every dev file with Kev's robustness checks, the augmentations (pairs, permute,
none-of-the-above, distractor), and 500 records of every source. Then `den compare` ends with `ship: <run>`, and
`den compare --paired` says whether the difference is beyond noise. Without round 2 it evaluates round 1 alone.

Set the shipped run from what `compare` printed:
```bash
export SHIP=runs/round2          # or runs/kev-recipe/4-skills; without round 2 always runs/kev-recipe/4-skills
```

### I.2 The locked test: once, for the shipped run only
```bash
make eval-test RUN=$SHIP 2>&1 | tee test.log
SRC_TEST=$(cd data/clean && ls test/sources/*/*.jsonl | tr '\n' ' ')
uv run den evaluate --final --run "$SHIP" --limit 1000 $SRC_TEST 2>&1 | tee -a test.log
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
DATA_PIN=$(grep -oP '^data commit: \K\S+' ~/setup.log)      # <repo>@<commit> that data-download fetched
make release VERSION=v1 RUN=$SHIP REPO="$HF_REPO" DATA_REPO="$DATA_PIN" \
  ARGS="--logs runs-timing.log runs-train.log runs-round2.log"
```
This publishes the run, tags that Hub commit `v1` and writes `releases/v1.json` (lineage, data hashes, dev and test
results, file hashes, the data copy it trained from, pinned to its commit). Drop `runs-round2.log` if you skipped round 2. Add
`--dry-run` to `ARGS` first if you want to see every check pass before uploading.

### J.2 Prove the published model answers
```bash
uv run den predict --run release:v1 scripts/request.jsonl
uv run den serve --run release:v1 --port 8009 > serve.log 2>&1 &
until grep -q "den serving" serve.log; do sleep 2; done
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d @scripts/request.jsonl; echo
kill %1
```
Both print JSON with `answers.department.probabilities` (and the other questions') summing to 1.

### J.3 Save the evidence to git
`releases/v1.json` and `reports/runs/` (every evaluation's numbers and per-question answers) are what v2 will be
compared against:
```bash
git add releases/ reports/runs/
git commit -m "Release v1: record and evaluation evidence"
git push
```
No write token on the box? Copy them to your Mac instead (run this **on your Mac**):
```bash
scp -r <user>@<gpu-host>:den/releases <user>@<gpu-host>:den/reports/runs ~/ramput-labs/den/
```

- [ ] v1 released, predict and serve answer, evidence pushed or copied

---

## Part K: shut it down

- [ ] `releases/v1.json` and `reports/runs/` are safe (pushed or copied)
- [ ] the model is on the Hub (`https://huggingface.co/$HF_REPO`, tag `v1`)
- [ ] **destroy the instance** in your cloud console. Stopping isn't always enough: some providers bill stopped disks

```bash
echo "$(date -u +%H:%M) done: v1 released" >> ~/progress.log && cat ~/progress.log
```
Send me `~/progress.log`, `test.log` and `eval-rounds.log`, and I'll help you read the results against Kev-4B.

---

## Later: v2

Once v1 is released and you've added new data (new pins, then `make data-normalize data-clean`, and
`make data-upload DATA_REPO=<repo> TAG=data-v2`):
```bash
make train-next FROM=v1 OUT=runs/v2 DATA="train/<new>.jsonl"
make eval-dev RUN=runs/v2 && make eval-test RUN=runs/v2
make release VERSION=v2 RUN=runs/v2 REPO="$HF_REPO" PARENT=v1 DATA_REPO=<repo>@data-v2
```
The release refuses v2 if any dev file falls more than a point below v1, and records the question-by-question
comparison.

---

## Quick fixes

| Symptom | Fix |
|---|---|
| `uv run` hangs at 0% CPU | another uv process holds the lock: use `.venv/bin/den ...` |
| a package vanished after a `uv run` | `UV_NO_SYNC=1` wasn't set: `make setup-gpu`, then the `export` in C.3 |
| `... falling back to its reference PyTorch implementation` | flash-linear-attention is missing (many times slower): send me the log before training |
| `CUDA out of memory` | halve that stage's `--batch`, double its `--accum` in the `Makefile` |
| SSH dropped | `tmux attach -t den`; training kept running |
| instance died mid-stage | `make -n h100-round1`, rerun the unfinished stage's line with `--resume latest`, then the rest |
| `merge: replaced N of 248` | the adapter is fine; send me the log, don't retrain |
| an upload stopped | run the same command again: uploads resume |
| `has already read [...]: the locked test set is read once` | working as intended: don't try to re-read test |
