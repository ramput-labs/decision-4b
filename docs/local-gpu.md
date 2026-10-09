# Local GPU test: the whole pipeline on your RTX 3070 (8 GB), start to finish

A step-by-step path for you, the person at the keyboard, before any H100 time is rented. It runs the **same pipeline**
as the cloud run: the same Unsloth install, the four-stage round 1 with replay and `--init-from` handoffs, round 2 on
the public sources, merge, integrity checks, evaluation, the ship decision and a real request. Everything is small,
so a bug in setup, data, memory handling, stage handoffs or merging costs nothing to find. Tick the boxes as you go.

**When you're stuck,** send me: the step number (e.g. "Step 3.2"), the command you ran, and the last ~40 lines of its
output (`tail -40 runs/local/logs/<log>`).

| | Local RTX 3070 (this guide) | Cloud H100 (`h100-guide.md`) |
|---|---|---|
| model | `qwen3.5-0.8b` (`LOCAL_MODEL`) | `qwen3.5-4b` |
| data | ≤ 1,000 records from any one file (`LOCAL_N`); 100 per public source in round 2 (`LOCAL_CAP`) | everything (17.6M + 8.8M tokens) |
| batch | 1 × 8 accumulation (the same effective batch of 8) | 4 × 2 or 2 × 4 per stage |
| validation | every 50 steps on 200 dev questions | every 400 steps on 600 |
| runs and logs | `runs/local/`, `runs/local/logs/` (gitignored) | `runs/kev-recipe/`, `runs/round2`, logs at the repo root |
| test set, evidence | never read, none written | read once for the shipped run; `reports/runs/`, `releases/` |

**Why 0.8B and not 4B:** Qwen3.5-4B's bf16 weights alone are 9.3 GB, more than the card holds, and this project never
trains in 4-bit. The code path is identical for every Qwen3.5 size, so 0.8B tests everything except the 4B's memory
use and speed, which the H100's timing run measures. `make local-doctor` refuses a model that doesn't fit the card.

## The plan at a glance

| Step | What | Time (rough) |
|---|---|---|
| 1 | Prepare the machine: Linux or WSL2, driver, packages | 10–30 min, once |
| 2 | Install the code, Unsloth, the model and the data | 30–40 min, once |
| 3 | Gates: CUDA tests, overfit tests | ~15 min |
| 4 | Train both rounds at small scale | ~2 h (1 h 50 min measured) |
| 5 | Evaluate, choose, answer a request | ~10 min |
| 6 | Read the results; go to the H100 | 5 min |

The times are estimates for this card; the logs record the real ones. `make local` runs steps 3–5 in one go once
step 2 is done, but the first time, run them one by one as below so you see each gate.

---

## Step 1: prepare the machine (once)

### 1.1 Linux, or WSL2 on Windows
- **Ubuntu 22.04 or 24.04:** nothing to do here.
- **Windows:** in PowerShell as administrator, `wsl --install -d Ubuntu-24.04`, reboot, open "Ubuntu". Do everything
  below inside that Ubuntu shell, and keep the repo in the Linux home (`~/`), not under `/mnt/c` (many times slower).

### 1.2 NVIDIA driver ≥ 580
The locked PyTorch is the CUDA 13 build, which needs driver 580 or newer.
- **Ubuntu:** `sudo ubuntu-drivers install` (or `sudo apt install nvidia-driver-580`), then reboot.
- **WSL2:** install the newest Windows driver from nvidia.com on the Windows side. Never install a driver inside WSL.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```
You should see `NVIDIA GeForce RTX 3070, 8192 MiB, 580.xx` (or newer).

### 1.3 System packages and disk
```bash
sudo apt update && sudo apt install -y git build-essential tmux curl    # Triton compiles kernels with gcc
df -h ~ | tail -1                                                       # need >= 30 GB free
```
Disk: data/ 6 GB, the 0.8B model 1.8 GB, the Python environment about 15 GB, plus runs.

- [ ] Linux shell, `nvidia-smi` shows the RTX 3070 with driver ≥ 580, packages installed, ≥ 30 GB free

---

## Step 2: code, environment, model and data (once)

Work inside tmux, so a closed terminal doesn't kill a run: `tmux new -s den` (later: `tmux attach -t den`).

### 2.1 uv, the code and a Hugging Face login
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/ramput-labs/den.git && cd den
git checkout <your branch>              # the branch you pushed from the Mac
uvx hf auth login                       # a read token is enough here; faster, rate-limit-free downloads
```

### 2.2 Everything else in one command
```bash
make local-setup 2>&1 | tee ~/local-setup.log
```
It runs, stopping at the first failure:
1. `setup-gpu`: `uv sync` (Python 3.12, torch 2.14.1 for CUDA 13, transformers 5.x, flash-linear-attention), a
   check that torch sees the GPU, then Unsloth installed on top without replacing torch or transformers, and a
   second check (`unsloth ... bf16 True`, no `BAD` line).
2. `model`: `qwen3.5-0.8b` (1.8 GB), every file checked against `locks/`.
3. `data-download`: `data/` from `a1i6ek/den-datasets` (~3.6 GB; it prints the exact `data commit` first), then a
   rebuild of the files the licences keep out of that copy (~20 min). It ends with `ok, every file matches`.
4. `data-check`: tokenizes the rehearsal's training data and prints its shape.
5. `local-doctor`: the readiness gate.

Then, so your own `uv run` commands never undo the Unsloth install (make targets already handle it):
```bash
export UV_NO_SYNC=1 && echo 'export UV_NO_SYNC=1' >> ~/.bashrc
```

### 2.3 What good looks like
- `data-download` ends with `data/ from datasets/a1i6ek/den-datasets: ok, every file matches`.
- `data-check` prints three lines, `train`, `dev` and `calib`, each ending in `skipped 0`.
- `local-doctor` prints `ok` on every check, including `GPU memory fits qwen3.5-0.8b in bf16`.

If something fails:
- **`CUDA is not available`:** the driver (Step 1.2), or on WSL2 an outdated Windows driver.
- **Unsloth fails to install:** `uv pip install --no-deps unsloth unsloth_zoo`, then install whatever import fails,
  one at a time, with `uv pip install -c .unsloth-pins.txt <package>`. Never install `vllm` or `bitsandbytes`.
- **`data-download` stopped midway:** run `make data-download` again. Finished files are verified, not redone.

- [ ] `make local-setup` finished, every doctor line `ok`, `UV_NO_SYNC=1` exported

---

## Step 3: gates: can this machine train at all? (~15 min)

```bash
make local-gates
```
Logs: `runs/local/logs/test-cuda.log`, `overfit-lora.log` (+ `overfit-lora.json`), `overfit-train.log`.

### 3.1 CUDA tests: `3 passed`
Padding and dtypes on CUDA, a toy LoRA + head step, and the real one: Unsloth loads qwen3.5-0.8b in bf16, puts LoRA on
every text projection (none in the vision tower), and one forward/backward step moves the adapters and the head.

### 3.2 Overfit gate (`den overfit`): ends with `PASS`
LoRA + head must fit 100 real training rows (train accuracy ≥ 0.95, deterministic, loss falls by more than half).
Note `content_agreement` in `overfit-lora.json`: near 1.0 means the head reads the options' content, not their position.

### 3.3 Overfit through the real training loop (`den train --overfit 100`)
The last `epoch dev  nll ...  acc ...` line in `overfit-train.log` must show **acc ≥ 0.95**, and `loss` must fall.

A model that can't fit 100 rows has a bug (positions, labels, masking, loss). More training won't fix it: stop and
send me the log.

- [ ] `3 passed`, `PASS`, overfit acc ≥ 0.95

---

## Step 4: train both rounds at small scale (~2 h)

```bash
make local-train
```
Logs: `runs/local/logs/round1.log` and `round2.log`. In a second tmux pane (`Ctrl-b %`), watch memory with
`watch -n 5 nvidia-smi`, especially during stage 3 (documents: the longest states).

It runs exactly what the H100 runs (`train-round1`, then `train-round2`), with `--limit 1000 --batch 1 --accum 8`:

| Stage | Data | Output |
|---|---|---|
| round 1, 1-base | `train/core`, 2 epochs, 25% none-of-the-above pairs | `runs/local/1-base` |
| round 1, 2-dates | `train/dates-unknowable` + `core` replay, from 1-base | `runs/local/2-dates` |
| round 1, 3-documents | `train/documents` + replay, from 2-dates | `runs/local/3-documents` |
| round 1, 4-skills | `train/skills` + `train/devtools` + replay, from 3-documents, merged | `runs/local/4-skills` |
| round 2 | 100 from each public source + 100 from each Kev suite, from 4-skills, merged | `runs/local/round2` |

Check while it runs, or after:
```bash
grep -E "^setup|^init|epoch dev|calibration|dev at T|saved|stopping|nan|Error|out of memory" runs/local/logs/round1.log
```
What good looks like, for every stage:
- a `setup` line with `engine unsloth  dtype bf16 ... trainable_dtype float32` (`bfloat16` there is a bug: stop)
- from stage 2 on: `init   from runs/local/...`
- `step N dev(200) ...` lines, then `epoch dev  nll ...  acc ...`, then `calibration  T ...` with T between 0.5 and 3
- `saved runs/local/<stage>`; 4-skills and round 2 also print `saved .../merged  (N merged weights)` and the
  integrity lines, all `ok`
- no `nan`, no `CUDA out of memory`

If something fails:
- **`CUDA out of memory`** (most likely in 3-documents): rerun with long states left out, and tell me, because the
  H100 might need a memory change too:
  `make local-train ARGS="--max-state 4096"`
- **A stage hits `--max-minutes 30`:** it stops, calibrates, saves and the next stage continues. Note which stage.
- **The terminal closed:** `tmux attach -t den`. If the machine rebooted, rerun `make local-train` (it starts over).

- [ ] four stages and round 2 saved, both merges ok, no nan, no out of memory

---

## Step 5: evaluate, choose, answer a request (~10 min)

```bash
make local-eval
```
Logs: `runs/local/logs/eval.log` and `predict.json`. It runs, for both `4-skills` and `round2`:
- `den check-run`: `integrity ok` for both rounds
- `den check-run`: every line `ok`, then `integrity ok`
- `den evaluate` on 200 records of `dev/core`, `dev/documents`, `dev/skills` and `dev/devtools` (acc, nll, brier,
  ece per file; nothing is written to `reports/`, and the test set is never read)
- `den compare`: a table of both rounds per file, ending with `ship: <run>`
- `den predict` on `scripts/request.jsonl`: a JSON answer with `department`, `escalate` and `frustration`, each
  question's probabilities summing to 1

- [ ] integrity ok for both, `ship: ...` printed, `predict.json` holds an answer

---

## Step 6: read the result, then go to the H100

With 1,000 records on a 0.8B model, accuracy is far below what the H100 run will reach. Don't read quality into it.
The rehearsal **passed** when:
- every gate above passed, and every stage ran, saved and merged
- `loss` fell in every stage, and `dev/core` accuracy beat chance (0.31)
- every T is between 0.5 and 3
- nothing ran out of memory or produced `nan`

Send me `runs/local/logs/` (or at least `round1.log`, `round2.log` and `eval.log`) if anything looked off.

Then commit and push from the Mac, and follow `h100-guide.md`. Nothing on the H100 is new: `h100-*` targets wrap the
same `train-round1`, `train-round2` and `eval-dev` you just ran, at full size.

- [ ] rehearsal passed; any memory change noted for the H100

---

## The whole rehearsal in one command

After the first time, steps 3–5 are one command, which stops at the first failed gate:
```bash
make local                  # local-doctor -> local-gates -> local-train -> local-eval
```
It ends with `local rehearsal passed: qwen3.5-0.8b, 1000 records a file; logs in runs/local/logs`.

## Knobs

| Command | Effect |
|---|---|
| `make local LOCAL_N=200` | quicker: 200 records a file |
| `make local-train LOCAL_N=5000` | longer; rerun `make local-eval` after |
| `make local-train ARGS="--max-state 4096"` | leave out states over 4,096 tokens (an out-of-memory workaround) |
| `make local LOCAL_MODEL=qwen3.5-2b` | the 2B (4.6 GB): `make model MODEL=qwen3.5-2b` first; little room for long states |
| `make local LOCAL_RUNS=runs/local-2` | keep an earlier rehearsal |
| `make local-clean` | delete `runs/local` |

`ARGS` reach every training stage. `make -n local-train` prints every command without running it.

## Quick fixes

| Symptom | Fix |
|---|---|
| `nvidia-smi` not found, or CUDA not available | Step 1.2: driver ≥ 580 (on WSL2, on the Windows side) |
| `uv run` hangs at 0% CPU | another uv process holds the lock: use `.venv/bin/den ...` |
| a package vanished after a `uv run` | `UV_NO_SYNC=1` wasn't set: `make setup-gpu`, then the `export` in Step 2.2 |
| `... falling back to its reference PyTorch implementation` | flash-linear-attention is missing (many times slower): send me the log |
| `causal_conv1d_fn is falling back ...` | expected: no `causal-conv1d` wheel for torch 2.14, and the fallback is cuDNN's conv1d, at most ~10% of a step (measured on this card). Nothing to do |
| `CUDA out of memory` | `make local-train ARGS="--max-state 4096"`, and tell me |
| very slow on WSL2 | the repo is under `/mnt/c`: clone it into `~/` instead |
| `merge: replaced N of M` | the adapter is fine; send me the log |
