# Training v1 on RunPod (1× RTX PRO 6000, 96 GB): brief for the cloud agent

You are a Claude agent on a rented RunPod pod: 1× NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM), 90 GB RAM, 16 vCPU.
Your job is den's **v1**: train Qwen3.5-4B in two rounds, pick the better round on dev, read the locked test set
once, release it to the Hugging Face Hub, and prove it answers over HTTP.

**No local GPU rehearsal was run:** the person has no NVIDIA card. Instead `make mac-smoke` passed on their Mac
(train, merge, integrity, evaluate, serve on qwen3.5-0.8b through PEFT), so this pod is the first time the code meets
CUDA, Unsloth and the GPU kernels. Where the runbook says the local rehearsal passed, read it as that. Phase 3's gates
are therefore the real first test: take them slowly and stop at the first failure.

**Your runbook is [`docs/h100-runbook.md`](h100-runbook.md).** Follow its phases 0–10 and their gates exactly. It
was written for an H100. This file lists only what differs on this pod, and it overrides the runbook where they
disagree. Read `CLAUDE.md` first: its hard rules apply (never train or tune on `test/`, fit T only on
`calibration/`, choose on `dev/` only, never edit pinned data or `locks/`).

## What "fine-tuning" means here

The recipe is fixed: **LoRA rank 16 on all 248 text-decoder projections, bf16 base, fp32 adapters**, plus the pointer
head, in Kev-4B's four stages (round 1) and the public-sources round (round 2). den does not implement
full-parameter fine-tuning, and you must not switch to it, raise the LoRA rank, change batch sizes, learning rates or
epochs, or run "quick experiments". The only allowed recipe edit is the runbook's out-of-memory fallback (halve
`--batch`, double `--accum` for that stage).

## Inputs (ask for any that are missing before spending GPU time)

| Variable | Meaning |
|---|---|
| `REPO_URL`, `BRANCH` | the code, already pushed |
| `GITHUB_TOKEN` | read access if the repo is private; write access lets you push the release record |
| `HF_TOKEN` | Hugging Face token with **write** access, set as a pod environment variable |
| `HF_REPO` | where the model goes, created **private** (e.g. `ramput-labs/den-qwen3.5-4b`) |
| `DATA_REPO` | the uploaded data copy, pinned: `a1i6ek/duck-datasets@data-v1` |
| `BUDGET_HOURS` | hard ceiling for the whole session (suggested: 10) |

Secrets: never print, echo, log or commit `HF_TOKEN` or `GITHUB_TOKEN`. Use them only as environment variables.

## Pod layout: everything that must survive lives on `/workspace`

On RunPod only the volume at `/workspace` survives a pod restart; the container disk (including `/root`) is wiped.
So, before phase 1:

```bash
cd /workspace
export HF_HOME=/workspace/.cache/huggingface UV_CACHE_DIR=/workspace/.cache/uv UV_NO_SYNC=1
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/.cache/huggingface UV_CACHE_DIR=/workspace/.cache/uv UV_NO_SYNC=1
EOF
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux git curl)
```

- Clone the repo to `/workspace/den`. Its `.venv`, `models/`, `data/` and `runs/` then persist.
- Write the progress log to **`/workspace/progress.log`**, not `~/progress.log`. Everywhere the runbook says
  `~/progress.log` or `~/setup.log`, use `/workspace/progress.log` and `/workspace/setup.log`.
- `UV_NO_SYNC=1` is exported from the start here. The first `make setup-gpu` still syncs, because it calls
  `uv sync` explicitly.
- After a pod restart: `source ~/.bashrc` is gone with the container disk, so re-run the three lines above, reinstall
  uv (`curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env`), `cd /workspace/den`, read
  `/workspace/progress.log`, and continue from the last passed gate. An interrupted training stage resumes with its
  `Makefile` line plus `--resume latest`.

## Phase overrides

**Phase 0 (machine).** Gate 0 becomes:
- `nvidia-smi` shows `NVIDIA RTX PRO 6000 Blackwell ...` with about 96 GB (≈ 97,000 MiB).
- `nvidia-smi --query-gpu=compute_cap --format=csv` prints `12.0` (Blackwell).
- **Driver ≥ 580** (the locked PyTorch is the CUDA 13 build). If older, stop and tell the person to pick another
  pod or template. Never install drivers.
- `df -h /workspace` shows **at least 150 GB free** (base 9.3 GB, data and rebuilt raws ~10 GB, venv ~15 GB, two
  merged runs ~18 GB, checkpoints, caches, headroom). `den doctor` checks 80 GB on the repo's own filesystem, so
  the repo must live on `/workspace`.
- `free -g` shows ~90 GB, `nproc` 16.

**Phase 1 (environment).** Unchanged, run from `/workspace`. This GPU is newer than the one the stack was validated
on, so two extra checks belong to Gate 1, and both failures mean **stop and report** (versions, the full error),
not workarounds:
- `uv run python -c "import torch; print(torch.cuda.get_device_capability(), torch.cuda.is_bf16_supported())"`
  prints `(12, 0) True`, with no "not compatible with the current PyTorch installation" warning.
- `import unsloth` works (Gate 1 already requires it).

**Phase 2 (model and data).** Unchanged: `make h100-setup DATA_REPO="$DATA_REPO" 2>&1 | tee /workspace/setup.log`.
The `h100-*` targets are not H100-specific; they train qwen3.5-4b on whatever CUDA GPU is present. Gate 2's
expected `data-check` lines are in the runbook and are exact. Any difference means the code or data copy differs
from what was validated: stop and report.

**Phase 3 (gates and timing).** Unchanged, and the most important phase on this hardware:
- `make test-cuda` (inside `make h100-gates`) is the first time Unsloth, Triton and flash-linear-attention compile
  their kernels for sm_120. It must show `3 passed`. A Triton or CUDA compilation error, or a log line saying
  flash-linear-attention is `falling back to its reference PyTorch implementation`, means stop and report. Training
  would be many times slower or wrong.
- In Gate 3, every stage's `setup` line shows `device NVIDIA RTX PRO 6000 ...` instead of `NVIDIA H100`.
- This card has more memory than an H100 (96 vs 80 GB) but lower bandwidth and throughput, so expect every stage to
  take longer than the runbook's H100 times. **Use only the measured estimate** from `make h100-timing
  BUDGET_HOURS=$BUDGET_HOURS SPENT=<minutes>`, and apply its decision as written (both rounds / round 1 only / stop).
- Don't raise batch sizes to use the extra memory. The recipe's effective batch is fixed at 8.

**Phases 4–10.** Unchanged. The H100 per-stage caps stay (`--max-minutes 100` a stage in round 1, `90` in round 2).
If the timing estimate says a stage needs longer, add `ARGS="--max-minutes <estimate + 30%>"` to that phase's
`make` command and note it in the progress log. Don't drop stages or data.

**Phase 8 (release record).** `make release` writes `releases/v1.json`, and phase 7/8 evaluations write
`reports/runs/`. Commit both on a new branch `runpod-v1` and push it if `GITHUB_TOKEN` has write access. If the push
is refused, upload them to the model repo instead, and say so in the report:

```bash
uvx hf upload "$HF_REPO" releases/v1.json evidence/releases/v1.json
uvx hf upload "$HF_REPO" reports/runs evidence/reports/runs
```

## Final report

As in the runbook's "Final report to the person", plus:
- the GPU line from Gate 0 (name, driver, compute capability) and the measured `tokens_per_second` of every stage;
- the typed-decisions dev/test rows, with `kl_to_target` next to accuracy. Its leaderboard reports KL; accuracy above
  ~0.74 there means fitting its teacher's quirks, not "better";
- the per-type temperatures (`calibration_report.temperatures`) of the shipped run;
- total pod hours, and whether `releases/v1.json` was pushed or uploaded to the Hub.

Then tell the person to **stop or terminate the pod**. A stopped pod still bills for its volume; a terminated one
deletes `/workspace`, so terminate only after the Hub upload is verified (phase 9).
