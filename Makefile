# den: run `make help` for targets. Model targets take MODEL=<key> (see `make models`); default qwen3.5-4b.
.DEFAULT_GOAL := help
UV := uv run
MODEL ?= $(or $(DEN_MODEL),qwen3.5-4b)
RUNS ?= runs/kev-recipe
ROUND2 ?= runs/round2
CAP ?= 1500
EVAL := --eval-steps 400 --eval-max 600
KEV_SUITES := train/core.jsonl train/dates-unknowable.jsonl train/documents.jsonl train/skills.jsonl train/devtools.jsonl
TRAIN := $(UV) den train --model $(MODEL)

.PHONY: test-cuda doctor train-kev train-round2 serve help setup check test-model list models model data data-raw-train data-raw-new data-raw-eval data-raw-bulk data-all verify normalize clean-data audit train-setup train-check train env smoke

help: ## show targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

setup: ## install Python 3.12 + locked dependencies
	uv sync

check: ## lint, strict type-check, unit tests
	$(UV) ruff check den tests
	$(UV) ruff format --check den tests
	$(UV) mypy
	$(UV) pytest -q

test-model: ## MLX vs PyTorch parity on the downloaded qwen3.5-4b (slow)
	$(UV) pytest -q -m model

test-cuda: ## CUDA checks on the GPU box: devices, dtypes, padding, and one real Unsloth LoRA + head step
	$(UV) pytest -q -m cuda

doctor: ## versions, GPU, BF16, driver, Unsloth, model revision; on the GPU box: make doctor ARGS="--require cuda"
	$(UV) den doctor $(ARGS)

models: ## list backbones of every size and the Kev reference checkpoints
	$(UV) den models

model: ## download MODEL (a key from `make models`, or hf:<org>/<name>@<commit>)
	$(UV) den model $(MODEL)

list: ## print every pinned dataset
	$(UV) den list

data: ## suites: Kev 1.0 train / calibration / dev / locked test (~100 MB)
	$(UV) den data suites

data-raw-train: ## raw sources Kev trained from, at Kev's pins (~0.8 GB)
	$(UV) den data raw-train

data-raw-new: ## new trainable sources aimed at Kev's weak spots (~0.3 GB)
	$(UV) den data raw-new

data-raw-eval: ## eval-only raw sources: transfer, devtools, breadth-v1 (~0.6 GB)
	$(UV) den data raw-eval

data-raw-bulk: ## CFPB, CommitPackFT, CodeReviewer, FlakeFlagger raws (~5 GB)
	$(UV) den data raw-bulk

data-all: ## every set except raw-bulk
	$(UV) den data all

verify: ## re-hash every downloaded file against locks/
	$(UV) den verify

normalize: ## raw sources -> canonical, leakage-free train / dev / test under data/*/sources/
	$(UV) den normalize

clean-data: ## write cleaned, deduplicated training copies under data/clean/ (Kev's pinned files are untouched)
	$(UV) den clean

audit: ## check every record and source is fit to train on; writes reports/data-audit.json
	$(UV) den audit --model $(MODEL)

train-setup: ## install Unsloth, keeping the locked torch/transformers (Linux + NVIDIA GPU; see instructions.md phase 1)
	uv pip freeze | grep -iE '^(torch|transformers|tokenizers|huggingface-hub|numpy|triton|flash-linear-attention|accelerate|peft)==' > .unsloth-pins.txt
	uv pip install unsloth -c .unsloth-pins.txt

train-check: ## tokenize the clean training data and report its shape (no GPU needed)
	$(UV) den train --dry-run --model $(MODEL)

train: ## LoRA + pointer-head fine-tune of MODEL with Unsloth (Linux + NVIDIA GPU; `make train-setup` first); ARGS="--epochs 2 ..."
	$(UV) den train --model $(MODEL) $(ARGS)

train-kev: ## round 1, Kev-4B's four stages, each from the last (Linux + NVIDIA GPU); ARGS go to every stage
	$(TRAIN) --data train/core.jsonl --epochs 2 --lr 5e-5 --batch 4 --accum 2 --p-none-pair 0.25 $(EVAL) \
		--dev dev/core.jsonl --out $(RUNS)/1-base $(ARGS)
	$(TRAIN) --data train/dates-unknowable.jsonl --replay 2000 --init-from $(RUNS)/1-base --lr 2e-5 --batch 4 --accum 2 \
		$(EVAL) --dev dev/core.jsonl --out $(RUNS)/2-dates $(ARGS)
	$(TRAIN) --data train/documents.jsonl --replay 2000 --init-from $(RUNS)/2-dates --lr 2e-5 --batch 2 --accum 4 \
		$(EVAL) --dev dev/documents.jsonl dev/core.jsonl --out $(RUNS)/3-documents $(ARGS)
	$(TRAIN) --data train/skills.jsonl train/devtools.jsonl --replay 4000 --init-from $(RUNS)/3-documents --lr 2e-5 \
		--batch 2 --accum 4 $(EVAL) --dev dev/skills.jsonl dev/devtools.jsonl dev/core.jsonl --merge \
		--out $(RUNS)/4-skills $(ARGS)

train-round2: ## round 2 from round 1: the 17 public sources (CAP each) + CAP replayed from each Kev suite; needs `make normalize`
	$(TRAIN) --data --sources-cap $(CAP) --replay $(CAP) --replay-from $(KEV_SUITES) --init-from $(RUNS)/4-skills \
		--lr 2e-5 --batch 4 --accum 2 $(EVAL) --dev dev/core.jsonl dev/documents.jsonl dev/skills.jsonl dev/devtools.jsonl \
		--merge --out $(ROUND2) $(ARGS)

serve: ## serve RUN (a run directory or hf:<org>/<name>) at POST /v1/systemone; PORT=8009
	$(UV) den serve --run $(RUN) --port $(or $(PORT),8009)

env: ## show the backend this machine uses (mlx | cuda | cpu)
	$(UV) den env

smoke: ## run MODEL once on this machine's backend
	$(UV) den smoke --model $(MODEL)
