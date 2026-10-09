# den: run `make help` for targets. Model targets take MODEL=<key> (see `make models`); default qwen3.5-4b.
#
# One pipeline, three machines. Targets are named <area>-<action>; the two GPU boxes share their verbs:
#   Mac (MLX)         make setup check model data-download        develop, evaluate, serve
#   local RTX 3070    make local-setup, then make local           the whole pipeline on 1,000 records (docs/local-gpu.md)
#   cloud H100        make h100-setup, then h100-* phase by phase  the real run (docs/h100-runbook.md)
.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
MAKEFLAGS += --no-print-directory

# After `make setup-gpu` has put Unsloth on top of the lockfile (.unsloth-pins.txt marks it), every `uv run` under
# make skips the sync that would take it away again.
UV := uv run
ifneq ($(wildcard .unsloth-pins.txt),)
export UV_NO_SYNC := 1
endif
MODEL ?= $(or $(DEN_MODEL),qwen3.5-4b)
RUNS ?= runs/kev-recipe
ROUND2 ?= runs/round2
CAP ?= 1500
EVAL ?= --eval-steps 400 --eval-max 600
SUITES_SCORED := core documents skills devtools transfer probes breadth
DEV := $(SUITES_SCORED:%=dev/%.jsonl) dev/binding.jsonl dev/semif.jsonl
TEST := $(SUITES_SCORED:%=test/%.jsonl)
SHIP_DEV := dev/core.jsonl dev/documents.jsonl dev/skills.jsonl dev/devtools.jsonl
KEV_SUITES := train/core.jsonl train/dates-unknowable.jsonl train/documents.jsonl train/skills.jsonl train/devtools.jsonl
# What shipped runs fit T on: Kev's calibration set plus every source's calibration split (`den normalize`), one T per
# question type. Read when a target runs, so it sees the data that data-download placed.
CALIB = --calibration calibration/core.jsonl $(patsubst data/clean/%,%,$(sort $(wildcard data/clean/calibration/sources/*/*.jsonl))) \
	--calibrate-by-type
NEXT_SOURCES ?= 500
TRAIN = $(UV) den train --model $(MODEL)
DATA_REPO ?= $(or $(DEN_DATA_REPO),a1i6ek/den-datasets)
GIVEN_DATA_REPO := $(filter command line environment,$(origin DATA_REPO))$(DEN_DATA_REPO)
log = 2>&1 | tee $(1)

# The local rehearsal: Qwen3.5-0.8B (4B does not fit 8 GB in bf16), at most LOCAL_N records from any one file,
# batch 1 x 8 (the H100's effective batch of 8), validation every 50 steps.
LOCAL_MODEL ?= qwen3.5-0.8b
LOCAL_N ?= 1000
LOCAL_CAP ?= 100
LOCAL_RUNS ?= runs/local
LOCAL_DISK ?= 30
LOCAL_LOGS := $(LOCAL_RUNS)/logs
LOCAL := MODEL=$(LOCAL_MODEL) RUNS=$(LOCAL_RUNS) ROUND2=$(LOCAL_RUNS)/round2 CAP=$(LOCAL_CAP) \
	EVAL="--eval-steps 50 --eval-max 200" ARGS="--limit $(LOCAL_N) --batch 1 --accum 8 --max-minutes 30 $(ARGS)"

# The H100 run (docs/h100-runbook.md): logs stay at the repo root, where `den publish --logs` picks them up.
H100_MODEL := qwen3.5-4b
BUDGET_HOURS ?= 5
SPENT ?= 75

.PHONY: help setup setup-gpu check doctor test-model test-cuda models model list env smoke serve \
	data-download data-upload data-verify data-check data-licences data-breadth data-normalize data-clean data-audit \
	train train-round1 train-round2 train-next train-rebuild eval-dev eval-test release release-list \
	local local-setup local-doctor local-gates local-train local-eval local-clean \
	h100-setup h100-doctor h100-gates h100-timing h100-round1 h100-round2 h100-eval

help: ## show targets
	@awk 'BEGIN {FS = ":.*## "} /^##@ / {printf "\n%s\n", substr($$0, 5)} /^[a-z0-9-]+:.*## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Setup and checks (any machine)

setup: ## install Python 3.12 + locked dependencies
	uv sync

setup-gpu: ## Linux + NVIDIA (driver >= 580): setup, then Unsloth on top without touching torch/transformers, both checked
	uv sync
	uv run --no-sync python -m scripts.check_env
	uv pip freeze | grep -iE '^(torch|transformers|tokenizers|huggingface-hub|numpy|triton|flash-linear-attention|accelerate|peft)==' > .unsloth-pins.txt
	uv pip install "unsloth>=2026.9" -c .unsloth-pins.txt
	uv run --no-sync python -m scripts.check_env --unsloth
	@echo "make targets now skip uv's sync; for your own uv runs: export UV_NO_SYNC=1 && echo 'export UV_NO_SYNC=1' >> ~/.bashrc"

check: ## lint, strict type-check, unit tests, claims vs evidence
	$(UV) ruff check den scripts tests
	$(UV) ruff format --check den scripts tests
	$(UV) mypy
	$(UV) pytest -q
	$(UV) python -m scripts.verify_claims

doctor: ## versions, GPU, BF16, driver, Unsloth, model revision; ARGS="--require cuda" to gate on them
	$(UV) den doctor --model $(MODEL) $(ARGS)

test-model: ## MLX vs PyTorch parity on the downloaded qwen3.5-4b (slow)
	$(UV) pytest -q -m model

test-cuda: ## CUDA checks: devices, dtypes, padding, and one real Unsloth LoRA + head step on MODEL
	DEN_MODEL=$(MODEL) $(UV) pytest -q -m cuda

models: ## list backbones of every size and the Kev reference checkpoints
	$(UV) den models

model: ## download MODEL (a key from `make models`, or hf:<org>/<name>@<commit>)
	$(UV) den model $(MODEL)

list: ## print every pinned dataset
	$(UV) den list

env: ## show the backend this machine uses (mlx | cuda | cpu)
	$(UV) den env

smoke: ## run MODEL once on this machine's backend
	$(UV) den smoke --model $(MODEL)

serve: ## serve RUN (a run directory or hf:<org>/<name>) at POST /v1/systemone; PORT=8009
	$(UV) den serve --run $(RUN) --port $(or $(PORT),8009)

##@ Data (data/ is gitignored; data-download is the one way to get it)

data-download: ## the Hub copy into data/, rebuilding only what the licences left out, every file checked: [DATA_REPO=<org>/<name>[@rev]] [NO_REBUILD=1]
	$(UV) python -m scripts.mirror download --repo $(DATA_REPO) $(if $(NO_REBUILD),--no-rebuild)

data-upload: ## data/ to the Hub, licence notices included, restricted sources left out: DATA_REPO=<org>/<name> [TAG=data-v1] [PUBLIC=1]
	@test -n "$(GIVEN_DATA_REPO)" || (echo "name the repo to upload to: DATA_REPO=<org>/<name> (or DEN_DATA_REPO)" && exit 1)
	$(UV) python -m scripts.mirror upload --repo $(DATA_REPO) $(if $(PUBLIC),--public) $(if $(TAG),--tag $(TAG))

data-verify: ## re-hash every downloaded file against locks/
	$(UV) den verify

data-check: ## tokenize the clean training data and print its shape (no GPU; the runbook's data gate)
	$(UV) den train --dry-run --model $(MODEL) $(ARGS)

data-licences: ## write each source's licence, evidence and texts into data/ (LICENSES.md, LICENSES/, README.md card)
	$(UV) den licences $(ARGS)

data-breadth: ## rebuild Kev's breadth-v1 panel -> data/{dev,test}/breadth.jsonl, sha256-checked; before data-normalize
	$(UV) python -m scripts.build_breadth

data-normalize: ## raw sources -> leakage-free train / dev / test under data/*/sources/ (after changing normalize.py)
	$(UV) den normalize

data-clean: ## rewrite the cleaned, deduplicated copies under data/clean/ (after changing clean.py)
	$(UV) den clean

data-audit: ## check every record and source is fit to train on; writes reports/data-audit.json
	$(UV) den audit --model $(MODEL)

##@ Training and evaluation (training: Linux + NVIDIA, after setup-gpu)

train: ## LoRA + pointer-head fine-tune of MODEL with Unsloth; ARGS="--epochs 2 ..."
	$(TRAIN) $(ARGS)

# Stage 1's head is fresh, so it trains at --head-lr 1e-3 (the overfit gate's rate and mode C's), not at the
# adapters' 5e-5: our change to Kev's recipe, which records no head rate. Later stages continue it at --lr.
train-round1: ## round 1, Kev-4B's four stages, each from the last -> RUNS/{1-base,...,4-skills}; ARGS go to every stage
	$(TRAIN) --data train/core.jsonl --epochs 2 --lr 5e-5 --head-lr 1e-3 --batch 4 --accum 2 --p-none-pair 0.25 $(EVAL) \
		--dev dev/core.jsonl --out $(RUNS)/1-base $(ARGS)
	$(TRAIN) --data train/dates-unknowable.jsonl --replay 2000 --init-from $(RUNS)/1-base --lr 2e-5 --batch 4 --accum 2 \
		$(EVAL) --dev dev/core.jsonl --out $(RUNS)/2-dates $(ARGS)
	$(TRAIN) --data train/documents.jsonl --replay 2000 --init-from $(RUNS)/2-dates --lr 2e-5 --batch 2 --accum 4 \
		$(EVAL) --dev dev/documents.jsonl dev/core.jsonl --out $(RUNS)/3-documents $(ARGS)
	$(TRAIN) --data train/skills.jsonl train/devtools.jsonl --replay 4000 --init-from $(RUNS)/3-documents --lr 2e-5 \
		--batch 2 --accum 4 $(EVAL) --dev dev/skills.jsonl dev/devtools.jsonl dev/core.jsonl $(CALIB) --merge \
		--out $(RUNS)/4-skills $(ARGS)

train-round2: ## round 2 from round 1 -> ROUND2: the 18 public sources (CAP each) + CAP replayed from each Kev suite
	$(TRAIN) --data --sources-cap $(CAP) --replay $(CAP) --replay-from $(KEV_SUITES) --init-from $(RUNS)/4-skills \
		--lr 2e-5 --batch 4 --accum 2 $(EVAL) --dev $(SHIP_DEV) $(CALIB) --merge --out $(ROUND2) $(ARGS)

# A new version continues from the last release on new data, replaying Kev's suites (REPLAY each) and the public
# sources (SOURCES each, default NEXT_SOURCES) so neither round's skills are forgotten.
train-next: ## the next version from a released one: FROM=v1 OUT=runs/v2 DATA="train/<new>.jsonl ..." [SOURCES=500] [REPLAY=1500]
	$(TRAIN) --data $(DATA) --sources-cap $(or $(SOURCES),$(NEXT_SOURCES)) --replay $(or $(REPLAY),$(CAP)) \
		--replay-from $(KEV_SUITES) --init-from release:$(FROM) --lr 2e-5 --batch 4 --accum 2 $(EVAL) \
		--dev $(SHIP_DEV) $(CALIB) --merge --out $(OUT) $(ARGS)

# The periodic rebuild: one run from the base on everything (Kev's suites in full, CAP of every source, any new DATA),
# so a chain of --init-from versions can be checked against a fresh model. Ship it only if it wins on dev
# (`den compare --paired`). Untested at full scale: the one-run mix is ours, not Kev's.
train-rebuild: ## one run from the base on all data -> OUT: OUT=runs/v3-rebuild [DATA="train/<new>.jsonl ..."] [EPOCHS=1]
	$(TRAIN) --data $(KEV_SUITES) $(DATA) --sources-cap $(CAP) --epochs $(or $(EPOCHS),1) --lr 5e-5 --head-lr 1e-3 \
		--batch 4 --accum 2 --p-none-pair 0.25 $(EVAL) --dev $(SHIP_DEV) $(CALIB) --merge --out $(OUT) $(ARGS)

eval-dev: ## a trained RUN on dev: integrity, metrics + Kev's robustness checks, augmentations
	$(UV) den check-run --run $(RUN)
	$(UV) den evaluate --run $(RUN) $(DEV)
	for AUG in pairs permute none-replace none-add distract; do \
		$(UV) den evaluate --run $(RUN) --augment $$AUG dev/core.jsonl dev/documents.jsonl dev/skills.jsonl || exit 1; \
	done

eval-test: ## the locked test set, read once, for the shipped RUN only (refuses a second read)
	$(UV) den evaluate --final --run $(RUN) $(TEST)

release: ## publish RUN as VERSION, tag it on REPO, record releases/VERSION.json: VERSION=v2 RUN=runs/v2 REPO=<org>/<name> [PARENT=v1] [DATA_REPO=<org>/<name>@<tag>]
	$(UV) den release create --version $(VERSION) --run $(RUN) --repo $(REPO) $(if $(PARENT),--parent $(PARENT)) \
		$(if $(GIVEN_DATA_REPO),--data-repo $(DATA_REPO)) $(ARGS)

release-list: ## every released version, its parent and headline numbers
	$(UV) den release list

##@ Local RTX 3070 (8 GB): the whole pipeline at small scale, before renting an H100 (docs/local-gpu.md)

local: local-doctor local-gates local-train local-eval ## the full rehearsal, gate by gate; logs in LOCAL_RUNS/logs
	@echo "local rehearsal passed: $(LOCAL_MODEL), $(LOCAL_N) records a file; logs in $(LOCAL_LOGS)"

local-setup: setup-gpu ## setup-gpu + LOCAL_MODEL + data/, the data shape at LOCAL_N, then local-doctor
	$(MAKE) model data-download MODEL=$(LOCAL_MODEL)
	$(MAKE) data-check MODEL=$(LOCAL_MODEL) ARGS="--limit $(LOCAL_N)"
	$(MAKE) local-doctor

local-doctor: ## the readiness gate for this card: CUDA, BF16, driver, Unsloth, kernels, VRAM for LOCAL_MODEL, disk
	$(MAKE) doctor MODEL=$(LOCAL_MODEL) ARGS="--require cuda --min-disk $(LOCAL_DISK)"

local-gates: ## CUDA tests on LOCAL_MODEL, then both overfit gates (head + LoRA must fit 100 rows)
	@mkdir -p $(LOCAL_LOGS)
	$(MAKE) test-cuda MODEL=$(LOCAL_MODEL) $(call log,$(LOCAL_LOGS)/test-cuda.log)
	$(UV) den overfit --model $(LOCAL_MODEL) --n 100 --steps 300 --batch 8 --out $(LOCAL_LOGS)/overfit-lora.json \
		$(call log,$(LOCAL_LOGS)/overfit-lora.log)
	$(UV) den train --model $(LOCAL_MODEL) --data train/core.jsonl --overfit 100 --epochs 25 --lr 2e-4 --head-lr 1e-3 \
		--batch 2 --accum 2 --out $(LOCAL_RUNS)/overfit $(call log,$(LOCAL_LOGS)/overfit-train.log)

local-train: ## both rounds exactly as on the H100 (every stage, replay, merge) at LOCAL_N records a file; ARGS add flags
	@mkdir -p $(LOCAL_LOGS)
	$(MAKE) train-round1 $(LOCAL) $(call log,$(LOCAL_LOGS)/round1.log)
	$(MAKE) train-round2 $(LOCAL) $(call log,$(LOCAL_LOGS)/round2.log)

local-eval: ## integrity of both merged rounds, dev sample scores, the ship decision, and a real request answered
	@mkdir -p $(LOCAL_LOGS)
	for RUN in $(LOCAL_RUNS)/4-skills $(LOCAL_RUNS)/round2; do \
		$(UV) den check-run --run $$RUN && $(UV) den evaluate --run $$RUN --limit 200 --no-evidence $(SHIP_DEV) || exit 1; \
	done $(call log,$(LOCAL_LOGS)/eval.log)
	$(UV) den compare $(LOCAL_RUNS)/4-skills $(LOCAL_RUNS)/round2 --files $(SHIP_DEV) | tee -a $(LOCAL_LOGS)/eval.log
	$(UV) den predict --run $(LOCAL_RUNS)/round2 scripts/request.jsonl | tee $(LOCAL_LOGS)/predict.json

local-clean: ## delete the rehearsal's runs and logs (LOCAL_RUNS)
	rm -rf $(LOCAL_RUNS)

##@ Cloud H100 (80 GB): the real run, one target per runbook phase (docs/h100-runbook.md)

h100-setup: setup-gpu ## phases 1-2: setup-gpu, qwen3.5-4b, data/, both rounds' data shapes, then h100-doctor
	$(MAKE) model data-download MODEL=$(H100_MODEL)
	$(MAKE) data-check MODEL=$(H100_MODEL)
	$(MAKE) data-check MODEL=$(H100_MODEL) ARGS="--data --sources-cap $(CAP) --replay $(CAP) --replay-from $(KEV_SUITES)"
	$(MAKE) h100-doctor

h100-doctor: ## phase 2's last gate: CUDA, BF16, driver, Unsloth, kernels, VRAM for qwen3.5-4b, 80 GB disk
	$(MAKE) doctor MODEL=$(H100_MODEL) ARGS="--require cuda" $(call log,doctor.log)

h100-gates: ## phase 3's gates: the CUDA tests on qwen3.5-4b and both overfit gates
	$(MAKE) test-cuda MODEL=$(H100_MODEL) $(call log,test-cuda.log)
	$(UV) den overfit --model $(H100_MODEL) --n 100 --steps 300 --out overfit-lora.json
	$(UV) den train --model $(H100_MODEL) --data train/core.jsonl --overfit 100 --epochs 25 --lr 2e-4 --head-lr 1e-3 \
		--batch 4 --accum 1 --out runs/overfit $(call log,overfit-train.log)
	rm -rf runs/overfit

h100-timing: ## phase 3's timing run: 20 steps of every stage, merge checks, the time estimate: BUDGET_HOURS=5 SPENT=<min>
	$(MAKE) train-round1 MODEL=$(H100_MODEL) RUNS=runs/timing ARGS="--max-steps 20" $(call log,runs-timing.log)
	$(MAKE) train-round2 MODEL=$(H100_MODEL) RUNS=runs/timing ROUND2=runs/timing/round2 ARGS="--max-steps 20" \
		2>&1 | tee -a runs-timing.log
	$(UV) den check-run --run runs/timing/4-skills && $(UV) den check-run --run runs/timing/round2
	$(UV) den predict --run runs/timing/round2 < /dev/null
	$(UV) python -m scripts.estimate_time --budget-hours $(BUDGET_HOURS) --spent-minutes $(SPENT)
	rm -rf runs/timing

h100-round1: ## phase 4: round 1 -> runs/kev-recipe (checkpoints every 500 steps, 100 min cap a stage)
	$(MAKE) train-round1 MODEL=$(H100_MODEL) ARGS="--max-minutes 100 --save-steps 500 $(ARGS)" $(call log,runs-train.log)

h100-round2: ## phase 6: round 2 -> runs/round2 (skip it if h100-timing said so)
	$(MAKE) train-round2 MODEL=$(H100_MODEL) ARGS="--max-minutes 90 --save-steps 500 $(ARGS)" $(call log,runs-round2.log)

h100-eval: ## phase 7: eval-dev on each round, 500 records of every source, the ship decision on dev (paired too)
	( SRC=$$(cd data/clean && ls dev/sources/*/*.jsonl | tr '\n' ' '); \
	for RUN in $(RUNS)/4-skills $(patsubst %/run.json,%,$(wildcard $(ROUND2)/run.json)); do \
		$(MAKE) eval-dev RUN=$$RUN && $(UV) den evaluate --run $$RUN --limit 500 $$SRC || exit 1; \
	done; \
	if [ -f $(ROUND2)/run.json ]; then \
		$(UV) den compare $(RUNS)/4-skills $(ROUND2) --files $(SHIP_DEV) dev/transfer.jsonl $$SRC && \
		$(UV) den compare --paired $(RUNS)/4-skills $(ROUND2); \
	fi ) $(call log,eval-rounds.log)
