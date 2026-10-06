# systemone: run `make help` for targets. Model targets take MODEL=<key> (see `make models`); default qwen3.5-4b.
.DEFAULT_GOAL := help
UV := uv run
MODEL ?= $(or $(SYSTEM_ONE_MODEL),qwen3.5-4b)

.PHONY: help setup check test-model list models model data data-raw-train data-raw-new data-raw-eval data-raw-bulk data-all verify normalize audit env smoke

help: ## show targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

setup: ## install Python 3.12 + locked dependencies
	uv sync

check: ## lint, strict type-check, unit tests
	$(UV) ruff check src tests
	$(UV) ruff format --check src tests
	$(UV) mypy
	$(UV) pytest -q

test-model: ## MLX vs PyTorch parity on the downloaded qwen3.5-4b (slow)
	$(UV) pytest -q -m model

models: ## list backbones of every size and the Kev reference checkpoints
	$(UV) systemone models

model: ## download MODEL (a key from `make models`, or hf:<org>/<name>@<commit>)
	$(UV) systemone model $(MODEL)

list: ## print every pinned dataset
	$(UV) systemone list

data: ## suites: Kev 1.0 train / calibration / dev / locked test (~100 MB)
	$(UV) systemone data suites

data-raw-train: ## raw sources Kev trained from, at Kev's pins (~0.8 GB)
	$(UV) systemone data raw-train

data-raw-new: ## new trainable sources aimed at Kev's weak spots (~0.3 GB)
	$(UV) systemone data raw-new

data-raw-eval: ## eval-only raw sources: transfer, devtools, breadth-v1 (~0.6 GB)
	$(UV) systemone data raw-eval

data-raw-bulk: ## CFPB, CommitPackFT, CodeReviewer, FlakeFlagger raws (~5 GB)
	$(UV) systemone data raw-bulk

data-all: ## every set except raw-bulk
	$(UV) systemone data all

verify: ## re-hash every downloaded file against locks/
	$(UV) systemone verify

normalize: ## raw sources -> canonical, leakage-free train / dev / test under data/*/sources/
	$(UV) systemone normalize

audit: ## check every record and source is fit to train on; writes reports/data-audit.json
	$(UV) systemone audit --model $(MODEL)

env: ## show the backend this machine uses (mlx | cuda | cpu)
	$(UV) systemone env

smoke: ## run MODEL once on this machine's backend
	$(UV) systemone smoke --model $(MODEL)
