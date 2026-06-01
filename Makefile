UV ?= uv
PYTHON_VERSION ?= 3.11
PROJECT ?= p02_policy_gradients
CONFIG ?= configs/ppo_cartpole_debug.yaml
OVERRIDES ?=

.PHONY: venv sync sync-all sync-llm lock format lint test smoke train-local modal-smoke modal-train doctor clean

venv:
	$(UV) venv --python $(PYTHON_VERSION)

sync:
	$(UV) sync --group dev --group logging --group modal

sync-llm:
	$(UV) sync --group dev --group logging --group modal --group llm

sync-all:
	$(UV) sync --all-groups

lock:
	$(UV) lock

format:
	$(UV) run --group dev ruff format rl_lab scripts

lint:
	$(UV) run --group dev ruff check rl_lab scripts

test:
	$(UV) run --group dev --group logging pytest -q

smoke:
	$(UV) run --group logging python -m rl_lab.projects.$(PROJECT).train \
		--config rl_lab/projects/$(PROJECT)/$(CONFIG) \
		--smoke \
		--overrides "$(OVERRIDES)"

train-local:
	$(UV) run --group logging python -m rl_lab.projects.$(PROJECT).train \
		--config rl_lab/projects/$(PROJECT)/$(CONFIG) \
		--overrides "$(OVERRIDES)"

modal-smoke:
	$(UV) run --group modal modal run -m rl_lab.modal.runner \
		--project $(PROJECT) \
		--config rl_lab/projects/$(PROJECT)/$(CONFIG) \
		--smoke true \
		--overrides "$(OVERRIDES)"

modal-train:
	$(UV) run --group modal modal run -m rl_lab.modal.runner \
		--project $(PROJECT) \
		--config rl_lab/projects/$(PROJECT)/$(CONFIG) \
		--smoke false \
		--overrides "$(OVERRIDES)"

doctor:
	$(UV) run --group logging python --version
	$(UV) run --group logging python -c "import torch, gymnasium; print('torch', torch.__version__); print('gymnasium', gymnasium.__version__)"

clean:
	rm -rf .pytest_cache .ruff_cache experiments/runs/tmp*
