-include infra/versions.env

CLUSTER := kubeflow-lab
CONTEXT := kind-$(CLUSTER)
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: venv test

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q
