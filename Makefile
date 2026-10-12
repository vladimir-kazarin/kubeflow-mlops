include infra/versions.env

CLUSTER := kubeflow-lab
CONTEXT := kind-$(CLUSTER)
VENV := .venv
PY := $(VENV)/bin/python
IMAGE := cifar-mlops:dev
EPOCHS ?= 3
LR ?= 0.001
BATCH_SIZE ?= 128
THRESHOLD ?= 0.85
COUNT ?= 8
PREDICT_PORT := 8081

.PHONY: setup teardown ui venv test image gpu-check run predict

setup:
	./infra/setup.sh

teardown:
	./infra/teardown.sh

ui:
	kubectl --context $(CONTEXT) port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

image:
	docker build --build-arg PYTORCH_BASE_IMAGE=$(PYTORCH_BASE_IMAGE) -t $(IMAGE) .
	kind load docker-image $(IMAGE) --name $(CLUSTER)

gpu-check:
	kubectl --context $(CONTEXT) delete pod torch-gpu-check --ignore-not-found
	kubectl --context $(CONTEXT) apply -f infra/torch-gpu-check.yaml
	kubectl --context $(CONTEXT) wait --for=jsonpath='{.status.phase}'=Succeeded pod/torch-gpu-check --timeout=300s
	kubectl --context $(CONTEXT) logs torch-gpu-check
	kubectl --context $(CONTEXT) delete pod torch-gpu-check

run:
	$(PY) -m pipelines.submit --epochs $(EPOCHS) --learning-rate $(LR) --batch-size $(BATCH_SIZE) --threshold $(THRESHOLD)

predict:
	@kubectl --context $(CONTEXT) -n serving port-forward svc/cifar10-predictor $(PREDICT_PORT):80 >/dev/null 2>&1 & \
	PF_PID=$$!; trap "kill $$PF_PID" EXIT; sleep 3; \
	$(PY) scripts/predict.py --url http://localhost:$(PREDICT_PORT) --count $(COUNT)
