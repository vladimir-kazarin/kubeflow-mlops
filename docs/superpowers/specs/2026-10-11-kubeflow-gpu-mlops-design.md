# Kubeflow GPU MLOps Pipeline — Design

**Date:** 2026-10-11
**Status:** Draft, awaiting review
**Purpose:** Learning project — build a working MLOps loop (train on GPU → evaluate → gated deploy → serve) on a local Kubeflow setup.

## 1. Goals and success criteria

**Goal:** Learn core MLOps mechanics with Kubeflow by building an end-to-end pipeline that trains an image classifier on a local GPU and serves it over HTTP.

**Done when:**
1. A pipeline run is submitted from Python and visible in the KFP UI with its artifacts, metrics and confusion matrix.
2. The `train` step runs on the RTX 4060 GPU (verified via `torch.cuda.is_available()` and logs).
3. A run whose accuracy is at or above the threshold deploys a model; `make predict` returns correct-looking labels for CIFAR-10 test images.
4. A run whose accuracy is below the threshold skips `deploy`, and the previously deployed model keeps serving.
5. `make test` passes without a cluster.

**Non-goals (out of scope):** CI/CD, Katib hyperparameter tuning, Kubeflow Model Registry, monitoring/drift detection, multi-node clusters, full Kubeflow platform (dashboard, notebooks, Istio, Dex), KServe serverless mode (Knative/Istio).

## 2. Environment and constraints

- Host: Windows + WSL2 (Ubuntu 24.04), 16 cores, ~15 GB RAM available to WSL.
- GPU: NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM, driver 610.88; visible in WSL via `nvidia-smi`.
- Tools present: `docker` 29.1.3, `kind` v0.32.0, `kubectl` v1.36.3, `helm`, `python3`.
- Missing: NVIDIA Container Toolkit (`nvidia-ctk`), Docker `nvidia` runtime, `kustomize` (`kubectl apply -k` is used instead).
- Existing kind cluster `kind-asi-lab` must not be touched.
- RAM budget rules out the full Kubeflow distribution; standalone KFP + KServe raw mode is chosen to fit.
- All component versions (KFP, KServe, cert-manager, NVIDIA device plugin, PyTorch base image) are pinned to the latest stable release at implementation time and recorded in `infra/versions.env`.

## 3. Architecture

```
            ┌──────────────────────── kind cluster: kubeflow-lab (1 node, GPU) ────────────────────────┐
 you ──▶ pipeline/run.py ──▶ KFP API ──▶ prepare_data ─▶ train (GPU) ─▶ evaluate ─▶ [acc ≥ thr] ─▶ deploy
                                 │              │              │              │                     │
                                 │              └──── artifacts (Dataset, Model, Metrics) ──▶ object store (ships with KFP)
                                 │                                                                     │
 you ──▶ make predict ──▶ port-forward ──▶ KServe InferenceService (raw mode, CPU) ◀── storageUri ────┘
```

Components:
- **Infrastructure** (`infra/`): kind cluster, NVIDIA device plugin, standalone KFP, cert-manager, KServe raw mode.
- **ML library** (`src/cifar_mlops/`): pure Python logic for data, training, evaluation, deploy manifest building and the predictor. It is unit-testable without a cluster.
- **Pipeline** (`pipeline/`): thin KFP v2 component wrappers around `src/`, plus the pipeline definition and the submit script.
- **Serving** (`src/cifar_mlops/predictor.py`): KServe custom predictor using the `kserve` SDK.
- **Container image** `cifar-mlops:dev`: one image for all pipeline steps and the predictor, built on a PyTorch CUDA runtime base and loaded into kind with `kind load docker-image` (no registry).

## 4. Section 1 — Cluster and GPU setup

**Cluster:** single-node kind cluster `kubeflow-lab`. `infra/kind-config.yaml` mounts the WSL GPU interfaces (`/dev/dxg`, `/usr/lib/wsl`) into the node.

**Install order, each with a verification gate:**

| # | Step | Who | Verification |
|---|------|-----|--------------|
| 1 | Install NVIDIA Container Toolkit; configure Docker `nvidia` runtime | User (sudo) | `docker run --rm --gpus all <cuda-image> nvidia-smi` lists the RTX 4060 |
| 2 | Create kind cluster; install NVIDIA device plugin | `setup.sh` | Node advertises `nvidia.com/gpu: 1`; a smoke-test pod prints `nvidia-smi` and `torch.cuda.is_available() == True` |
| 3 | Install standalone Kubeflow Pipelines (official manifests) | `setup.sh` | All `kubeflow` namespace pods Ready; UI reachable at `localhost:8080` via port-forward |
| 4 | Install cert-manager, then KServe in RawDeployment mode | `setup.sh` | A sample `InferenceService` reaches Ready |

**Risk (highest in project):** GPU passthrough into kind on WSL2 is non-standard. Step 2 is an isolated smoke test before heavy installs. **Fallback:** if it cannot be made to work, `train` runs on CPU with identical pipeline structure (≈20–30 min vs ≈2–3 min). Switching to CPU is an explicit decision made with the user, never automatic.

**Files:** `infra/kind-config.yaml`, `infra/versions.env`, `infra/setup.sh` (idempotent, steps 2–4), `infra/teardown.sh`, `infra/gpu-smoke-test.yaml`, `infra/README.md` (user steps, expected outputs, troubleshooting).

## 5. Section 2 — Pipeline

Defined with the KFP v2 Python SDK.

```
prepare_data ──▶ train (GPU) ──▶ evaluate ──▶ dsl.If(accuracy ≥ accuracy_threshold) ──▶ deploy
```

| Step | Behavior | Outputs |
|------|----------|---------|
| `prepare_data` | Download CIFAR-10 via torchvision; produce train/val/test splits. Caching enabled. | `Dataset` artifact |
| `train` | Fine-tune ResNet-18 from ImageNet weights with mixed precision. Requests `nvidia.com/gpu: 1`. Fails fast if CUDA is unavailable. | `Model` artifact (TorchScript `model.pt` + `labels.json`), `Metrics` (per-epoch train loss, val accuracy) |
| `evaluate` | Compute test-set accuracy and confusion matrix. | `accuracy: float`, `ClassificationMetrics` (confusion matrix rendered in UI) |
| `deploy` | Build and apply/patch `InferenceService` `cifar10` with `storageUri` = this run's model artifact URI; wait for Ready with timeout. | Endpoint name |

**Parameters:** `epochs` (default 3), `learning_rate` (default 1e-3), `batch_size` (default 128), `accuracy_threshold` (default 0.85).

**RBAC:** a Role/RoleBinding grants the KFP pipeline-runner service account `get/list/create/patch/watch` on `inferenceservices.serving.kserve.io` in the serving namespace only.

**Code layout:**
- `src/cifar_mlops/data.py`, `train.py`, `evaluate.py`, `deploy.py`, `predictor.py`
- `pipeline/pipeline.py` — components + pipeline definition
- `pipeline/run.py` — compile and submit with CLI parameters, print run URL
- `Dockerfile` — single image `cifar-mlops:dev`

## 6. Section 3 — Serving

- **Mechanism:** KServe `InferenceService` in RawDeployment mode with a custom predictor container (same image) implementing `kserve.Model`: `load()` reads TorchScript + labels; `predict()` decodes base64 PNG, normalizes, runs inference, returns label + confidence.
- **Model fetch:** `storageUri` points to the model artifact in the KFP object store; KServe's storage initializer downloads it. Credentials are supplied via a Secret + ServiceAccount configured for the KFP object store's S3-compatible endpoint.
- **Compute:** CPU only. The single GPU stays reserved for training (no time-slicing).
- **API (KServe v1 protocol):**
  ```
  POST /v1/models/cifar10:predict
  {"instances": [{"image_b64": "<base64 PNG>"}]}
  → {"predictions": [{"label": "cat", "confidence": 0.93}]}
  ```
- **Updates / rollback:** each passing run patches `storageUri` → Kubernetes rolling update; a failing readiness probe leaves the old pod serving. All model artifacts are retained, so rollback = patch `storageUri` to a previous run's artifact.
- **Access:** port-forward to the predictor service; `scripts/predict.py` sends N random CIFAR-10 test images and prints predicted vs. actual labels.

## 7. Section 4 — Operations and testing

**Makefile targets:** `setup`, `image`, `run` (accepts `EPOCHS`, `LR`, `BATCH_SIZE`, `THRESHOLD`), `ui`, `predict`, `test`, `teardown`.

**Testing layers:**
1. **Unit (pytest, no cluster, seconds):** data split sizes; model output shape `(N, 10)`; accuracy-gate logic; `InferenceService` manifest builder; predictor request parsing/response shape; 1-batch CPU training smoke test.
2. **Infra verification:** the gates in §4.
3. **End-to-end (manual, documented):** run with defaults → deploy → `make predict` returns sensible labels; run with `accuracy_threshold=0.99` → `deploy` skipped, previous model still serving.

**Error handling:**
- Components raise clear, actionable errors (e.g. "CUDA not available — run the GPU smoke test in infra/README.md").
- No silent CPU fallback inside the pipeline.
- `deploy` fails the run if the `InferenceService` is not Ready within the timeout (default 5 min), so a successful run implies a working endpoint.
- `infra/README.md` troubleshooting: GPU not visible in pod, pods Pending due to memory, image not loaded into kind, port-forward conflicts.

## 8. Open risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU passthrough to kind on WSL2 fails | No GPU training | Isolated smoke test first; explicit CPU fallback decision |
| RAM pressure (KFP + KServe + training) | Pods evicted / Pending | Raw mode (no Istio/Knative); resource requests set; WSL memory can be raised in `.wslconfig` |
| Large CUDA image (several GB) | Slow `kind load` | Built once; code changes rebuild only the top layer |
| Object-store access from KServe | Model fetch fails | Verified explicitly during first deploy; credentials setup documented |
