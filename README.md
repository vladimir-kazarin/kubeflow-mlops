# Kubeflow GPU MLOps lab

A learning project: a Kubeflow pipeline that fine-tunes ResNet-18 on CIFAR-10 on
your local NVIDIA GPU, evaluates it, and deploys it to a KServe HTTP endpoint
**only if** it is accurate enough.

```
prepare_data ──▶ train (GPU) ──▶ evaluate ──▶ [accuracy ≥ threshold?] ──▶ deploy (KServe)
```

## Quickstart

| Step | Command |
|------|---------|
| 1. GPU for Docker (once, needs sudo) | follow `infra/README.md` → Step 1 |
| 2. Cluster + Kubeflow Pipelines + KServe | `make setup` |
| 3. Python env + unit tests | `make venv test` |
| 4. Build image and check GPU in a pod | `make image gpu-check` |
| 5. Open the KFP UI (leave running) | `make ui` → http://localhost:8080 |
| 6. Train + evaluate + deploy | `make run` (try `EPOCHS=5`, `THRESHOLD=0.9`) |
| 7. Call the model | `make predict COUNT=16` |
| Start over | `make teardown` |

## What to look at while learning

- **Runs → your run → each step.** Inputs/outputs are *artifacts* stored in KFP's
  object store. Click `metrics` on `train-op` for per-epoch loss/accuracy, and
  `confusion` on `evaluate-op` for the confusion matrix.
- **The accuracy gate.** `make run THRESHOLD=0.99` trains, evaluates, and skips
  deploy, so a worse model never replaces the one in production.
- **Serving.** `kubectl --context kind-kubeflow-lab -n serving get isvc,deploy,pods`
  shows what KServe created from one `InferenceService` object
  (`src/cifar_mlops/deploy.py`). Each passing run rolls out the new model.
- **Rollback.** Every run's model stays in the object store. Point the
  InferenceService's `STORAGE_URI` at an older run's model URI (from the UI) with
  `kubectl --context kind-kubeflow-lab -n serving edit isvc cifar10`.

## Layout

| Path | What |
|------|------|
| `infra/` | cluster/GPU/KFP/KServe setup scripts, pinned versions, troubleshooting |
| `src/cifar_mlops/` | all ML + serving + deploy logic (unit-tested, no cluster needed) |
| `pipelines/` | KFP components/pipeline (thin wrappers) and the submit CLI |
| `scripts/predict.py` | sends CIFAR-10 test images to the endpoint |
| `docs/superpowers/` | design spec and implementation plan |

## Ideas for next steps
CI/CD that compiles and submits on push · Katib hyperparameter tuning ·
Kubeflow Model Registry · request logging and drift monitoring.
