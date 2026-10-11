# Kubeflow GPU MLOps Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Kubeflow setup on kind (WSL2) where a KFP v2 pipeline fine-tunes ResNet-18 on CIFAR-10 on the RTX 4060 GPU, evaluates it, and, if accuracy passes a threshold, deploys it to a KServe endpoint you can call over HTTP.

**Architecture:** A single-node kind cluster `kubeflow-lab` gets GPU access through the NVIDIA container runtime and device plugin, then standalone Kubeflow Pipelines, cert-manager and KServe (RawDeployment mode). One Python package, `cifar_mlops`, holds all ML and deploy logic and is unit-tested without a cluster. Thin KFP lightweight components wrap it, and they run inside one image, `cifar-mlops:dev`, which also serves as the KServe custom predictor.

**Tech Stack:** kind, kubectl, helm, NVIDIA Container Toolkit, NVIDIA k8s-device-plugin, Kubeflow Pipelines v2 (backend + `kfp` SDK), cert-manager, KServe (`kserve` SDK), PyTorch + torchvision, kubernetes Python client, pytest, Make.

**Spec:** `docs/superpowers/specs/2026-10-11-kubeflow-gpu-mlops-design.md`

**Deliberate deviations from the spec (decided while planning):**
- Pipeline code lives in `pipelines/cifar10_pipeline.py` and `pipelines/submit.py`, not `pipeline/pipeline.py` and `pipeline/run.py`. A top-level module named `pipeline` would shadow too easily.
- `kind-config.yaml` uses NVIDIA's documented "volume-mount" trick, letting Docker's nvidia runtime inject the GPU into the node, instead of bind-mounting `/dev/dxg` and `/usr/lib/wsl` by hand. The manual mounts are the documented fallback in the troubleshooting guide.
- The `torch.cuda.is_available()` check runs in Task 10 with our own image. The Task 2 smoke test uses a small CUDA base image and `nvidia-smi`.

## Global Constraints

- Cluster name `kubeflow-lab`, kube context `kind-kubeflow-lab`. Every `kubectl`/`helm` call passes this context explicitly. The existing `kind-asi-lab` cluster must never be modified.
- All infra versions are pinned in `infra/versions.env` (resolved once in Task 2, then committed). The `kfp` SDK version is pinned in `pyproject.toml`.
- One container image `cifar-mlops:dev`, loaded with `kind load docker-image`; no registry.
- Training requests `nvidia.com/gpu: 1`. Inference runs on CPU only.
- No silent CPU fallback in `train`: if CUDA is missing it raises `RuntimeError` mentioning the GPU smoke test.
- Pipeline parameter defaults: `epochs=3`, `learning_rate=1e-3`, `batch_size=128`, `accuracy_threshold=0.85`. The gate is inclusive (`accuracy >= threshold`).
- Deploy waits at most `300` seconds for the InferenceService to become Ready.
- InferenceService name `cifar10` in namespace `serving`, annotation `serving.kserve.io/deploymentMode: RawDeployment`, service account `model-puller`.
- Predict API: `POST /v1/models/cifar10:predict` with body `{"instances": [{"image_b64": "<base64 PNG>"}]}`, response `{"predictions": [{"label": "<class>", "confidence": <float>}]}`.
- Python ≥ 3.10 (local is 3.12.3). `make test` must pass with no cluster.

## Review Focus

1. **Non-CIFAR images sent to the endpoint** (grayscale, RGBA, other sizes) should be converted to RGB 32×32 and classified, not crash. Tests are in Task 8 (`tests/test_imaging.py`).
2. **Malformed predict requests** (missing `instances`, empty list, wrong types, bad base64, too many instances, oversized payload) should return a 400 `InvalidInput` with a clear message, not a 500. Tests are in Task 8 (`tests/test_imaging.py`, `tests/test_predictor.py`).
3. **Re-running the pipeline when `cifar10` already exists** should update the InferenceService in place, not fail with 409 Conflict. A non-404 API error should propagate, not be swallowed. Tests are in Task 9 (`tests/test_deploy.py`).
4. **Accuracy exactly equal to the threshold** should deploy, and a threshold outside `[0, 1]` (or NaN) should be rejected before submitting. Tests are in Task 6 (`tests/test_evaluate.py`) and Task 11 (`tests/test_submit.py`).
5. **`epochs=0`** (the "fail the gate" demo) should run without crashing and export an untrained model. Tests are in Task 7 (`tests/test_train.py`).

---

## File Structure

```
kubeflow/
├── Makefile                          # day-to-day commands (setup, image, run, ui, predict, test, teardown)
├── pyproject.toml                    # cifar_mlops package + pinned kfp + pytest config
├── Dockerfile                        # cifar-mlops:dev (pipeline steps + predictor)
├── .dockerignore / .gitignore
├── README.md                         # project overview + learning walkthrough
├── infra/
│   ├── versions.env                  # pinned versions (KEY=VALUE, sourced by bash and make)
│   ├── kind-config.yaml              # single-node cluster with GPU injection
│   ├── gpu-smoke-test.yaml           # nvidia-smi pod
│   ├── torch-gpu-check.yaml          # torch.cuda check using cifar-mlops:dev
│   ├── sample-isvc.yaml              # KServe sanity-check InferenceService (sklearn iris)
│   ├── serving.yaml                  # serving namespace, model-puller SA, RBAC for pipeline-runner
│   ├── setup.sh                      # idempotent install of everything
│   ├── teardown.sh                   # deletes the cluster
│   └── README.md                     # user-run steps, expected output, troubleshooting
├── src/cifar_mlops/
│   ├── __init__.py
│   ├── data.py                       # Split type, train/val split, save/load, batching, CIFAR-10 download
│   ├── model.py                      # ServingModel (uint8 → logits), build/export/load
│   ├── evaluate.py                   # predictions, accuracy, confusion matrix, accuracy gate
│   ├── train.py                      # TrainConfig, device resolution, training loop
│   ├── imaging.py                    # base64 PNG encode/decode with validation
│   ├── predictor.py                  # KServe custom predictor
│   └── deploy.py                     # InferenceService manifest, apply, wait-for-ready
├── pipelines/
│   ├── __init__.py
│   ├── cifar10_pipeline.py           # KFP components + pipeline definition
│   └── submit.py                     # CLI: validate params, submit run
├── scripts/
│   └── predict.py                    # send CIFAR-10 test images to the endpoint
└── tests/
    ├── conftest.py
    ├── test_data.py  test_model.py  test_evaluate.py  test_train.py
    ├── test_imaging.py  test_predictor.py  test_deploy.py
    └── test_pipeline.py  test_submit.py
```

---

### Task 1: Host GPU runtime (user-run, needs sudo)

**Files:**
- Create: `infra/README.md`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: Docker with the `nvidia` runtime set as the default and `accept-nvidia-visible-devices-as-volume-mounts=true`. Task 2 depends on this.

- [ ] **Step 1: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
build/
dist/
pipeline.yaml
```

- [ ] **Step 2: Write `infra/README.md` with the user-run step**

````markdown
# Infrastructure

Everything runs in a dedicated kind cluster called `kubeflow-lab` (context `kind-kubeflow-lab`).
Your other clusters (e.g. `kind-asi-lab`) are never touched by these scripts.

## Step 1 — Give Docker access to the GPU (you run this, needs sudo)

Your Windows NVIDIA driver already exposes the GPU to WSL (`nvidia-smi` works).
Docker needs the NVIDIA Container Toolkit to pass it into containers.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Make nvidia the default Docker runtime (kind cannot pass --gpus to its nodes)
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
# Let kind request GPUs through a volume mount (see infra/kind-config.yaml)
sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
sudo systemctl restart docker
```

**Restarting Docker stops your other kind clusters.** Bring them back with:

```bash
docker ps -a --filter label=io.x-k8s.kind.cluster --format '{{.Names}}\t{{.Status}}'
docker start asi-lab-control-plane   # repeat for each stopped node listed above
```

**Check:**

```bash
docker info --format '{{.DefaultRuntime}}'                      # → nvidia
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L           # → GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU ...
```
````

- [ ] **Step 3: Ask the user to run Step 1 from `infra/README.md`**

Tell the user to run the commands, for example by typing `! <command>` in the prompt, and to paste the output of the two check commands. **Stop until both checks pass.** Expected: `nvidia`, and a line beginning with `GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU`.

- [ ] **Step 4: Confirm the existing cluster survived**

Run: `kubectl --context kind-asi-lab get nodes`
Expected: node `Ready`. If not, follow the "Restarting Docker" note above.

- [ ] **Step 5: Commit**

```bash
git add .gitignore infra/README.md
git commit -m "docs: add host GPU runtime setup instructions"
```

---

### Task 2: kind cluster with GPU, device plugin, smoke test

**Files:**
- Create: `infra/versions.env`
- Create: `infra/kind-config.yaml`
- Create: `infra/gpu-smoke-test.yaml`
- Create: `infra/setup.sh`
- Create: `infra/teardown.sh`
- Create: `Makefile`
- Modify: `infra/README.md` (append Step 2 and troubleshooting)

**Interfaces:**
- Consumes: Task 1 (Docker `nvidia` default runtime).
- Produces: cluster `kubeflow-lab` whose node advertises `nvidia.com/gpu: 1`. `infra/versions.env` defines the keys `KFP_VERSION`, `KFP_ENV`, `CERT_MANAGER_VERSION`, `KSERVE_VERSION`, `NVDP_VERSION`, `CUDA_SMOKE_IMAGE`, `PYTORCH_BASE_IMAGE`. `infra/setup.sh` defines the shell helpers `log`, `die`, `KCTL` (array), `CONTEXT`, `NODE`, `SERVING_NAMESPACE`, and the `main` function that Task 3 extends. Make targets: `setup`, `teardown`.

- [ ] **Step 1: Resolve current stable versions**

Run:

```bash
latest() { curl -fsSL "https://api.github.com/repos/$1/releases/latest" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])'; }
echo "KFP_VERSION=$(latest kubeflow/pipelines)"
echo "CERT_MANAGER_VERSION=$(latest cert-manager/cert-manager)"
echo "KSERVE_VERSION=$(latest kserve/kserve)"
echo "NVDP_VERSION=$(latest NVIDIA/k8s-device-plugin)"
curl -fsSL "https://hub.docker.com/v2/repositories/nvidia/cuda/tags?page_size=100&name=base-ubuntu24.04" \
  | python3 -c 'import json,sys; print(sorted((t["name"] for t in json.load(sys.stdin)["results"]), key=lambda s:[int(p) if p.isdigit() else p for p in s.split("-")[0].split(".")])[-1])'
curl -fsSL "https://hub.docker.com/v2/repositories/pytorch/pytorch/tags?page_size=100&name=runtime" \
  | python3 -c 'import json,sys; print("\n".join(sorted(t["name"] for t in json.load(sys.stdin)["results"] if "cuda12" in t["name"] and "cudnn" in t["name"])[-5:]))'
```

Expected: one tag per line. For PyTorch, pick the highest `X.Y.Z-cuda12.N-cudnnM-runtime` tag. Then check the KFP manifest env name exists at that tag:

```bash
source <(echo "KFP_VERSION=<value from above>")
curl -fsSI "https://github.com/kubeflow/pipelines/tree/${KFP_VERSION}/manifests/kustomize/env/platform-agnostic" >/dev/null && echo platform-agnostic \
  || (curl -fsSI "https://github.com/kubeflow/pipelines/tree/${KFP_VERSION}/manifests/kustomize/env/dev" >/dev/null && echo dev)
```

- [ ] **Step 2: Write `infra/versions.env` with the resolved values**

Format (substitute the exact strings printed in Step 1; values have no quotes and no spaces so both bash `source` and make `include` read them):

```bash
# Pinned versions. Resolved 2026-10-11 from GitHub/Docker Hub "latest stable".
KFP_VERSION=<kubeflow/pipelines tag>
KFP_ENV=<platform-agnostic|dev>
CERT_MANAGER_VERSION=<cert-manager tag, with leading v>
KSERVE_VERSION=<kserve tag, with leading v>
NVDP_VERSION=<k8s-device-plugin tag, with leading v>
CUDA_SMOKE_IMAGE=nvidia/cuda:<base-ubuntu24.04 tag>
PYTORCH_BASE_IMAGE=pytorch/pytorch:<runtime tag>
```

- [ ] **Step 3: Write `infra/kind-config.yaml`**

```yaml
# Single-node cluster for the Kubeflow GPU lab.
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: kubeflow-lab
nodes:
  - role: control-plane
    extraMounts:
      # With accept-nvidia-visible-devices-as-volume-mounts=true (infra/README.md Step 1),
      # this mount tells Docker's nvidia runtime to inject all GPUs into the node container.
      - hostPath: /dev/null
        containerPath: /var/run/nvidia-container-devices/all
```

- [ ] **Step 4: Write `infra/gpu-smoke-test.yaml`**

```yaml
# Proves a pod can see the GPU. __CUDA_IMAGE__ is replaced by setup.sh.
apiVersion: v1
kind: Pod
metadata:
  name: gpu-smoke-test
spec:
  restartPolicy: Never
  containers:
    - name: cuda
      image: __CUDA_IMAGE__
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
```

- [ ] **Step 5: Write `infra/setup.sh`**

```bash
#!/usr/bin/env bash
# Creates the kubeflow-lab kind cluster with GPU support.
# Safe to re-run: each step skips work that is already done.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "${SCRIPT_DIR}/versions.env"

CLUSTER_NAME="kubeflow-lab"
CONTEXT="kind-${CLUSTER_NAME}"
NODE="${CLUSTER_NAME}-control-plane"
SERVING_NAMESPACE="serving"
KCTL=(kubectl --context "${CONTEXT}")
GPU_WAIT_ATTEMPTS=30
GPU_WAIT_SECONDS=5

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

check_host_gpu() {
  log "Checking Docker GPU access"
  [[ "$(docker info --format '{{.DefaultRuntime}}')" == "nvidia" ]] \
    || die "Docker default runtime is not 'nvidia'. Complete Step 1 in infra/README.md first."
  docker run --rm --gpus all "${CUDA_SMOKE_IMAGE}" nvidia-smi -L \
    || die "'docker run --gpus all' failed. Complete Step 1 in infra/README.md first."
}

create_cluster() {
  if kind get clusters | grep -qx "${CLUSTER_NAME}"; then
    log "Cluster ${CLUSTER_NAME} already exists, skipping creation"
    return
  fi
  log "Creating kind cluster ${CLUSTER_NAME}"
  kind create cluster --config "${SCRIPT_DIR}/kind-config.yaml" --wait 120s
}

setup_node_gpu_runtime() {
  if docker exec "${NODE}" test -x /usr/bin/nvidia-ctk; then
    log "NVIDIA toolkit already present in node, skipping"
    return
  fi
  log "Installing NVIDIA container toolkit inside the kind node"
  docker exec "${NODE}" bash -euo pipefail -c '
    apt-get update -qq && apt-get install -y -qq curl gpg >/dev/null
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit >/dev/null
    nvidia-ctk runtime configure --runtime=containerd --set-as-default
    systemctl restart containerd'
  "${KCTL[@]}" wait --for=condition=Ready node --all --timeout=120s
}

wait_for_gpu_capacity() {
  local gpus
  for _ in $(seq 1 "${GPU_WAIT_ATTEMPTS}"); do
    gpus="$("${KCTL[@]}" get node "${NODE}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
    if [[ "${gpus}" =~ ^[1-9] ]]; then
      log "Node reports ${gpus} GPU(s)"
      return
    fi
    sleep "${GPU_WAIT_SECONDS}"
  done
  die "Node never reported nvidia.com/gpu. See 'GPU not visible' in infra/README.md."
}

install_device_plugin() {
  log "Installing NVIDIA device plugin ${NVDP_VERSION}"
  "${KCTL[@]}" label node "${NODE}" nvidia.com/gpu.present=true --overwrite
  helm repo add nvdp https://nvidia.github.io/k8s-device-plugin --force-update >/dev/null
  helm upgrade --install nvdp nvdp/nvidia-device-plugin \
    --kube-context "${CONTEXT}" --namespace nvidia-device-plugin --create-namespace \
    --version "${NVDP_VERSION#v}" --wait
  wait_for_gpu_capacity
}

run_gpu_smoke_test() {
  log "Running GPU smoke test pod"
  "${KCTL[@]}" delete pod gpu-smoke-test --ignore-not-found
  sed "s#__CUDA_IMAGE__#${CUDA_SMOKE_IMAGE}#" "${SCRIPT_DIR}/gpu-smoke-test.yaml" | "${KCTL[@]}" apply -f -
  if ! "${KCTL[@]}" wait --for=jsonpath='{.status.phase}'=Succeeded pod/gpu-smoke-test --timeout=300s; then
    "${KCTL[@]}" describe pod gpu-smoke-test || true
    "${KCTL[@]}" logs gpu-smoke-test || true
    die "GPU smoke test failed. See 'GPU not visible' in infra/README.md."
  fi
  "${KCTL[@]}" logs gpu-smoke-test
  "${KCTL[@]}" delete pod gpu-smoke-test
}

main() {
  check_host_gpu
  create_cluster
  setup_node_gpu_runtime
  install_device_plugin
  run_gpu_smoke_test
  log "GPU cluster ready"
}

main "$@"
```

- [ ] **Step 6: Write `infra/teardown.sh`**

```bash
#!/usr/bin/env bash
# Deletes the kubeflow-lab cluster. Other kind clusters are left alone.
set -euo pipefail
kind delete cluster --name kubeflow-lab
```

Run: `chmod +x infra/setup.sh infra/teardown.sh`

- [ ] **Step 7: Write the initial `Makefile`**

```makefile
include infra/versions.env

CLUSTER := kubeflow-lab
CONTEXT := kind-$(CLUSTER)

.PHONY: setup teardown

setup:
	./infra/setup.sh

teardown:
	./infra/teardown.sh
```

- [ ] **Step 8: Append Step 2 and troubleshooting to `infra/README.md`**

````markdown
## Step 2 — Create the cluster and install everything

```bash
make setup
```

Each stage prints `==> ...` and checks itself. The GPU stage ends with the
`nvidia-smi` table from inside a Kubernetes pod.

## Troubleshooting

### GPU not visible (node never reports `nvidia.com/gpu`, or smoke test fails)
1. Re-check Step 1: `docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L`.
2. Check the GPU reached the node: `docker exec kubeflow-lab-control-plane nvidia-smi -L`.
   If this fails, the volume-mount trick did not apply. Check
   `grep accept-nvidia /etc/nvidia-container-runtime/config.toml` shows `true`.
3. Device plugin logs: `kubectl --context kind-kubeflow-lab -n nvidia-device-plugin logs -l app.kubernetes.io/name=nvidia-device-plugin`.
4. WSL fallback: add these to the node in `infra/kind-config.yaml`, then `make teardown setup`:
   ```yaml
       - hostPath: /usr/lib/wsl
         containerPath: /usr/lib/wsl
       - hostPath: /dev/dxg
         containerPath: /dev/dxg
   ```

### Pods stuck Pending
`kubectl --context kind-kubeflow-lab describe pod <name>`. If it says
`Insufficient memory`, raise WSL memory in `%UserProfile%\.wslconfig`
(`[wsl2]` / `memory=14GB`) and run `wsl --shutdown` from Windows.

### Start over
`make teardown && make setup`
````

- [ ] **Step 9: Run setup and verify the GPU gate**

Run: `make setup`
Expected: ends with an `nvidia-smi` table listing `NVIDIA GeForce RTX 4060 Laptop GPU`, then `==> GPU cluster ready`.
Then run: `kubectl --context kind-kubeflow-lab get node kubeflow-lab-control-plane -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'`
Expected: `1`

**If this gate fails after the troubleshooting steps, stop and report to the user.** The spec makes switching to CPU fallback an explicit user decision.

- [ ] **Step 10: Verify re-running is safe**

Run: `make setup`
Expected: "already exists, skipping" and "already present in node, skipping" messages, ending in `GPU cluster ready`.

- [ ] **Step 11: Commit**

```bash
git add infra/ Makefile
git commit -m "feat: create kind cluster with GPU support and device plugin"
```

---

### Task 3: Kubeflow Pipelines, cert-manager, KServe, serving namespace

**Files:**
- Create: `infra/serving.yaml`
- Create: `infra/sample-isvc.yaml`
- Modify: `infra/setup.sh` (add functions, extend `main`)
- Modify: `Makefile` (add `ui`)
- Modify: `infra/README.md` (add checks)

**Interfaces:**
- Consumes: Task 2's `setup.sh` helpers (`log`, `die`, `KCTL`, `CONTEXT`, `SERVING_NAMESPACE`) and `versions.env` keys.
- Produces: KFP UI/API reachable via `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`. KFP pods run as SA `kubeflow/pipeline-runner`. Namespace `serving` with SA `model-puller` (credentials for the object store at `minio-service.kubeflow:9000`). Secret `serving/kfp-object-store`. RBAC lets `kubeflow/pipeline-runner` manage `inferenceservices.serving.kserve.io` in `serving`. KServe controller in RawDeployment mode. Make target `ui`.

- [ ] **Step 1: Write `infra/serving.yaml`**

```yaml
# Namespace where models are served, plus the permissions the pipeline's deploy step needs.
apiVersion: v1
kind: Namespace
metadata:
  name: serving
---
# Pods of the InferenceService run as this account; its secret lets KServe download
# model artifacts from the KFP object store.
apiVersion: v1
kind: ServiceAccount
metadata:
  name: model-puller
  namespace: serving
secrets:
  - name: kfp-object-store
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: inferenceservice-deployer
  namespace: serving
rules:
  - apiGroups: ["serving.kserve.io"]
    resources: ["inferenceservices"]
    verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pipeline-runner-deploys-models
  namespace: serving
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: inferenceservice-deployer
subjects:
  - kind: ServiceAccount
    name: pipeline-runner
    namespace: kubeflow
```

- [ ] **Step 2: Write `infra/sample-isvc.yaml`**

```yaml
# Known-good KServe example used only to verify the install, then deleted.
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
  namespace: serving
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: gs://kfserving-examples/models/sklearn/1.0/model
```

- [ ] **Step 3: Add install functions to `infra/setup.sh`**

Insert these functions directly above `main()`:

```bash
install_kfp() {
  log "Installing Kubeflow Pipelines ${KFP_VERSION} (${KFP_ENV})"
  "${KCTL[@]}" apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KFP_VERSION}"
  "${KCTL[@]}" wait --for condition=established --timeout=60s crd/applications.app.k8s.io
  "${KCTL[@]}" apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/${KFP_ENV}?ref=${KFP_VERSION}"
  log "Waiting for KFP deployments (first install pulls many images; can take 10+ minutes)"
  "${KCTL[@]}" wait --for=condition=Available deployment --all -n kubeflow --timeout=1200s
}

install_cert_manager() {
  log "Installing cert-manager ${CERT_MANAGER_VERSION}"
  "${KCTL[@]}" apply -f "https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml"
  "${KCTL[@]}" wait --for=condition=Available deployment --all -n cert-manager --timeout=300s
}

install_kserve() {
  log "Installing KServe ${KSERVE_VERSION} (RawDeployment mode)"
  helm upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd \
    --version "${KSERVE_VERSION}" --kube-context "${CONTEXT}" \
    --namespace kserve --create-namespace --wait
  helm upgrade --install kserve oci://ghcr.io/kserve/charts/kserve \
    --version "${KSERVE_VERSION}" --kube-context "${CONTEXT}" --namespace kserve \
    --set kserve.controller.deploymentMode=RawDeployment \
    --set kserve.controller.gateway.disableIngressCreation=true \
    --wait
}

configure_serving_namespace() {
  log "Configuring namespace ${SERVING_NAMESPACE}"
  local access_key secret_key
  access_key="$("${KCTL[@]}" -n kubeflow get secret mlpipeline-minio-artifact -o jsonpath='{.data.accesskey}' | base64 -d)"
  secret_key="$("${KCTL[@]}" -n kubeflow get secret mlpipeline-minio-artifact -o jsonpath='{.data.secretkey}' | base64 -d)"
  [[ -n "${access_key}" && -n "${secret_key}" ]] || die "Could not read kubeflow/mlpipeline-minio-artifact credentials"
  "${KCTL[@]}" get svc -n kubeflow minio-service >/dev/null \
    || die "kubeflow/minio-service not found; KFP object store layout changed. Check 'kubectl -n kubeflow get svc'."

  "${KCTL[@]}" apply -f "${SCRIPT_DIR}/serving.yaml"
  # Credentials go through stdin, never through command-line arguments.
  "${KCTL[@]}" apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: kfp-object-store
  namespace: ${SERVING_NAMESPACE}
  annotations:
    serving.kserve.io/s3-endpoint: minio-service.kubeflow:9000
    serving.kserve.io/s3-usehttps: "0"
    serving.kserve.io/s3-verifyssl: "0"
    serving.kserve.io/s3-region: us-east-1
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: ${access_key}
  AWS_SECRET_ACCESS_KEY: ${secret_key}
EOF
  "${KCTL[@]}" auth can-i create inferenceservices.serving.kserve.io -n "${SERVING_NAMESPACE}" \
    --as system:serviceaccount:kubeflow:pipeline-runner | grep -qx yes \
    || die "pipeline-runner cannot create InferenceServices in ${SERVING_NAMESPACE}"
}

verify_kserve() {
  log "Verifying KServe with a sample InferenceService"
  "${KCTL[@]}" apply -f "${SCRIPT_DIR}/sample-isvc.yaml"
  if ! "${KCTL[@]}" wait --for=condition=Ready isvc/sklearn-iris -n "${SERVING_NAMESPACE}" --timeout=600s; then
    "${KCTL[@]}" describe isvc/sklearn-iris -n "${SERVING_NAMESPACE}" || true
    die "Sample InferenceService did not become Ready"
  fi
  "${KCTL[@]}" delete -f "${SCRIPT_DIR}/sample-isvc.yaml"
}
```

- [ ] **Step 4: Extend `main()` in `infra/setup.sh`**

Replace the existing `main()` with:

```bash
main() {
  check_host_gpu
  create_cluster
  setup_node_gpu_runtime
  install_device_plugin
  run_gpu_smoke_test
  install_kfp
  install_cert_manager
  install_kserve
  configure_serving_namespace
  verify_kserve
  log "Setup complete. Next: 'make ui' in one terminal, then 'make venv test image'."
}
```

- [ ] **Step 5: Add the `ui` target to `Makefile`**

Change `.PHONY` to `.PHONY: setup teardown ui` and append:

```makefile
ui:
	kubectl --context $(CONTEXT) port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

- [ ] **Step 6: Run setup and verify**

Run: `make setup`
Expected: every stage passes and ends with `==> Setup complete.`

Run: `kubectl --context kind-kubeflow-lab get pods -n kubeflow`
Expected: all `Running` (or `Completed`).

Run `make ui` in a second terminal, then: `curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8080/`
Expected: `200`. Ask the user to open http://localhost:8080 and confirm they see the Kubeflow Pipelines UI.

Run: `free -g` (WSL)
Expected: `available` ≥ 6 GB. If it's less, record it; training needs ~3–6 GB (see troubleshooting).

- [ ] **Step 7: Add checks to `infra/README.md`**

Insert before `## Troubleshooting`:

````markdown
## What "done" looks like

```bash
kubectl --context kind-kubeflow-lab get pods -n kubeflow        # all Running
kubectl --context kind-kubeflow-lab get pods -n kserve          # kserve-controller-manager Running
make ui                                                          # then open http://localhost:8080
```

### Object store credentials
KFP stores every artifact (datasets, models, metrics) in its own S3-compatible
store (`minio-service.kubeflow:9000`). `setup.sh` copies its credentials into
`serving/kfp-object-store` so KServe can download trained models from there.
````

- [ ] **Step 8: Commit**

```bash
git add infra/ Makefile
git commit -m "feat: install Kubeflow Pipelines and KServe in raw deployment mode"
```

---

### Task 4: Python package scaffold and data module

**Files:**
- Create: `pyproject.toml`
- Create: `src/cifar_mlops/__init__.py`
- Create: `src/cifar_mlops/data.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data.py`
- Modify: `Makefile` (add `venv`, `test`)

**Interfaces:**
- Consumes: nothing.
- Produces (in `cifar_mlops.data`):
  - `CIFAR10_CLASSES: tuple[str, ...]` (10 names), `LABELS_FILE = "labels.json"`, `SPLIT_NAMES = ("train", "val", "test")`
  - `@dataclass(frozen=True) class Split(images: torch.Tensor, labels: torch.Tensor)`, where images are uint8 `(N, 3, H, W)` and labels int64 `(N,)`; `len(split)`
  - `split_train_val(full: Split, val_fraction: float, seed: int) -> tuple[Split, Split]`
  - `save_split(split: Split, path: Path) -> None`, `load_split(path: Path) -> Split`
  - `iterate_batches(split: Split, batch_size: int, shuffle: bool, generator: torch.Generator | None = None) -> Iterator[tuple[Tensor, Tensor]]`
  - `prepare_cifar10(output_dir: Path, val_fraction: float = 0.1, seed: int = 42) -> dict[str, int]`, which writes `train.pt`, `val.pt`, `test.pt`, `labels.json`
- Produces (in `tests/conftest.py`): fixture `tiny_split` with 20 images (3×32×32), where `images[i, 0, 0, 0] == i`.
- Make targets: `venv`, `test`.

- [ ] **Step 1: Resolve the kfp SDK version**

Run: `uv pip compile --quiet - <<<"kfp" 2>/dev/null | grep '^kfp==' || pip index versions kfp 2>/dev/null | head -1`
Expected: a line such as `kfp==2.N.M` or `kfp (2.N.M)`. Use that exact version in Step 2.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "cifar-mlops"
version = "0.1.0"
description = "CIFAR-10 GPU training, evaluation and serving on Kubeflow"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "torchvision",
    "numpy",
    "pillow",
    "kubernetes",
    "kserve",
    "kfp==<version from Step 1>",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "."]
```

- [ ] **Step 3: Add `venv` and `test` targets to `Makefile`**

Add below the `CONTEXT :=` line:

```makefile
VENV := .venv
PY := $(VENV)/bin/python
```

Change `.PHONY` to `.PHONY: setup teardown ui venv test` and append:

```makefile
venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q
```

- [ ] **Step 4: Create the package and test fixture**

`src/cifar_mlops/__init__.py`:

```python
"""CIFAR-10 MLOps: training, evaluation, serving and deployment logic used by the Kubeflow pipeline."""
```

`tests/conftest.py`:

```python
import pytest
import torch

from cifar_mlops.data import Split

TINY_SPLIT_SIZE = 20


@pytest.fixture
def tiny_split() -> Split:
    """20 random 32x32 RGB images; images[i, 0, 0, 0] == i so tests can track identity."""
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (TINY_SPLIT_SIZE, 3, 32, 32), dtype=torch.uint8, generator=generator)
    images[:, 0, 0, 0] = torch.arange(TINY_SPLIT_SIZE, dtype=torch.uint8)
    labels = torch.randint(0, 10, (TINY_SPLIT_SIZE,), dtype=torch.int64, generator=generator)
    return Split(images=images, labels=labels)


def image_ids(split: Split) -> list[int]:
    return split.images[:, 0, 0, 0].tolist()
```

- [ ] **Step 5: Write the failing tests `tests/test_data.py`**

```python
import pytest
import torch

from cifar_mlops.data import Split, iterate_batches, load_split, save_split, split_train_val
from tests.conftest import TINY_SPLIT_SIZE, image_ids


def test_split_rejects_float_images():
    with pytest.raises(ValueError, match="uint8"):
        Split(images=torch.zeros(2, 3, 32, 32), labels=torch.zeros(2, dtype=torch.int64))


def test_split_rejects_label_count_mismatch():
    with pytest.raises(ValueError, match="labels"):
        Split(images=torch.zeros(2, 3, 32, 32, dtype=torch.uint8), labels=torch.zeros(3, dtype=torch.int64))


def test_split_train_val_sizes_are_disjoint_and_complete(tiny_split):
    train, val = split_train_val(tiny_split, val_fraction=0.25, seed=0)

    assert (len(train), len(val)) == (15, 5)
    assert set(image_ids(train)).isdisjoint(image_ids(val))
    assert sorted(image_ids(train) + image_ids(val)) == list(range(TINY_SPLIT_SIZE))


def test_split_train_val_is_deterministic_for_a_seed(tiny_split):
    first = split_train_val(tiny_split, val_fraction=0.25, seed=7)
    second = split_train_val(tiny_split, val_fraction=0.25, seed=7)

    assert image_ids(first[1]) == image_ids(second[1])


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5, 0.01])
def test_split_train_val_rejects_fractions_that_leave_an_empty_split(tiny_split, fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        split_train_val(tiny_split, val_fraction=fraction, seed=0)


def test_save_and_load_split_round_trip(tiny_split, tmp_path):
    path = tmp_path / "train.pt"

    save_split(tiny_split, path)
    loaded = load_split(path)

    assert torch.equal(loaded.images, tiny_split.images)
    assert torch.equal(loaded.labels, tiny_split.labels)


def test_load_split_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        load_split(tmp_path / "missing.pt")


def test_iterate_batches_covers_every_item_once_with_partial_last_batch(tiny_split):
    batches = list(iterate_batches(tiny_split, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(1)))

    assert [len(labels) for _, labels in batches] == [8, 8, 4]
    seen = [i for images, _ in batches for i in images[:, 0, 0, 0].tolist()]
    assert sorted(seen) == list(range(TINY_SPLIT_SIZE))


def test_iterate_batches_without_shuffle_keeps_order(tiny_split):
    batches = list(iterate_batches(tiny_split, batch_size=64, shuffle=False))

    assert len(batches) == 1
    assert batches[0][0][:, 0, 0, 0].tolist() == list(range(TINY_SPLIT_SIZE))


def test_iterate_batches_rejects_non_positive_batch_size(tiny_split):
    with pytest.raises(ValueError, match="batch_size"):
        list(iterate_batches(tiny_split, batch_size=0, shuffle=False))
```

- [ ] **Step 6: Create the venv and run the tests to verify they fail**

Run: `make venv && .venv/bin/python -m pytest tests/test_data.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'cifar_mlops.data'` (or ImportError).

- [ ] **Step 7: Implement `src/cifar_mlops/data.py`**

```python
"""CIFAR-10 download, train/val/test splits, and batching over in-memory uint8 tensors."""
from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
LABELS_FILE = "labels.json"


@dataclass(frozen=True)
class Split:
    """A set of images (uint8, N x 3 x H x W) with integer class labels (int64, N)."""

    images: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        if self.images.dtype != torch.uint8 or self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError(
                f"images must be uint8 with shape (N, 3, H, W); got {self.images.dtype} {tuple(self.images.shape)}"
            )
        if self.labels.ndim != 1 or self.labels.shape[0] != self.images.shape[0]:
            raise ValueError(
                f"labels must have shape ({self.images.shape[0]},); got {tuple(self.labels.shape)}"
            )

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def split_train_val(full: Split, val_fraction: float, seed: int) -> tuple[Split, Split]:
    """Randomly split into (train, val); the same seed always gives the same split."""
    n_total = len(full)
    n_val = round(n_total * val_fraction)
    if not 0.0 < val_fraction < 1.0 or n_val == 0 or n_val == n_total:
        raise ValueError(f"val_fraction={val_fraction} leaves an empty split for {n_total} items")
    order = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = order[:n_val], order[n_val:]
    return (
        Split(images=full.images[train_idx], labels=full.labels[train_idx]),
        Split(images=full.images[val_idx], labels=full.labels[val_idx]),
    )


def save_split(split: Split, path: Path) -> None:
    torch.save({"images": split.images, "labels": split.labels}, path)


def load_split(path: Path) -> Split:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    payload = torch.load(path, weights_only=True)
    return Split(images=payload["images"], labels=payload["labels"])


def iterate_batches(
    split: Split,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (images, labels) batches; the last batch may be smaller."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    n_total = len(split)
    order = torch.randperm(n_total, generator=generator) if shuffle else torch.arange(n_total)
    for start in range(0, n_total, batch_size):
        idx = order[start:start + batch_size]
        yield split.images[idx], split.labels[idx]


def _split_from_torchvision(dataset) -> Split:
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(dataset.targets, dtype=torch.int64)
    return Split(images=images, labels=labels)


def prepare_cifar10(output_dir: Path, val_fraction: float = 0.1, seed: int = 42) -> dict[str, int]:
    """Download CIFAR-10 and write train/val/test splits plus labels.json into output_dir."""
    from torchvision.datasets import CIFAR10

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as download_dir:
        train_full = _split_from_torchvision(CIFAR10(download_dir, train=True, download=True))
        test = _split_from_torchvision(CIFAR10(download_dir, train=False, download=True))
    train, val = split_train_val(train_full, val_fraction=val_fraction, seed=seed)
    splits = dict(zip(SPLIT_NAMES, (train, val, test)))
    for name, split in splits.items():
        save_split(split, output_dir / f"{name}.pt")
    (output_dir / LABELS_FILE).write_text(json.dumps(list(CIFAR10_CLASSES)))
    return {name: len(split) for name, split in splits.items()}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `make test`
Expected: all tests in `tests/test_data.py` PASS.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml Makefile src/ tests/
git commit -m "feat: add CIFAR-10 data splits and batching"
```

---

### Task 5: Model module

**Files:**
- Create: `src/cifar_mlops/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `cifar_mlops.data.LABELS_FILE`.
- Produces (in `cifar_mlops.model`):
  - Constants `IMAGE_SIZE = 128`, `NUM_CLASSES = 10`, `MODEL_FILE = "model.pt"`
  - `class ServingModel(nn.Module)`: `forward(images: uint8 Tensor (N, 3, H, W) of any H, W) -> float logits (N, 10)`. Resizing and normalization happen inside the model.
  - `build_model(pretrained: bool) -> ServingModel`
  - `export_model(model: nn.Module, labels: Sequence[str], output_dir: Path) -> None`, which writes `model.pt` (TorchScript) and `labels.json`
  - `load_exported(model_dir: Path) -> tuple[torch.jit.ScriptModule, list[str]]`

- [ ] **Step 1: Write the failing tests `tests/test_model.py`**

```python
import pytest
import torch

from cifar_mlops.data import CIFAR10_CLASSES
from cifar_mlops.model import NUM_CLASSES, build_model, export_model, load_exported


def _uint8_images(n: int, height: int = 32, width: int = 32) -> torch.Tensor:
    return torch.randint(0, 256, (n, 3, height, width), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))


def test_model_returns_ten_logits_per_uint8_image():
    model = build_model(pretrained=False).eval()

    logits = model(_uint8_images(2))

    assert logits.shape == (2, NUM_CLASSES)
    assert logits.dtype == torch.float32


def test_model_accepts_non_cifar_image_sizes():
    model = build_model(pretrained=False).eval()

    assert model(_uint8_images(1, height=48, width=64)).shape == (1, NUM_CLASSES)


def test_export_and_load_round_trip_gives_identical_outputs(tmp_path):
    model = build_model(pretrained=False).eval()
    images = _uint8_images(3)
    expected = model(images)

    export_model(model, CIFAR10_CLASSES, tmp_path / "model")
    loaded, labels = load_exported(tmp_path / "model")

    assert labels == list(CIFAR10_CLASSES)
    assert torch.allclose(loaded(images), expected, atol=1e-5)


def test_load_exported_missing_model_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.pt"):
        load_exported(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cifar_mlops.model'`.

- [ ] **Step 3: Implement `src/cifar_mlops/model.py`**

```python
"""ResNet-18 classifier that takes raw uint8 images, so training and serving share one preprocessing path."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from cifar_mlops.data import LABELS_FILE

IMAGE_SIZE = 128
NUM_CLASSES = 10
MODEL_FILE = "model.pt"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ServingModel(nn.Module):
    """uint8 images (N, 3, H, W) -> logits (N, NUM_CLASSES); resizing and normalization happen inside."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float() / 255.0
        x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)


def build_model(pretrained: bool) -> ServingModel:
    """ResNet-18 with a 10-class head; pretrained=True starts from ImageNet weights."""
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
    return ServingModel(backbone)


def export_model(model: nn.Module, labels: Sequence[str], output_dir: Path) -> None:
    """Write a CPU TorchScript model and its class labels; this directory is what KServe serves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.to("cpu").eval())
    scripted.save(str(output_dir / MODEL_FILE))
    (output_dir / LABELS_FILE).write_text(json.dumps(list(labels)))


def load_exported(model_dir: Path) -> tuple[torch.jit.ScriptModule, list[str]]:
    model_path, labels_path = model_dir / MODEL_FILE, model_dir / LABELS_FILE
    for path in (model_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Exported model is missing {path}")
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    return model, json.loads(labels_path.read_text())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `make test`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cifar_mlops/model.py tests/test_model.py
git commit -m "feat: add ResNet-18 serving model with TorchScript export"
```

---

### Task 6: Evaluation and accuracy gate

**Files:**
- Create: `src/cifar_mlops/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Split`, `iterate_batches` (Task 4); `build_model` (Task 5, tests only).
- Produces (in `cifar_mlops.evaluate`):
  - `predict_labels(model: Callable[[Tensor], Tensor], split: Split, batch_size: int, device: torch.device) -> Tensor` (int64, on CPU)
  - `accuracy(y_true: Tensor, y_pred: Tensor) -> float`
  - `evaluate_accuracy(model, split: Split, batch_size: int, device: torch.device) -> float`
  - `confusion_matrix(y_true: Tensor, y_pred: Tensor, num_classes: int) -> list[list[int]]` (rows are true labels, columns are predicted labels)
  - `validate_threshold(threshold: float) -> float` (raises `ValueError` unless `0 <= threshold <= 1`)
  - `passes_gate(accuracy_value: float, threshold: float) -> bool` (inclusive)

- [ ] **Step 1: Write the failing tests `tests/test_evaluate.py`**

```python
import math

import pytest
import torch

from cifar_mlops.evaluate import (
    accuracy,
    confusion_matrix,
    evaluate_accuracy,
    passes_gate,
    predict_labels,
    validate_threshold,
)
from cifar_mlops.model import build_model

CPU = torch.device("cpu")


def test_accuracy_is_fraction_of_matches():
    assert accuracy(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 0, 3])) == pytest.approx(0.75)


def test_accuracy_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        accuracy(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))


def test_accuracy_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        accuracy(torch.tensor([0, 1]), torch.tensor([0]))


def test_confusion_matrix_rows_are_true_labels():
    matrix = confusion_matrix(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1]), num_classes=2)

    assert matrix == [[1, 1], [0, 1]]


def test_passes_gate_is_inclusive_at_the_threshold():
    assert passes_gate(0.85, 0.85) is True
    assert passes_gate(0.8499, 0.85) is False


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan])
def test_validate_threshold_rejects_values_outside_unit_interval(threshold):
    with pytest.raises(ValueError, match="threshold"):
        validate_threshold(threshold)


def test_predict_labels_returns_one_label_per_image(tiny_split):
    model = build_model(pretrained=False)

    predictions = predict_labels(model, tiny_split, batch_size=8, device=CPU)

    assert predictions.shape == (len(tiny_split),)
    assert predictions.dtype == torch.int64
    assert int(predictions.min()) >= 0 and int(predictions.max()) <= 9


def test_evaluate_accuracy_is_between_zero_and_one(tiny_split):
    value = evaluate_accuracy(build_model(pretrained=False), tiny_split, batch_size=8, device=CPU)

    assert 0.0 <= value <= 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cifar_mlops.evaluate'`.

- [ ] **Step 3: Implement `src/cifar_mlops/evaluate.py`**

```python
"""Model evaluation: predictions, accuracy, confusion matrix, and the deploy gate."""
from __future__ import annotations

from collections.abc import Callable

import torch

from cifar_mlops.data import Split, iterate_batches


def predict_labels(
    model: Callable[[torch.Tensor], torch.Tensor],
    split: Split,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Predicted class index for every image in split, returned on CPU."""
    if hasattr(model, "eval"):
        model.eval()
    predictions = []
    with torch.inference_mode():
        for images, _ in iterate_batches(split, batch_size, shuffle=False):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            predictions.append(logits.argmax(dim=1).cpu())
    return torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.int64)


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same length; got {tuple(y_true.shape)} and {tuple(y_pred.shape)}")
    if y_true.numel() == 0:
        raise ValueError("cannot compute accuracy of an empty set")
    return (y_true == y_pred).float().mean().item()


def evaluate_accuracy(model, split: Split, batch_size: int, device: torch.device) -> float:
    return accuracy(split.labels, predict_labels(model, split, batch_size, device))


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> list[list[int]]:
    """counts[i][j] = number of images of true class i predicted as class j."""
    flat_index = y_true.to(torch.int64) * num_classes + y_pred.to(torch.int64)
    counts = torch.bincount(flat_index, minlength=num_classes * num_classes)
    return counts.view(num_classes, num_classes).tolist()


def validate_threshold(threshold: float) -> float:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"accuracy threshold must be between 0 and 1; got {threshold}")
    return threshold


def passes_gate(accuracy_value: float, threshold: float) -> bool:
    """True when the model is good enough to deploy (accuracy >= threshold)."""
    return accuracy_value >= validate_threshold(threshold)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `make test`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cifar_mlops/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation metrics and inclusive accuracy gate"
```

---

### Task 7: Training loop

**Files:**
- Create: `src/cifar_mlops/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `Split`, `iterate_batches` (Task 4); `build_model`, `ServingModel` (Task 5); `evaluate_accuracy` (Task 6).
- Produces (in `cifar_mlops.train`):
  - `@dataclass(frozen=True) class TrainConfig(epochs: int, learning_rate: float, batch_size: int, pretrained: bool = True, seed: int = 42)`, which validates on construction
  - `@dataclass(frozen=True) class EpochResult(epoch: int, train_loss: float, val_accuracy: float)`
  - `GPU_MISSING_MESSAGE: str`
  - `resolve_device(require_gpu: bool) -> torch.device`
  - `random_flip(images: Tensor, generator: torch.Generator) -> Tensor`
  - `train_model(train: Split, val: Split, config: TrainConfig, device: torch.device, on_epoch_end: Callable[[EpochResult], None] | None = None) -> tuple[ServingModel, list[EpochResult]]`

- [ ] **Step 1: Write the failing tests `tests/test_train.py`**

```python
import math

import pytest
import torch

from cifar_mlops.model import NUM_CLASSES
from cifar_mlops.train import TrainConfig, random_flip, resolve_device, train_model

CPU = torch.device("cpu")


@pytest.mark.parametrize(
    "overrides",
    [{"epochs": -1}, {"learning_rate": 0.0}, {"learning_rate": -1e-3}, {"batch_size": 0}],
)
def test_train_config_rejects_invalid_values(overrides):
    values = {"epochs": 1, "learning_rate": 1e-3, "batch_size": 8, **overrides}

    with pytest.raises(ValueError):
        TrainConfig(**values)


def test_resolve_device_raises_when_gpu_required_but_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device(require_gpu=True)


def test_resolve_device_uses_cpu_when_gpu_optional_and_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device(require_gpu=False) == CPU


def test_random_flip_returns_each_image_either_unchanged_or_mirrored(tiny_split):
    flipped = random_flip(tiny_split.images, torch.Generator().manual_seed(0))

    for original, result in zip(tiny_split.images, flipped):
        assert torch.equal(result, original) or torch.equal(result, original.flip(-1))


def test_train_model_one_epoch_on_cpu_reports_history(tiny_split):
    config = TrainConfig(epochs=1, learning_rate=1e-3, batch_size=8, pretrained=False)
    reported = []

    model, history = train_model(tiny_split, tiny_split, config, CPU, on_epoch_end=reported.append)

    assert len(history) == 1 and history == reported
    assert history[0].epoch == 1
    assert math.isfinite(history[0].train_loss)
    assert 0.0 <= history[0].val_accuracy <= 1.0
    assert model(tiny_split.images[:2]).shape == (2, NUM_CLASSES)


def test_train_model_zero_epochs_returns_untrained_model_without_error(tiny_split):
    config = TrainConfig(epochs=0, learning_rate=1e-3, batch_size=8, pretrained=False)

    model, history = train_model(tiny_split, tiny_split, config, CPU)

    assert history == []
    assert model(tiny_split.images[:1]).shape == (1, NUM_CLASSES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cifar_mlops.train'`.

- [ ] **Step 3: Implement `src/cifar_mlops/train.py`**

```python
"""Fine-tune the ResNet-18 ServingModel on CIFAR-10 with mixed precision on GPU."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from cifar_mlops.data import Split, iterate_batches
from cifar_mlops.evaluate import evaluate_accuracy
from cifar_mlops.model import ServingModel, build_model

GPU_MISSING_MESSAGE = (
    "GPU was requested but CUDA is not available in this container. "
    "Run the GPU smoke test in infra/README.md (make gpu-check)."
)
FLIP_PROBABILITY = 0.5


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    batch_size: int
    pretrained: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0; got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0; got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {self.batch_size}")


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    val_accuracy: float


def resolve_device(require_gpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError(GPU_MISSING_MESSAGE)
    return torch.device("cpu")


def random_flip(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Mirror each image horizontally with probability FLIP_PROBABILITY (simple augmentation)."""
    mask = torch.rand(images.shape[0], generator=generator) < FLIP_PROBABILITY
    return torch.where(mask.view(-1, 1, 1, 1), images.flip(-1), images)


def _train_one_epoch(
    model: ServingModel,
    train: Split,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    total_loss, total_seen = 0.0, 0
    for images, labels in iterate_batches(train, config.batch_size, shuffle=True, generator=generator):
        images = random_flip(images, generator).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.shape[0]
        total_seen += labels.shape[0]
    return total_loss / max(total_seen, 1)


def train_model(
    train: Split,
    val: Split,
    config: TrainConfig,
    device: torch.device,
    on_epoch_end: Callable[[EpochResult], None] | None = None,
) -> tuple[ServingModel, list[EpochResult]]:
    """Train for config.epochs epochs; epochs=0 returns the initial (untrained) model."""
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = build_model(pretrained=config.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[EpochResult] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(model, train, config, optimizer, scaler, device, generator)
        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_accuracy=evaluate_accuracy(model, val, config.batch_size, device),
        )
        history = [*history, result]
        if on_epoch_end is not None:
            on_epoch_end(result)
    return model, history
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `make test`
Expected: all PASS. The training tests run ResNet-18 at 128 px on CPU for 20 images; this should take under ~30 s.

- [ ] **Step 5: Commit**

```bash
git add src/cifar_mlops/train.py tests/test_train.py
git commit -m "feat: add mixed-precision training loop with GPU requirement check"
```

---

### Task 8: Image encoding and KServe predictor

**Files:**
- Create: `src/cifar_mlops/imaging.py`
- Create: `src/cifar_mlops/predictor.py`
- Create: `tests/test_imaging.py`
- Create: `tests/test_predictor.py`

**Interfaces:**
- Consumes: `load_exported`, `build_model`, `export_model` (Task 5); `CIFAR10_CLASSES` (Task 4).
- Produces:
  - `cifar_mlops.imaging`: `CIFAR_SIDE = 32`, `MAX_IMAGE_B64_CHARS = 2_000_000`, `encode_png_b64(image_hwc: np.ndarray) -> str`, `decode_image_b64(data: str) -> Tensor` (uint8 `(3, 32, 32)`; raises `ValueError`)
  - `cifar_mlops.predictor`: `MAX_INSTANCES = 64`, `parse_instances(payload) -> list[str]`, `format_predictions(probabilities: Tensor, labels: Sequence[str]) -> list[dict]`, `classify(model, labels, payload) -> dict`, `class Cifar10Model(kserve.Model)` with `__init__(name: str, model_dir: Path)`, `load() -> bool`, `predict(payload, headers=None) -> dict`. Entrypoint `python -m cifar_mlops.predictor --model_name cifar10 --model_dir /mnt/models`.

- [ ] **Step 1: Write the failing tests `tests/test_imaging.py`**

```python
import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from cifar_mlops.imaging import MAX_IMAGE_B64_CHARS, decode_image_b64, encode_png_b64


def _pil_to_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_round_trip_preserves_a_cifar_image_exactly():
    image = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)

    decoded = decode_image_b64(encode_png_b64(image))

    assert torch.equal(decoded, torch.from_numpy(image).permute(2, 0, 1))


def test_decode_converts_grayscale_of_any_size_to_rgb_32x32():
    decoded = decode_image_b64(_pil_to_b64(Image.new("L", (64, 48), color=128)))

    assert decoded.shape == (3, 32, 32)
    assert decoded.dtype == torch.uint8


def test_decode_drops_alpha_channel():
    decoded = decode_image_b64(_pil_to_b64(Image.new("RGBA", (32, 32), color=(10, 20, 30, 40))))

    assert decoded.shape == (3, 32, 32)
    assert decoded[:, 0, 0].tolist() == [10, 20, 30]


@pytest.mark.parametrize("data", ["not base64!!", base64.b64encode(b"hello").decode("ascii"), ""])
def test_decode_rejects_data_that_is_not_an_image(data):
    with pytest.raises(ValueError, match="image_b64"):
        decode_image_b64(data)


def test_decode_rejects_oversized_payload():
    with pytest.raises(ValueError, match="too large"):
        decode_image_b64("A" * (MAX_IMAGE_B64_CHARS + 4))
```

- [ ] **Step 2: Write the failing tests `tests/test_predictor.py`**

```python
import numpy as np
import pytest
import torch
from kserve.errors import InvalidInput

from cifar_mlops.data import CIFAR10_CLASSES
from cifar_mlops.imaging import encode_png_b64
from cifar_mlops.model import build_model, export_model
from cifar_mlops.predictor import MAX_INSTANCES, Cifar10Model, format_predictions, parse_instances

BLACK_IMAGE_B64 = encode_png_b64(np.zeros((32, 32, 3), dtype=np.uint8))


@pytest.fixture
def exported_dir(tmp_path):
    model_dir = tmp_path / "model"
    export_model(build_model(pretrained=False).eval(), CIFAR10_CLASSES, model_dir)
    return model_dir


def test_parse_instances_returns_the_encoded_images():
    payload = {"instances": [{"image_b64": "a"}, {"image_b64": "b"}]}

    assert parse_instances(payload) == ["a", "b"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"instances": []},
        {"instances": "not-a-list"},
        {"instances": [{}]},
        {"instances": [{"image_b64": 5}]},
        {"instances": ["raw-string"]},
    ],
)
def test_parse_instances_rejects_malformed_payloads(payload):
    with pytest.raises(ValueError):
        parse_instances(payload)


def test_parse_instances_rejects_too_many_images():
    payload = {"instances": [{"image_b64": "a"}] * (MAX_INSTANCES + 1)}

    with pytest.raises(ValueError, match=str(MAX_INSTANCES)):
        parse_instances(payload)


def test_format_predictions_picks_the_most_likely_class():
    probabilities = torch.tensor([[0.1, 0.9], [0.7, 0.3]])

    assert format_predictions(probabilities, ["cat", "dog"]) == [
        {"label": "dog", "confidence": 0.9},
        {"label": "cat", "confidence": 0.7},
    ]


def test_model_predict_returns_a_label_and_confidence_per_image(exported_dir):
    model = Cifar10Model("cifar10", exported_dir)
    assert model.load() is True

    result = model.predict({"instances": [{"image_b64": BLACK_IMAGE_B64}] * 2})

    assert len(result["predictions"]) == 2
    for prediction in result["predictions"]:
        assert prediction["label"] in CIFAR10_CLASSES
        assert 0.0 <= prediction["confidence"] <= 1.0


@pytest.mark.parametrize(
    "payload",
    [{"instances": []}, {"instances": [{"image_b64": "not base64!!"}]}],
)
def test_model_predict_turns_bad_requests_into_invalid_input(exported_dir, payload):
    model = Cifar10Model("cifar10", exported_dir)
    model.load()

    with pytest.raises(InvalidInput):
        model.predict(payload)


def test_model_load_reports_missing_model_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.pt"):
        Cifar10Model("cifar10", tmp_path).load()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_imaging.py tests/test_predictor.py -q`
Expected: FAIL with `ModuleNotFoundError` for `cifar_mlops.imaging` / `cifar_mlops.predictor`.

- [ ] **Step 4: Implement `src/cifar_mlops/imaging.py`**

```python
"""Base64 PNG <-> uint8 tensor conversion shared by the predictor and the client script."""
from __future__ import annotations

import base64
import binascii
import io

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

CIFAR_SIDE = 32
MAX_IMAGE_B64_CHARS = 2_000_000  # ~1.5 MB decoded; far above any CIFAR-sized image


def encode_png_b64(image_hwc: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 array as a base64 PNG string."""
    buffer = io.BytesIO()
    Image.fromarray(image_hwc).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image_b64(data: str) -> torch.Tensor:
    """Decode a base64 image of any mode/size into a uint8 (3, 32, 32) RGB tensor."""
    if len(data) > MAX_IMAGE_B64_CHARS:
        raise ValueError(f"image_b64 is too large ({len(data)} chars; max {MAX_IMAGE_B64_CHARS})")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
        image.load()
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as err:
        raise ValueError("image_b64 is not a valid base64-encoded image") from err
    rgb = image.convert("RGB").resize((CIFAR_SIDE, CIFAR_SIDE), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(rgb, dtype=np.uint8)).permute(2, 0, 1).contiguous()
```

- [ ] **Step 5: Implement `src/cifar_mlops/predictor.py`**

```python
"""KServe custom predictor: base64 PNG images in, CIFAR-10 labels with confidence out."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from kserve import Model, ModelServer, model_server
from kserve.errors import InvalidInput

from cifar_mlops.imaging import decode_image_b64
from cifar_mlops.model import load_exported

MAX_INSTANCES = 64
REQUEST_SHAPE_HINT = 'expected {"instances": [{"image_b64": "<base64 PNG>"}]}'


def parse_instances(payload) -> list[str]:
    """Validate the request body and return the base64 image strings."""
    if not isinstance(payload, dict) or "instances" not in payload:
        raise ValueError(f"Request body is missing 'instances'; {REQUEST_SHAPE_HINT}")
    instances = payload["instances"]
    if not isinstance(instances, list) or not instances:
        raise ValueError(f"'instances' must be a non-empty list; {REQUEST_SHAPE_HINT}")
    if len(instances) > MAX_INSTANCES:
        raise ValueError(f"At most {MAX_INSTANCES} instances per request; got {len(instances)}")
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict) or not isinstance(instance.get("image_b64"), str):
            raise ValueError(f"instances[{index}] must be an object with a string 'image_b64'")
    return [instance["image_b64"] for instance in instances]


def format_predictions(probabilities: torch.Tensor, labels: Sequence[str]) -> list[dict]:
    confidences, indices = probabilities.max(dim=1)
    return [
        {"label": labels[index], "confidence": round(confidence, 4)}
        for confidence, index in zip(confidences.tolist(), indices.tolist())
    ]


def classify(model, labels: Sequence[str], payload) -> dict:
    batch = torch.stack([decode_image_b64(encoded) for encoded in parse_instances(payload)])
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch).float(), dim=1)
    return {"predictions": format_predictions(probabilities, labels)}


class Cifar10Model(Model):
    def __init__(self, name: str, model_dir: Path) -> None:
        super().__init__(name)
        self.model_dir = model_dir
        self._model = None
        self._labels: list[str] = []

    def load(self) -> bool:
        self._model, self._labels = load_exported(self.model_dir)
        self.ready = True
        return self.ready

    def predict(self, payload, headers=None) -> dict:
        try:
            return classify(self._model, self._labels, payload)
        except ValueError as err:
            raise InvalidInput(str(err)) from err


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    parser.add_argument("--model_dir", type=Path, default=Path("/mnt/models"))
    args, _ = parser.parse_known_args(argv)
    model = Cifar10Model(args.model_name, args.model_dir)
    model.load()
    ModelServer().start([model])


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `make test`
Expected: all PASS. If `kserve.model_server.parser` has no `--model_name` option in the installed kserve version, `main` would fail at startup. Check with `.venv/bin/python -c "from kserve import model_server; print('--model_name' in model_server.parser.format_help())"`, which should print `True`. If it prints `False`, add `parser.add_argument("--model_name", default="cifar10")` in `main`.

- [ ] **Step 7: Commit**

```bash
git add src/cifar_mlops/imaging.py src/cifar_mlops/predictor.py tests/test_imaging.py tests/test_predictor.py
git commit -m "feat: add KServe custom predictor with request validation"
```

---

### Task 9: Deploy logic (InferenceService manifest, apply, wait)

**Files:**
- Create: `src/cifar_mlops/deploy.py`
- Create: `tests/test_deploy.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (it talks to a `CustomObjectsApi`-shaped object).
- Produces (in `cifar_mlops.deploy`):
  - `to_s3_uri(artifact_uri: str) -> str`
  - `@dataclass(frozen=True) class ServingConfig(name: str, namespace: str, image: str, service_account: str, model_dir: str = "/mnt/models")`
  - `build_inference_service(config: ServingConfig, storage_uri: str) -> dict`
  - `is_ready(isvc: dict) -> bool`
  - `apply_inference_service(api, manifest: dict) -> str` (`"created"` or `"updated"`)
  - `wait_until_ready(api, name: str, namespace: str, timeout_s: float, poll_s: float = 5.0, clock=time.monotonic, sleep=time.sleep) -> dict`

- [ ] **Step 1: Write the failing tests `tests/test_deploy.py`**

```python
import pytest
from kubernetes.client.exceptions import ApiException

from cifar_mlops.deploy import (
    ServingConfig,
    apply_inference_service,
    build_inference_service,
    is_ready,
    to_s3_uri,
    wait_until_ready,
)

CONFIG = ServingConfig(name="cifar10", namespace="serving", image="cifar-mlops:dev", service_account="model-puller")
READY = {"conditions": [{"type": "Ready", "status": "True"}]}
NOT_READY = {"conditions": [{"type": "Ready", "status": "False", "reason": "Pending"}]}


class FakeCustomObjectsApi:
    """In-memory stand-in for kubernetes.client.CustomObjectsApi."""

    def __init__(self, existing=None, statuses=(), get_error=None):
        self.objects = {} if existing is None else {("serving", "cifar10"): existing}
        self.statuses = list(statuses)
        self.get_error = get_error
        self.calls = []

    def get_namespaced_custom_object(self, group, version, namespace, plural, name):
        if self.get_error is not None:
            raise self.get_error
        if (namespace, name) not in self.objects:
            raise ApiException(status=404, reason="Not Found")
        obj = self.objects[(namespace, name)]
        return {**obj, "status": self.statuses.pop(0)} if self.statuses else obj

    def create_namespaced_custom_object(self, group, version, namespace, plural, body):
        self.calls.append(("create", body))
        self.objects[(namespace, body["metadata"]["name"])] = body

    def replace_namespaced_custom_object(self, group, version, namespace, plural, name, body):
        self.calls.append(("replace", body))
        self.objects[(namespace, name)] = body


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_to_s3_uri_converts_kfp_minio_uris():
    assert to_s3_uri("minio://mlpipeline/v2/artifacts/run/model") == "s3://mlpipeline/v2/artifacts/run/model"


def test_to_s3_uri_keeps_s3_uris():
    assert to_s3_uri("s3://bucket/model") == "s3://bucket/model"


@pytest.mark.parametrize("uri", ["gs://bucket/model", "minio://", "/local/path", "minio:///abs"])
def test_to_s3_uri_rejects_unsupported_uris(uri):
    with pytest.raises(ValueError, match="artifact URI"):
        to_s3_uri(uri)


def test_build_inference_service_uses_raw_mode_custom_predictor():
    manifest = build_inference_service(CONFIG, "s3://mlpipeline/model")

    assert manifest["metadata"]["annotations"]["serving.kserve.io/deploymentMode"] == "RawDeployment"
    predictor = manifest["spec"]["predictor"]
    container = predictor["containers"][0]
    assert predictor["serviceAccountName"] == "model-puller"
    assert container["image"] == "cifar-mlops:dev"
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert {"name": "STORAGE_URI", "value": "s3://mlpipeline/model"} in container["env"]
    assert container["command"][:3] == ["python", "-m", "cifar_mlops.predictor"]
    assert "nvidia.com/gpu" not in str(container["resources"])


def test_apply_creates_when_service_does_not_exist():
    api = FakeCustomObjectsApi()

    assert apply_inference_service(api, build_inference_service(CONFIG, "s3://b/m1")) == "created"
    assert api.calls[0][0] == "create"


def test_apply_replaces_existing_service_keeping_resource_version():
    existing = {"metadata": {"name": "cifar10", "namespace": "serving", "resourceVersion": "41"}}
    api = FakeCustomObjectsApi(existing=existing)

    result = apply_inference_service(api, build_inference_service(CONFIG, "s3://b/m2"))

    action, body = api.calls[0]
    assert (result, action) == ("updated", "replace")
    assert body["metadata"]["resourceVersion"] == "41"


def test_apply_propagates_errors_other_than_not_found():
    api = FakeCustomObjectsApi(get_error=ApiException(status=500, reason="boom"))

    with pytest.raises(ApiException):
        apply_inference_service(api, build_inference_service(CONFIG, "s3://b/m"))


def test_is_ready_requires_ready_condition_true():
    assert is_ready({"status": READY}) is True
    assert is_ready({"status": NOT_READY}) is False
    assert is_ready({}) is False


def test_is_ready_ignores_status_from_an_older_generation():
    stale = {"metadata": {"generation": 3}, "status": {**READY, "observedGeneration": 2}}

    assert is_ready(stale) is False


def test_wait_until_ready_polls_until_ready():
    api = FakeCustomObjectsApi(existing={"metadata": {"name": "cifar10"}}, statuses=[NOT_READY, NOT_READY, READY])
    clock = FakeClock()

    result = wait_until_ready(api, "cifar10", "serving", timeout_s=60, poll_s=5, clock=clock, sleep=clock.sleep)

    assert is_ready(result)
    assert clock.now == 10


def test_wait_until_ready_times_out_with_actionable_message():
    api = FakeCustomObjectsApi(existing={"metadata": {"name": "cifar10"}}, statuses=[NOT_READY] * 100)
    clock = FakeClock()

    with pytest.raises(TimeoutError, match="kubectl"):
        wait_until_ready(api, "cifar10", "serving", timeout_s=20, poll_s=5, clock=clock, sleep=clock.sleep)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_deploy.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cifar_mlops.deploy'`.

- [ ] **Step 3: Implement `src/cifar_mlops/deploy.py`**

```python
"""Create or update the KServe InferenceService that serves a trained model."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from kubernetes.client.exceptions import ApiException

GROUP = "serving.kserve.io"
VERSION = "v1beta1"
PLURAL = "inferenceservices"
NOT_FOUND = 404
SUPPORTED_URI_PREFIXES = ("minio://", "s3://")


def to_s3_uri(artifact_uri: str) -> str:
    """KFP stores artifacts as minio://bucket/key; KServe reads the same object as s3://bucket/key."""
    for prefix in SUPPORTED_URI_PREFIXES:
        if artifact_uri.startswith(prefix):
            bucket_and_key = artifact_uri[len(prefix):]
            if bucket_and_key and not bucket_and_key.startswith("/"):
                return f"s3://{bucket_and_key}"
    raise ValueError(f"Unsupported artifact URI {artifact_uri!r}; expected minio://bucket/key or s3://bucket/key")


@dataclass(frozen=True)
class ServingConfig:
    name: str
    namespace: str
    image: str
    service_account: str
    model_dir: str = "/mnt/models"


def build_inference_service(config: ServingConfig, storage_uri: str) -> dict:
    return {
        "apiVersion": f"{GROUP}/{VERSION}",
        "kind": "InferenceService",
        "metadata": {
            "name": config.name,
            "namespace": config.namespace,
            "annotations": {"serving.kserve.io/deploymentMode": "RawDeployment"},
        },
        "spec": {
            "predictor": {
                "serviceAccountName": config.service_account,
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": config.image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": [
                            "python", "-m", "cifar_mlops.predictor",
                            "--model_name", config.name,
                            "--model_dir", config.model_dir,
                        ],
                        # KServe's storage initializer downloads STORAGE_URI into /mnt/models.
                        "env": [{"name": "STORAGE_URI", "value": storage_uri}],
                        "resources": {
                            "requests": {"cpu": "500m", "memory": "1Gi"},
                            "limits": {"cpu": "2", "memory": "2Gi"},
                        },
                    }
                ],
            }
        },
    }


def is_ready(isvc: dict) -> bool:
    status = isvc.get("status") or {}
    generation = (isvc.get("metadata") or {}).get("generation")
    observed = status.get("observedGeneration")
    if generation is not None and observed is not None and observed < generation:
        return False
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in status.get("conditions", [])
    )


def apply_inference_service(api, manifest: dict) -> str:
    """Create the InferenceService, or replace it in place if it already exists."""
    name, namespace = manifest["metadata"]["name"], manifest["metadata"]["namespace"]
    try:
        existing = api.get_namespaced_custom_object(GROUP, VERSION, namespace, PLURAL, name)
    except ApiException as err:
        if err.status != NOT_FOUND:
            raise
        api.create_namespaced_custom_object(GROUP, VERSION, namespace, PLURAL, manifest)
        return "created"
    updated = {
        **manifest,
        "metadata": {**manifest["metadata"], "resourceVersion": existing["metadata"]["resourceVersion"]},
    }
    api.replace_namespaced_custom_object(GROUP, VERSION, namespace, PLURAL, name, updated)
    return "updated"


def wait_until_ready(
    api,
    name: str,
    namespace: str,
    timeout_s: float,
    poll_s: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    deadline = clock() + timeout_s
    while True:
        isvc = api.get_namespaced_custom_object(GROUP, VERSION, namespace, PLURAL, name)
        if is_ready(isvc):
            return isvc
        if clock() >= deadline:
            conditions = (isvc.get("status") or {}).get("conditions", [])
            raise TimeoutError(
                f"InferenceService {namespace}/{name} not Ready after {timeout_s:.0f}s; conditions={conditions}. "
                f"Inspect with: kubectl --context kind-kubeflow-lab -n {namespace} describe isvc {name}"
            )
        sleep(poll_s)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `make test`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cifar_mlops/deploy.py tests/test_deploy.py
git commit -m "feat: add InferenceService create-or-update and readiness wait"
```

---

### Task 10: Container image and in-cluster GPU check

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `infra/torch-gpu-check.yaml`
- Modify: `Makefile` (add `image`, `gpu-check`)

**Interfaces:**
- Consumes: the package from Tasks 4–9; `PYTORCH_BASE_IMAGE` from `infra/versions.env`.
- Produces: image `cifar-mlops:dev` loaded into kind, containing `cifar_mlops`, `kfp` (same version as local), `kserve`, `kubernetes`, and ResNet-18 ImageNet weights cached at `TORCH_HOME=/opt/torch`. Make targets `image` and `gpu-check`.

- [ ] **Step 1: Write `.dockerignore`**

```
.venv
.git
.pytest_cache
**/__pycache__
docs
tests
*.egg-info
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
# One image for every pipeline step and for the KServe predictor.
ARG PYTORCH_BASE_IMAGE
FROM ${PYTORCH_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/torch

WORKDIR /app

# Dependencies first so code changes don't reinstall them.
COPY pyproject.toml ./
RUN mkdir -p src/cifar_mlops && touch src/cifar_mlops/__init__.py \
    && pip install --no-cache-dir . \
    && python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')" \
    && chmod -R a+rX /opt/torch

COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .
```

- [ ] **Step 3: Write `infra/torch-gpu-check.yaml`**

```yaml
# Proves PyTorch inside cifar-mlops:dev can use the GPU.
apiVersion: v1
kind: Pod
metadata:
  name: torch-gpu-check
spec:
  restartPolicy: Never
  containers:
    - name: torch
      image: cifar-mlops:dev
      imagePullPolicy: IfNotPresent
      command:
        - python
        - -c
        - "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"
      resources:
        limits:
          nvidia.com/gpu: 1
```

- [ ] **Step 4: Add `image` and `gpu-check` targets to `Makefile`**

Add below `PY :=`:

```makefile
IMAGE := cifar-mlops:dev
```

Change `.PHONY` to `.PHONY: setup teardown ui venv test image gpu-check` and append:

```makefile
image:
	docker build --build-arg PYTORCH_BASE_IMAGE=$(PYTORCH_BASE_IMAGE) -t $(IMAGE) .
	kind load docker-image $(IMAGE) --name $(CLUSTER)

gpu-check:
	kubectl --context $(CONTEXT) delete pod torch-gpu-check --ignore-not-found
	kubectl --context $(CONTEXT) apply -f infra/torch-gpu-check.yaml
	kubectl --context $(CONTEXT) wait --for=jsonpath='{.status.phase}'=Succeeded pod/torch-gpu-check --timeout=300s
	kubectl --context $(CONTEXT) logs torch-gpu-check
	kubectl --context $(CONTEXT) delete pod torch-gpu-check
```

- [ ] **Step 5: Build, load and check the image**

Run: `make image`
Expected: the build succeeds and `kind load` prints `Image: "cifar-mlops:dev" ... loading...`. The first build downloads several GB.

Run: `docker run --rm cifar-mlops:dev python -c "import kfp, kserve, cifar_mlops.predictor; print(kfp.__version__)"`
Expected: prints the same kfp version as `pyproject.toml`. If pip reports a dependency conflict between `kfp` and `kserve` during the build, stop and report the exact conflict to the user rather than loosening pins silently.

- [ ] **Step 6: Verify PyTorch sees the GPU inside the cluster**

Run: `make gpu-check`
Expected: `CUDA OK: NVIDIA GeForce RTX 4060 Laptop GPU`.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile .dockerignore infra/torch-gpu-check.yaml Makefile
git commit -m "feat: build cifar-mlops image and verify CUDA inside the cluster"
```

---

### Task 11: KFP pipeline and submit CLI

**Files:**
- Create: `pipelines/__init__.py` (empty)
- Create: `pipelines/cifar10_pipeline.py`
- Create: `pipelines/submit.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_submit.py`
- Modify: `Makefile` (add `run`)

**Interfaces:**
- Consumes: everything in `cifar_mlops` (inside component bodies, at runtime in the image); `validate_threshold` (Task 6) in `submit.py`; `IMAGE` tag `cifar-mlops:dev` (Task 10); namespace `serving` and SA `model-puller` (Task 3).
- Produces:
  - `pipelines.cifar10_pipeline`: constants `IMAGE`, `SERVING_NAMESPACE`, `SERVING_ACCOUNT`, `DEPLOY_TIMEOUT_S = 300`; components `prepare_data_op`, `train_op`, `evaluate_op` (outputs `accuracy: float`, `approved: bool`), `deploy_op`; pipeline `cifar10_pipeline(epochs: int = 3, learning_rate: float = 1e-3, batch_size: int = 128, accuracy_threshold: float = 0.85)`.
  - `pipelines.submit`: `parse_args(argv) -> argparse.Namespace` (with `epochs`, `learning_rate`, `batch_size`, `threshold`, `host`, `experiment`), `build_arguments(args) -> dict`, `main(argv=None)`.
  - Make target `run` with variables `EPOCHS`, `LR`, `BATCH_SIZE`, `THRESHOLD`.

- [ ] **Step 1: Write the failing tests `tests/test_pipeline.py`**

```python
import yaml
from kfp import compiler

from pipelines.cifar10_pipeline import cifar10_pipeline


def _compile(tmp_path) -> tuple[str, dict]:
    output = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(cifar10_pipeline, str(output))
    text = output.read_text()
    return text, yaml.safe_load(text)


def test_pipeline_compiles_with_gpu_training_and_gated_deploy(tmp_path):
    text, spec = _compile(tmp_path)

    assert "nvidia.com/gpu" in text
    assert "comp-deploy-op" in spec["components"]
    assert "comp-condition-1" in spec["components"]
    assert "deploy-op" not in spec["root"]["dag"]["tasks"]


def test_pipeline_parameter_defaults_match_spec(tmp_path):
    _, spec = _compile(tmp_path)
    parameters = spec["root"]["inputDefinitions"]["parameters"]

    defaults = {name: parameters[name]["defaultValue"] for name in parameters}
    assert defaults == {"epochs": 3, "learning_rate": 1e-3, "batch_size": 128, "accuracy_threshold": 0.85}
```

- [ ] **Step 2: Write the failing tests `tests/test_submit.py`**

```python
import pytest

from pipelines.submit import build_arguments, parse_args


def test_parse_args_defaults_match_pipeline_defaults():
    args = parse_args([])

    assert (args.epochs, args.learning_rate, args.batch_size, args.threshold) == (3, 1e-3, 128, 0.85)


@pytest.mark.parametrize(
    "argv",
    [
        ["--epochs", "-1"],
        ["--learning-rate", "0"],
        ["--batch-size", "0"],
        ["--threshold", "1.5"],
        ["--threshold", "-0.1"],
        ["--threshold", "nan"],
    ],
)
def test_parse_args_rejects_invalid_values(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_build_arguments_maps_cli_flags_to_pipeline_parameters():
    args = parse_args(["--epochs", "0", "--threshold", "0.99"])

    assert build_arguments(args) == {
        "epochs": 0,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "accuracy_threshold": 0.99,
    }
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py tests/test_submit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipelines'`.

- [ ] **Step 4: Implement `pipelines/cifar10_pipeline.py`**

```python
"""Kubeflow pipeline: prepare CIFAR-10 -> train on GPU -> evaluate -> deploy if accurate enough.

Component bodies run inside the cifar-mlops:dev image, so all imports happen inside the functions.
"""
from typing import NamedTuple

from kfp import dsl

IMAGE = "cifar-mlops:dev"
SERVING_NAMESPACE = "serving"
SERVING_ACCOUNT = "model-puller"
DEPLOY_TIMEOUT_S = 300


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def prepare_data_op(dataset: dsl.Output[dsl.Dataset], val_fraction: float = 0.1, seed: int = 42):
    from pathlib import Path

    from cifar_mlops.data import prepare_cifar10

    sizes = prepare_cifar10(Path(dataset.path), val_fraction=val_fraction, seed=seed)
    dataset.metadata.update(sizes)
    print(f"Prepared CIFAR-10 splits: {sizes}")


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def train_op(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
    epochs: int,
    learning_rate: float,
    batch_size: int,
):
    import json
    from pathlib import Path

    from cifar_mlops.data import LABELS_FILE, load_split
    from cifar_mlops.model import export_model
    from cifar_mlops.train import TrainConfig, resolve_device, train_model

    data_dir = Path(dataset.path)
    device = resolve_device(require_gpu=True)
    config = TrainConfig(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    def report(result):
        print(f"epoch {result.epoch}: train_loss={result.train_loss:.4f} val_accuracy={result.val_accuracy:.4f}")
        metrics.log_metric(f"epoch_{result.epoch}_train_loss", result.train_loss)
        metrics.log_metric(f"epoch_{result.epoch}_val_accuracy", result.val_accuracy)

    trained, history = train_model(
        load_split(data_dir / "train.pt"), load_split(data_dir / "val.pt"), config, device, on_epoch_end=report
    )
    labels = json.loads((data_dir / LABELS_FILE).read_text())
    export_model(trained, labels, Path(model.path))
    model.metadata.update({"framework": "pytorch-torchscript", "epochs": epochs})
```

Append to the same file:

```python
@dsl.component(base_image=IMAGE, install_kfp_package=False)
def evaluate_op(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Input[dsl.Model],
    confusion: dsl.Output[dsl.ClassificationMetrics],
    accuracy_threshold: float,
) -> NamedTuple("EvaluationOutputs", [("accuracy", float), ("approved", bool)]):
    from collections import namedtuple
    from pathlib import Path

    from cifar_mlops.data import load_split
    from cifar_mlops.evaluate import accuracy, confusion_matrix, passes_gate, predict_labels
    from cifar_mlops.model import load_exported
    from cifar_mlops.train import resolve_device

    eval_batch_size = 256
    device = resolve_device(require_gpu=False)
    loaded, labels = load_exported(Path(model.path))
    test = load_split(Path(dataset.path) / "test.pt")
    predictions = predict_labels(loaded.to(device), test, eval_batch_size, device)
    test_accuracy = accuracy(test.labels, predictions)
    confusion.log_confusion_matrix(labels, confusion_matrix(test.labels, predictions, len(labels)))
    approved = passes_gate(test_accuracy, accuracy_threshold)
    print(f"test accuracy={test_accuracy:.4f} threshold={accuracy_threshold} approved={approved}")
    outputs = namedtuple("EvaluationOutputs", ["accuracy", "approved"])
    return outputs(test_accuracy, approved)


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def deploy_op(
    model: dsl.Input[dsl.Model],
    image: str,
    namespace: str,
    service_account: str,
    timeout_s: int,
) -> str:
    from kubernetes import client, config

    from cifar_mlops.deploy import (
        ServingConfig,
        apply_inference_service,
        build_inference_service,
        to_s3_uri,
        wait_until_ready,
    )

    config.load_incluster_config()
    api = client.CustomObjectsApi()
    serving = ServingConfig(name="cifar10", namespace=namespace, image=image, service_account=service_account)
    storage_uri = to_s3_uri(model.uri)
    action = apply_inference_service(api, build_inference_service(serving, storage_uri))
    print(f"InferenceService {namespace}/{serving.name} {action} with {storage_uri}; waiting for Ready")
    wait_until_ready(api, serving.name, namespace, timeout_s=timeout_s)
    print("InferenceService is Ready")
    return f"{namespace}/{serving.name}"


@dsl.pipeline(name="cifar10-gpu-mlops", description="Train ResNet-18 on CIFAR-10 on GPU and deploy it if accurate enough")
def cifar10_pipeline(
    epochs: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    accuracy_threshold: float = 0.85,
):
    data = prepare_data_op()

    train = train_op(
        dataset=data.outputs["dataset"], epochs=epochs, learning_rate=learning_rate, batch_size=batch_size
    )
    train.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    train.set_cpu_limit("4").set_memory_request("3G").set_memory_limit("6G")
    train.set_caching_options(False)

    evaluation = evaluate_op(
        dataset=data.outputs["dataset"], model=train.outputs["model"], accuracy_threshold=accuracy_threshold
    )
    evaluation.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    evaluation.set_memory_limit("4G")
    evaluation.set_caching_options(False)

    with dsl.If(evaluation.outputs["approved"] == True, name="accuracy-gate"):  # noqa: E712 (KFP condition syntax)
        deploy = deploy_op(
            model=train.outputs["model"],
            image=IMAGE,
            namespace=SERVING_NAMESPACE,
            service_account=SERVING_ACCOUNT,
            timeout_s=DEPLOY_TIMEOUT_S,
        )
        deploy.set_caching_options(False)
```

Component bodies are shipped to the pod as standalone source, so they cannot see module-level constants. That's why `evaluate_op` defines `eval_batch_size` locally, and why `IMAGE`, the namespace and the timeout are passed into `deploy_op` as arguments.

- [ ] **Step 5: Implement `pipelines/submit.py`**

```python
"""Submit a cifar10-gpu-mlops run to Kubeflow Pipelines (requires `make ui` port-forward)."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from cifar_mlops.evaluate import validate_threshold
from pipelines.cifar10_pipeline import cifar10_pipeline

DEFAULT_HOST = "http://localhost:8080"
DEFAULT_EXPERIMENT = "cifar10"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.85, help="minimum test accuracy to deploy (0-1)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    args = parser.parse_args(argv)
    if args.epochs < 0:
        parser.error("--epochs must be >= 0")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    try:
        validate_threshold(args.threshold)
    except ValueError as err:
        parser.error(str(err))
    return args


def build_arguments(args: argparse.Namespace) -> dict:
    return {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "accuracy_threshold": args.threshold,
    }


def main(argv: Sequence[str] | None = None) -> None:
    import kfp

    args = parse_args(argv)
    client = kfp.Client(host=args.host)
    run = client.create_run_from_pipeline_func(
        cifar10_pipeline, arguments=build_arguments(args), experiment_name=args.experiment
    )
    print(f"Submitted run {run.run_id}")
    print(f"Watch it at {args.host}/#/runs/details/{run.run_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add the `run` target to `Makefile`**

Add below `IMAGE :=`:

```makefile
EPOCHS ?= 3
LR ?= 0.001
BATCH_SIZE ?= 128
THRESHOLD ?= 0.85
```

Change `.PHONY` to `.PHONY: setup teardown ui venv test image gpu-check run` and append:

```makefile
run:
	$(PY) -m pipelines.submit --epochs $(EPOCHS) --learning-rate $(LR) --batch-size $(BATCH_SIZE) --threshold $(THRESHOLD)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `make test`
Expected: all PASS. If `comp-condition-1` is named differently in this kfp version, print `list(spec["components"])` and adjust the assertion to the condition component's actual name. The intent is that `deploy-op` sits under a condition and not in the root DAG.

- [ ] **Step 8: Run the pipeline end to end on the cluster**

With `make ui` running in another terminal, run: `make run`
Expected: `Submitted run <id>` plus a URL. Ask the user to open it and watch the run. Checks:
- `train-op` logs show 3 lines like `epoch N: train_loss=... val_accuracy=...` and no CUDA error.
- `evaluate-op` logs show `approved=True` with accuracy ≥ 0.85.
- `deploy-op` logs end with `InferenceService is Ready`.
- `kubectl --context kind-kubeflow-lab -n serving get isvc cifar10` shows `READY True`.

If `deploy-op` times out, check `kubectl --context kind-kubeflow-lab -n serving describe isvc cifar10` and the storage-initializer container logs of the `cifar10-predictor` pod (object-store credentials and `storageUri` are the usual causes).

- [ ] **Step 9: Commit**

```bash
git add pipelines/ tests/test_pipeline.py tests/test_submit.py Makefile
git commit -m "feat: add KFP pipeline with GPU training and accuracy-gated deploy"
```

---

### Task 12: Prediction client, end-to-end gate demo, project README

**Files:**
- Create: `scripts/predict.py`
- Modify: `Makefile` (add `predict`)
- Create: `README.md`

**Interfaces:**
- Consumes: `encode_png_b64` (Task 8), `MAX_INSTANCES` (Task 8), the deployed service `serving/cifar10-predictor` (Task 11).
- Produces: `make predict [COUNT=8]`; the final project documentation.

- [ ] **Step 1: Implement `scripts/predict.py`**

```python
"""Send random CIFAR-10 test images to the deployed model and compare predictions with the true labels."""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

from cifar_mlops.data import CIFAR10_CLASSES
from cifar_mlops.imaging import encode_png_b64
from cifar_mlops.predictor import MAX_INSTANCES

DATA_CACHE = Path.home() / ".cache" / "cifar10"
REQUEST_TIMEOUT_S = 30


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as err:
        sys.exit(f"Server returned HTTP {err.code}: {err.read().decode(errors='replace')}")
    except urllib.error.URLError as err:
        sys.exit(f"Could not reach {url}: {err.reason}. Is the port-forward running?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8081")
    parser.add_argument("--model", default="cifar10")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not 1 <= args.count <= MAX_INSTANCES:
        parser.error(f"--count must be between 1 and {MAX_INSTANCES}")

    from torchvision.datasets import CIFAR10

    test_set = CIFAR10(str(DATA_CACHE), train=False, download=True)
    indices = random.Random(args.seed).sample(range(len(test_set.data)), args.count)
    payload = {"instances": [{"image_b64": encode_png_b64(test_set.data[i])} for i in indices]}

    result = post_json(f"{args.url}/v1/models/{args.model}:predict", payload)

    correct = 0
    print(f"{'actual':<12}{'predicted':<12}confidence")
    for index, prediction in zip(indices, result["predictions"]):
        actual = CIFAR10_CLASSES[test_set.targets[index]]
        correct += actual == prediction["label"]
        mark = "✓" if actual == prediction["label"] else "✗"
        print(f"{actual:<12}{prediction['label']:<12}{prediction['confidence']:.2f} {mark}")
    print(f"\n{correct}/{args.count} correct")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the `predict` target to `Makefile`**

Add below `THRESHOLD ?=`:

```makefile
COUNT ?= 8
PREDICT_PORT := 8081
```

Change `.PHONY` to `.PHONY: setup teardown ui venv test image gpu-check run predict` and append:

```makefile
predict:
	@kubectl --context $(CONTEXT) -n serving port-forward svc/cifar10-predictor $(PREDICT_PORT):80 >/dev/null 2>&1 & \
	PF_PID=$$!; trap "kill $$PF_PID" EXIT; sleep 3; \
	$(PY) scripts/predict.py --url http://localhost:$(PREDICT_PORT) --count $(COUNT)
```

- [ ] **Step 3: Verify predictions against the deployed model**

Run: `kubectl --context kind-kubeflow-lab -n serving get svc`
Expected: a service named `cifar10-predictor`. If the name differs, update `svc/cifar10-predictor` in the Makefile to match.

Run: `make predict`
Expected: an `actual / predicted / confidence` table and something like `7/8 correct` (most rows ✓).

Run this malformed-request check:

```bash
kubectl --context kind-kubeflow-lab -n serving port-forward svc/cifar10-predictor 8081:80 >/dev/null 2>&1 & PF=$!; sleep 3
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8081/v1/models/cifar10:predict -H 'Content-Type: application/json' -d '{"instances": []}'
kill $PF
```

Expected: `400`.

- [ ] **Step 4: Demonstrate the accuracy gate blocking a bad model**

Run: `make run THRESHOLD=0.99 EPOCHS=1`
Expected: in the UI the run completes, `evaluate-op` logs `approved=False`, and `deploy-op` is shown as skipped/not executed. Then `make predict` still answers, because the earlier model keeps serving.

Run: `make run EPOCHS=0`
Expected: `train-op` completes with no epochs logged, accuracy is around 0.1, and deploy is skipped.

- [ ] **Step 5: Write `README.md`**

````markdown
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
  `kubectl edit isvc cifar10 -n serving`.

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
````

- [ ] **Step 6: Run the full unit-test suite one last time**

Run: `make test`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/predict.py Makefile README.md
git commit -m "feat: add prediction client and project README"
```
