#!/usr/bin/env bash
# Creates the kubeflow-lab kind cluster with GPU support, Kubeflow Pipelines and KServe.
# Safe to re-run: each step skips or re-applies work that is already done.
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
KSERVE_RELEASE_URL="https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}"

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

configure_kserve_standard_mode() {
  # "Standard" is KServe's current name for RawDeployment: plain Deployments + Services,
  # no Knative/Istio. No ingress controller exists in this cluster, so skip Ingress objects.
  "${KCTL[@]}" -n kserve get configmap inferenceservice-config -o json | python3 -c '
import json, sys
cm = json.load(sys.stdin)
deploy = json.loads(cm["data"]["deploy"])
ingress = json.loads(cm["data"]["ingress"])
cm["data"]["deploy"] = json.dumps({**deploy, "defaultDeploymentMode": "Standard"}, indent=2)
cm["data"]["ingress"] = json.dumps({**ingress, "disableIngressCreation": True}, indent=2)
for key in ("resourceVersion", "uid", "creationTimestamp", "managedFields"):
    cm["metadata"].pop(key, None)
json.dump(cm, sys.stdout)' | "${KCTL[@]}" apply --server-side --force-conflicts -f -
  "${KCTL[@]}" -n kserve rollout restart deployment/kserve-controller-manager
  "${KCTL[@]}" -n kserve rollout status deployment/kserve-controller-manager --timeout=300s
}

install_kserve() {
  log "Installing KServe ${KSERVE_VERSION} (Standard/RawDeployment mode)"
  # kserve.yaml expects the namespace to exist already (KServe's own installer creates it first).
  "${KCTL[@]}" create namespace kserve --dry-run=client -o yaml | "${KCTL[@]}" apply -f -
  "${KCTL[@]}" apply --server-side --force-conflicts -f "${KSERVE_RELEASE_URL}/kserve.yaml"
  "${KCTL[@]}" -n kserve wait --for=condition=Available deployment/kserve-controller-manager --timeout=600s
  # Only the ClusterServingRuntimes are needed (the sample sklearn model uses one). The file's
  # LLMInferenceServiceConfig objects are rejected by v0.21.0's own webhook, so they are skipped.
  local runtimes attempt
  runtimes="$(curl -fsSL "${KSERVE_RELEASE_URL}/kserve-cluster-resources.yaml" | python3 -c '
import sys, yaml
docs = [d for d in yaml.safe_load_all(sys.stdin) if d and d.get("kind") == "ClusterServingRuntime"]
yaml.safe_dump_all(docs, sys.stdout)')"
  [[ -n "${runtimes}" ]] || die "No ClusterServingRuntimes found in KServe ${KSERVE_VERSION} cluster resources"
  # The KServe webhook may need a moment after startup before it accepts requests.
  for attempt in 1 2 3 4 5; do
    "${KCTL[@]}" apply --server-side --force-conflicts -f - <<<"${runtimes}" && break
    [[ "${attempt}" == 5 ]] && die "Could not apply KServe cluster resources"
    sleep 10
  done
  configure_kserve_standard_mode
}

configure_serving_namespace() {
  log "Configuring namespace ${SERVING_NAMESPACE}"
  local access_key secret_key
  access_key="$("${KCTL[@]}" -n kubeflow get secret mlpipeline-minio-artifact -o jsonpath='{.data.accesskey}' | base64 -d)"
  secret_key="$("${KCTL[@]}" -n kubeflow get secret mlpipeline-minio-artifact -o jsonpath='{.data.secretkey}' | base64 -d)"
  [[ -n "${access_key}" && -n "${secret_key}" ]] || die "Could not read kubeflow/mlpipeline-minio-artifact credentials"
  # Ask KFP where its object store lives (MinIO in older releases, SeaweedFS since 2.15).
  local store_env store_host store_port
  store_env="$("${KCTL[@]}" -n kubeflow get deploy ml-pipeline \
    -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}={.value}{"\n"}{end}')"
  store_host="$(sed -n 's/^OBJECTSTORECONFIG_HOST=//p' <<<"${store_env}")"
  store_port="$(sed -n 's/^OBJECTSTORECONFIG_PORT=//p' <<<"${store_env}")"
  [[ -n "${store_host}" && -n "${store_port}" ]] \
    || die "Could not read OBJECTSTORECONFIG_HOST/PORT from kubeflow/ml-pipeline"
  OBJECT_STORE_ENDPOINT="${store_host}:${store_port}"
  log "KFP object store endpoint: ${OBJECT_STORE_ENDPOINT}"

  "${KCTL[@]}" apply -f "${SCRIPT_DIR}/serving.yaml"
  # Credentials go through stdin, never through command-line arguments. An env file keeps
  # each value literal, so keys that look like numbers, booleans or YAML syntax stay intact.
  "${KCTL[@]}" -n "${SERVING_NAMESPACE}" create secret generic kfp-object-store \
      --from-env-file=/dev/stdin --dry-run=client -o yaml <<EOF \
    | "${KCTL[@]}" annotate --local -f - -o yaml \
        "serving.kserve.io/s3-endpoint=${OBJECT_STORE_ENDPOINT}" \
        serving.kserve.io/s3-usehttps=0 \
        serving.kserve.io/s3-verifyssl=0 \
        serving.kserve.io/s3-region=us-east-1 \
    | "${KCTL[@]}" apply -f -
AWS_ACCESS_KEY_ID=${access_key}
AWS_SECRET_ACCESS_KEY=${secret_key}
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

main "$@"
