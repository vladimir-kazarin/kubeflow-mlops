#!/usr/bin/env bash
# Step 1 of infra/README.md as one script: lets Docker (and therefore kind) use the NVIDIA GPU.
# Run once with: sudo bash infra/install-host-gpu-runtime.sh
set -euo pipefail

[[ "${EUID}" -eq 0 ]] || { echo "Run with sudo: sudo bash $0" >&2; exit 1; }

KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
SOURCES=/etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "==> Installing NVIDIA Container Toolkit"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o "${KEYRING}"
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=${KEYRING}] https://#g" > "${SOURCES}"
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit

echo "==> Making nvidia the default Docker runtime and enabling GPU requests via volume mounts"
nvidia-ctk runtime configure --runtime=docker --set-as-default
nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place

# Restarting Docker stops kind node containers; remember which were running and start them again.
running_kind_nodes="$(docker ps --filter label=io.x-k8s.kind.cluster --format '{{.Names}}')"
echo "==> Restarting Docker"
systemctl restart docker
for node in ${running_kind_nodes}; do
  echo "==> Restarting kind node ${node}"
  docker start "${node}" >/dev/null
done

echo "==> Raising inotify limits (kind's documented fix; the default 128 instances runs out with 2+ clusters)"
cat > /etc/sysctl.d/99-kind-inotify.conf <<'SYSCTL'
fs.inotify.max_user_instances = 512
fs.inotify.max_user_watches = 524288
SYSCTL
sysctl --quiet --load /etc/sysctl.d/99-kind-inotify.conf

echo "==> Checks"
echo "Default runtime: $(docker info --format '{{.DefaultRuntime}}')"
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L
echo "Done. Tell Claude to continue."
