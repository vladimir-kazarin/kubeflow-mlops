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
