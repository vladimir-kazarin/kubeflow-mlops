# Infrastructure

Everything runs in a dedicated kind cluster called `kubeflow-lab` (context `kind-kubeflow-lab`).
Your other clusters (e.g. `kind-asi-lab`) are never touched by these scripts.

## Step 1 — Give Docker access to the GPU (you run this, needs sudo)

Your Windows NVIDIA driver already exposes the GPU to WSL (`nvidia-smi` works).
Docker needs the NVIDIA Container Toolkit to pass it into containers.

**Quick way:** `sudo bash infra/install-host-gpu-runtime.sh` does everything below,
restarts any kind nodes that were running, and prints the checks.

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

## Step 2 — Create the cluster and install everything

```bash
make setup
```

Each stage prints `==> ...` and checks itself. The GPU stage ends with the
`nvidia-smi` table from inside a Kubernetes pod; the last stage deploys and
deletes a sample KServe model.

## What "done" looks like

```bash
kubectl --context kind-kubeflow-lab get pods -n kubeflow        # all Running
kubectl --context kind-kubeflow-lab get pods -n kserve          # kserve-controller-manager Running
make ui                                                          # then open http://localhost:8080
```

### Object store credentials
KFP stores every artifact (datasets, models, metrics) in its own S3-compatible
store (SeaweedFS at `seaweedfs.kubeflow:9000` in KFP 2.17; MinIO in older releases). `setup.sh` reads the
endpoint from the `ml-pipeline` deployment and copies the store's credentials into
`serving/kfp-object-store` so KServe can download trained models from there.

### KServe "Standard" mode
KServe calls the no-Knative/no-Istio mode "Standard" (older name: "RawDeployment").
Each InferenceService becomes a plain Deployment + Service. `setup.sh` makes it the
default and turns off Ingress creation (this cluster has no ingress controller);
you reach models with `kubectl port-forward` instead.

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

### Device plugin crashes with "too many open files"
The host ran out of inotify instances (default 128, shared by every kind cluster).
Re-run `sudo bash infra/install-host-gpu-runtime.sh`, or just:
`sudo sysctl -w fs.inotify.max_user_instances=512`, then
`kubectl --context kind-kubeflow-lab -n nvidia-device-plugin delete pod --all` and `make setup`.

### Pods stuck Pending
`kubectl --context kind-kubeflow-lab describe pod <name>`. If it says
`Insufficient memory`, raise WSL memory in `%UserProfile%\.wslconfig`
(`[wsl2]` / `memory=14GB`) and run `wsl --shutdown` from Windows.

### `deploy-op` times out
`kubectl --context kind-kubeflow-lab -n serving describe isvc cifar10` and the logs of the
`storage-initializer` init container of the `cifar10-predictor` pod. Wrong object-store
credentials or `storageUri` are the usual causes.

### Start over
`make teardown && make setup`
