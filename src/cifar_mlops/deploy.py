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
