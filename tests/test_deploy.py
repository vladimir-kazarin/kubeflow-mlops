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
