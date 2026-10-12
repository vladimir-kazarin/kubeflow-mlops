"""Kubeflow pipeline: prepare CIFAR-10 -> train on GPU -> evaluate -> deploy if accurate enough.

Component bodies run inside the cifar-mlops:dev image as standalone source, so all imports
happen inside the functions and module-level constants are passed in as arguments.
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
