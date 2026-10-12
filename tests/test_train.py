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
