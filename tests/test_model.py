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
