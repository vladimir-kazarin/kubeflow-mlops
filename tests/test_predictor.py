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
