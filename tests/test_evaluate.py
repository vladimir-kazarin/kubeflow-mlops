import math

import pytest
import torch

from cifar_mlops.evaluate import (
    accuracy,
    confusion_matrix,
    evaluate_accuracy,
    passes_gate,
    predict_labels,
    validate_threshold,
)
from cifar_mlops.model import build_model

CPU = torch.device("cpu")


def test_accuracy_is_fraction_of_matches():
    assert accuracy(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 0, 3])) == pytest.approx(0.75)


def test_accuracy_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        accuracy(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))


def test_accuracy_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        accuracy(torch.tensor([0, 1]), torch.tensor([0]))


def test_confusion_matrix_rows_are_true_labels():
    matrix = confusion_matrix(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1]), num_classes=2)

    assert matrix == [[1, 1], [0, 1]]


@pytest.mark.parametrize("correct, threshold", [(9000, 0.9), (8800, 0.88), (7000, 0.7), (9500, 0.95)])
def test_accuracy_exactly_at_threshold_passes_gate_with_real_tensors(correct, threshold):
    total = 10_000
    y_true = torch.zeros(total, dtype=torch.int64)
    y_pred = torch.cat([torch.zeros(correct, dtype=torch.int64), torch.ones(total - correct, dtype=torch.int64)])

    assert passes_gate(accuracy(y_true, y_pred), threshold) is True


def test_passes_gate_is_inclusive_at_the_threshold():
    assert passes_gate(0.85, 0.85) is True
    assert passes_gate(0.8499, 0.85) is False


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan])
def test_validate_threshold_rejects_values_outside_unit_interval(threshold):
    with pytest.raises(ValueError, match="threshold"):
        validate_threshold(threshold)


def test_predict_labels_returns_one_label_per_image(tiny_split):
    model = build_model(pretrained=False)

    predictions = predict_labels(model, tiny_split, batch_size=8, device=CPU)

    assert predictions.shape == (len(tiny_split),)
    assert predictions.dtype == torch.int64
    assert int(predictions.min()) >= 0 and int(predictions.max()) <= 9


def test_evaluate_accuracy_is_between_zero_and_one(tiny_split):
    value = evaluate_accuracy(build_model(pretrained=False), tiny_split, batch_size=8, device=CPU)

    assert 0.0 <= value <= 1.0
