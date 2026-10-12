import pytest
import torch

from cifar_mlops.data import Split, iterate_batches, load_split, save_split, split_train_val
from tests.conftest import TINY_SPLIT_SIZE, image_ids


def test_split_rejects_float_images():
    with pytest.raises(ValueError, match="uint8"):
        Split(images=torch.zeros(2, 3, 32, 32), labels=torch.zeros(2, dtype=torch.int64))


def test_split_rejects_label_count_mismatch():
    with pytest.raises(ValueError, match="labels"):
        Split(images=torch.zeros(2, 3, 32, 32, dtype=torch.uint8), labels=torch.zeros(3, dtype=torch.int64))


def test_split_train_val_sizes_are_disjoint_and_complete(tiny_split):
    train, val = split_train_val(tiny_split, val_fraction=0.25, seed=0)

    assert (len(train), len(val)) == (15, 5)
    assert set(image_ids(train)).isdisjoint(image_ids(val))
    assert sorted(image_ids(train) + image_ids(val)) == list(range(TINY_SPLIT_SIZE))


def test_split_train_val_is_deterministic_for_a_seed(tiny_split):
    first = split_train_val(tiny_split, val_fraction=0.25, seed=7)
    second = split_train_val(tiny_split, val_fraction=0.25, seed=7)

    assert image_ids(first[1]) == image_ids(second[1])


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5, 0.01])
def test_split_train_val_rejects_fractions_that_leave_an_empty_split(tiny_split, fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        split_train_val(tiny_split, val_fraction=fraction, seed=0)


def test_save_and_load_split_round_trip(tiny_split, tmp_path):
    path = tmp_path / "train.pt"

    save_split(tiny_split, path)
    loaded = load_split(path)

    assert torch.equal(loaded.images, tiny_split.images)
    assert torch.equal(loaded.labels, tiny_split.labels)


def test_load_split_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        load_split(tmp_path / "missing.pt")


def test_iterate_batches_covers_every_item_once_with_partial_last_batch(tiny_split):
    batches = list(iterate_batches(tiny_split, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(1)))

    assert [len(labels) for _, labels in batches] == [8, 8, 4]
    seen = [i for images, _ in batches for i in images[:, 0, 0, 0].tolist()]
    assert sorted(seen) == list(range(TINY_SPLIT_SIZE))


def test_iterate_batches_without_shuffle_keeps_order(tiny_split):
    batches = list(iterate_batches(tiny_split, batch_size=64, shuffle=False))

    assert len(batches) == 1
    assert batches[0][0][:, 0, 0, 0].tolist() == list(range(TINY_SPLIT_SIZE))


def test_iterate_batches_rejects_non_positive_batch_size(tiny_split):
    with pytest.raises(ValueError, match="batch_size"):
        list(iterate_batches(tiny_split, batch_size=0, shuffle=False))
