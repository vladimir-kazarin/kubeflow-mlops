import pytest
import torch

from cifar_mlops.data import Split

TINY_SPLIT_SIZE = 20


@pytest.fixture
def tiny_split() -> Split:
    """20 random 32x32 RGB images; images[i, 0, 0, 0] == i so tests can track identity."""
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (TINY_SPLIT_SIZE, 3, 32, 32), dtype=torch.uint8, generator=generator)
    images[:, 0, 0, 0] = torch.arange(TINY_SPLIT_SIZE, dtype=torch.uint8)
    labels = torch.randint(0, 10, (TINY_SPLIT_SIZE,), dtype=torch.int64, generator=generator)
    return Split(images=images, labels=labels)


def image_ids(split: Split) -> list[int]:
    return split.images[:, 0, 0, 0].tolist()
