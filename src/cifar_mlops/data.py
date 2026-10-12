"""CIFAR-10 download, train/val/test splits, and batching over in-memory uint8 tensors."""
from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
LABELS_FILE = "labels.json"


@dataclass(frozen=True)
class Split:
    """A set of images (uint8, N x 3 x H x W) with integer class labels (int64, N)."""

    images: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        if self.images.dtype != torch.uint8 or self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError(
                f"images must be uint8 with shape (N, 3, H, W); got {self.images.dtype} {tuple(self.images.shape)}"
            )
        if self.labels.ndim != 1 or self.labels.shape[0] != self.images.shape[0]:
            raise ValueError(
                f"labels must have shape ({self.images.shape[0]},); got {tuple(self.labels.shape)}"
            )

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def split_train_val(full: Split, val_fraction: float, seed: int) -> tuple[Split, Split]:
    """Randomly split into (train, val); the same seed always gives the same split."""
    n_total = len(full)
    n_val = round(n_total * val_fraction)
    if not 0.0 < val_fraction < 1.0 or n_val == 0 or n_val == n_total:
        raise ValueError(f"val_fraction={val_fraction} leaves an empty split for {n_total} items")
    order = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = order[:n_val], order[n_val:]
    return (
        Split(images=full.images[train_idx], labels=full.labels[train_idx]),
        Split(images=full.images[val_idx], labels=full.labels[val_idx]),
    )


def save_split(split: Split, path: Path) -> None:
    torch.save({"images": split.images, "labels": split.labels}, path)


def load_split(path: Path) -> Split:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    payload = torch.load(path, weights_only=True)
    return Split(images=payload["images"], labels=payload["labels"])


def iterate_batches(
    split: Split,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (images, labels) batches; the last batch may be smaller."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    n_total = len(split)
    order = torch.randperm(n_total, generator=generator) if shuffle else torch.arange(n_total)
    for start in range(0, n_total, batch_size):
        idx = order[start:start + batch_size]
        yield split.images[idx], split.labels[idx]


def _split_from_torchvision(dataset) -> Split:
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(dataset.targets, dtype=torch.int64)
    return Split(images=images, labels=labels)


def prepare_cifar10(output_dir: Path, val_fraction: float = 0.1, seed: int = 42) -> dict[str, int]:
    """Download CIFAR-10 and write train/val/test splits plus labels.json into output_dir."""
    from torchvision.datasets import CIFAR10

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as download_dir:
        train_full = _split_from_torchvision(CIFAR10(download_dir, train=True, download=True))
        test = _split_from_torchvision(CIFAR10(download_dir, train=False, download=True))
    train, val = split_train_val(train_full, val_fraction=val_fraction, seed=seed)
    splits = dict(zip(SPLIT_NAMES, (train, val, test)))
    for name, split in splits.items():
        save_split(split, output_dir / f"{name}.pt")
    (output_dir / LABELS_FILE).write_text(json.dumps(list(CIFAR10_CLASSES)))
    return {name: len(split) for name, split in splits.items()}
