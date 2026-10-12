"""Fine-tune the ResNet-18 ServingModel on CIFAR-10 with mixed precision on GPU."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from cifar_mlops.data import Split, iterate_batches
from cifar_mlops.evaluate import evaluate_accuracy
from cifar_mlops.model import ServingModel, build_model

GPU_MISSING_MESSAGE = (
    "GPU was requested but CUDA is not available in this container. "
    "Run the GPU smoke test in infra/README.md (make gpu-check)."
)
FLIP_PROBABILITY = 0.5


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    batch_size: int
    pretrained: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0; got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0; got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {self.batch_size}")


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    val_accuracy: float


def resolve_device(require_gpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError(GPU_MISSING_MESSAGE)
    return torch.device("cpu")


def random_flip(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Mirror each image horizontally with probability FLIP_PROBABILITY (simple augmentation)."""
    mask = torch.rand(images.shape[0], generator=generator) < FLIP_PROBABILITY
    return torch.where(mask.view(-1, 1, 1, 1), images.flip(-1), images)


def _train_one_epoch(
    model: ServingModel,
    train: Split,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    total_loss, total_seen = 0.0, 0
    for images, labels in iterate_batches(train, config.batch_size, shuffle=True, generator=generator):
        images = random_flip(images, generator).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.shape[0]
        total_seen += labels.shape[0]
    return total_loss / max(total_seen, 1)


def train_model(
    train: Split,
    val: Split,
    config: TrainConfig,
    device: torch.device,
    on_epoch_end: Callable[[EpochResult], None] | None = None,
) -> tuple[ServingModel, list[EpochResult]]:
    """Train for config.epochs epochs; epochs=0 returns the initial (untrained) model."""
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = build_model(pretrained=config.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[EpochResult] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(model, train, config, optimizer, scaler, device, generator)
        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_accuracy=evaluate_accuracy(model, val, config.batch_size, device),
        )
        history = [*history, result]
        if on_epoch_end is not None:
            on_epoch_end(result)
    return model, history
