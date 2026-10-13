"""KServe custom predictor: base64 PNG images in, CIFAR-10 labels with confidence out."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from kserve import Model, ModelServer, model_server
from kserve.errors import InvalidInput

from cifar_mlops.imaging import decode_image_b64
from cifar_mlops.model import load_exported

MAX_INSTANCES = 64
REQUEST_SHAPE_HINT = 'expected {"instances": [{"image_b64": "<base64 PNG>"}]}'


def parse_instances(payload) -> list[str]:
    """Validate the request body and return the base64 image strings."""
    if not isinstance(payload, dict) or "instances" not in payload:
        raise ValueError(f"Request body is missing 'instances'; {REQUEST_SHAPE_HINT}")
    instances = payload["instances"]
    if not isinstance(instances, list) or not instances:
        raise ValueError(f"'instances' must be a non-empty list; {REQUEST_SHAPE_HINT}")
    if len(instances) > MAX_INSTANCES:
        raise ValueError(f"At most {MAX_INSTANCES} instances per request; got {len(instances)}")
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict) or not isinstance(instance.get("image_b64"), str):
            raise ValueError(f"instances[{index}] must be an object with a string 'image_b64'")
    return [instance["image_b64"] for instance in instances]


def format_predictions(probabilities: torch.Tensor, labels: Sequence[str]) -> list[dict]:
    confidences, indices = probabilities.max(dim=1)
    return [
        {"label": labels[index], "confidence": round(confidence, 4)}
        for confidence, index in zip(confidences.tolist(), indices.tolist(), strict=True)
    ]


def classify(model, labels: Sequence[str], payload) -> dict:
    batch = torch.stack([decode_image_b64(encoded) for encoded in parse_instances(payload)])
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch).float(), dim=1)
    return {"predictions": format_predictions(probabilities, labels)}


class Cifar10Model(Model):
    def __init__(self, name: str, model_dir: Path) -> None:
        super().__init__(name)
        self.model_dir = model_dir
        self._model = None
        self._labels: list[str] = []

    def load(self) -> bool:
        self._model, self._labels = load_exported(self.model_dir)
        self.ready = True
        return self.ready

    def predict(self, payload, headers=None) -> dict:
        try:
            return classify(self._model, self._labels, payload)
        except ValueError as err:
            raise InvalidInput(str(err)) from err


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    parser.add_argument("--model_dir", type=Path, default=Path("/mnt/models"))
    args, _ = parser.parse_known_args(argv)
    model = Cifar10Model(args.model_name, args.model_dir)
    model.load()
    ModelServer().start([model])


if __name__ == "__main__":
    main()
