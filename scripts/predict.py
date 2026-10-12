"""Send random CIFAR-10 test images to the deployed model and compare predictions with the true labels."""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

from cifar_mlops.data import CIFAR10_CLASSES
from cifar_mlops.imaging import encode_png_b64
from cifar_mlops.predictor import MAX_INSTANCES

DATA_CACHE = Path.home() / ".cache" / "cifar10"
REQUEST_TIMEOUT_S = 30


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as err:
        sys.exit(f"Server returned HTTP {err.code}: {err.read().decode(errors='replace')}")
    except urllib.error.URLError as err:
        sys.exit(f"Could not reach {url}: {err.reason}. Is the port-forward running?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8081")
    parser.add_argument("--model", default="cifar10")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not 1 <= args.count <= MAX_INSTANCES:
        parser.error(f"--count must be between 1 and {MAX_INSTANCES}")

    from torchvision.datasets import CIFAR10

    test_set = CIFAR10(str(DATA_CACHE), train=False, download=True)
    indices = random.Random(args.seed).sample(range(len(test_set.data)), args.count)
    payload = {"instances": [{"image_b64": encode_png_b64(test_set.data[i])} for i in indices]}

    result = post_json(f"{args.url}/v1/models/{args.model}:predict", payload)

    correct = 0
    print(f"{'actual':<12}{'predicted':<12}confidence")
    for index, prediction in zip(indices, result["predictions"]):
        actual = CIFAR10_CLASSES[test_set.targets[index]]
        correct += actual == prediction["label"]
        mark = "✓" if actual == prediction["label"] else "✗"
        print(f"{actual:<12}{prediction['label']:<12}{prediction['confidence']:.2f} {mark}")
    print(f"\n{correct}/{args.count} correct")


if __name__ == "__main__":
    main()
