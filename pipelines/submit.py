"""Submit a cifar10-gpu-mlops run to Kubeflow Pipelines (requires `make ui` port-forward)."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from cifar_mlops.evaluate import validate_threshold
from pipelines.cifar10_pipeline import cifar10_pipeline

DEFAULT_HOST = "http://localhost:8080"
DEFAULT_EXPERIMENT = "cifar10"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.85, help="minimum test accuracy to deploy (0-1)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    args = parser.parse_args(argv)
    if args.epochs < 0:
        parser.error("--epochs must be >= 0")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    try:
        validate_threshold(args.threshold)
    except ValueError as err:
        parser.error(str(err))
    return args


def build_arguments(args: argparse.Namespace) -> dict:
    return {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "accuracy_threshold": args.threshold,
    }


def main(argv: Sequence[str] | None = None) -> None:
    import kfp

    args = parse_args(argv)
    client = kfp.Client(host=args.host)
    run = client.create_run_from_pipeline_func(
        cifar10_pipeline, arguments=build_arguments(args), experiment_name=args.experiment
    )
    print(f"Submitted run {run.run_id}")
    print(f"Watch it at {args.host}/#/runs/details/{run.run_id}")


if __name__ == "__main__":
    main()
