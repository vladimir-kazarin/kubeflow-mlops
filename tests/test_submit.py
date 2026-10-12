import pytest

from pipelines.submit import build_arguments, parse_args


def test_parse_args_defaults_match_pipeline_defaults():
    args = parse_args([])

    assert (args.epochs, args.learning_rate, args.batch_size, args.threshold) == (3, 1e-3, 128, 0.85)


@pytest.mark.parametrize(
    "argv",
    [
        ["--epochs", "-1"],
        ["--learning-rate", "0"],
        ["--batch-size", "0"],
        ["--threshold", "1.5"],
        ["--threshold", "-0.1"],
        ["--threshold", "nan"],
    ],
)
def test_parse_args_rejects_invalid_values(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_build_arguments_maps_cli_flags_to_pipeline_parameters():
    args = parse_args(["--epochs", "0", "--threshold", "0.99"])

    assert build_arguments(args) == {
        "epochs": 0,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "accuracy_threshold": 0.99,
    }
