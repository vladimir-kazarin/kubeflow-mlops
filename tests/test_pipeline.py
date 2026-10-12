import yaml
from kfp import compiler

from pipelines.cifar10_pipeline import cifar10_pipeline


def _compile(tmp_path) -> tuple[str, dict]:
    output = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(cifar10_pipeline, str(output))
    text = output.read_text()
    return text, yaml.safe_load(text)


def test_pipeline_compiles_with_gpu_training_and_gated_deploy(tmp_path):
    text, spec = _compile(tmp_path)

    assert "nvidia.com/gpu" in text
    assert "comp-deploy-op" in spec["components"]
    assert "comp-condition-1" in spec["components"]
    assert "deploy-op" not in spec["root"]["dag"]["tasks"]


def test_pipeline_parameter_defaults_match_spec(tmp_path):
    _, spec = _compile(tmp_path)
    parameters = spec["root"]["inputDefinitions"]["parameters"]

    defaults = {name: parameters[name]["defaultValue"] for name in parameters}
    assert defaults == {"epochs": 3, "learning_rate": 1e-3, "batch_size": 128, "accuracy_threshold": 0.85}
