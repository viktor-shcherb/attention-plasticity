import textwrap

import pytest

from attention_plasticity.config import RunnerConfig, load_runner_config


def write_config(tmp_path, content: str):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_load_runner_config_with_overrides(tmp_path):
    path = write_config(
        tmp_path,
        """
        model_dir: /model
        dataset_name: custom/ds
        num_layers: 2
        num_q_heads: 4
        num_k_heads: 2
        seed: 123
        """,
    )

    cfg = load_runner_config(str(path))
    assert cfg.model_dir == "/model"
    assert cfg.dataset_name == "custom/ds"
    assert cfg.num_layers == 2
    assert cfg.max_workers is None

    updated = cfg.with_overrides(max_workers=5, output_csv="custom.csv")
    assert updated.max_workers == 5
    assert updated.output_csv == "custom.csv"
    assert updated.model_dir == "/model"


def test_load_runner_config_missing_required(tmp_path):
    path = write_config(
        tmp_path,
        """
        model_dir: /model
        num_layers: 2
        """,
    )

    with pytest.raises(ValueError):
        load_runner_config(str(path))
