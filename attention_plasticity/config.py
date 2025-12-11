"""Configuration loader for attention plasticity analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .head_analysis import DEFAULT_DATASET_NAME


@dataclass(frozen=True)
class RunnerConfig:
    model_dir: str
    num_layers: int
    num_q_heads: int
    num_k_heads: int
    max_tokens_per_head: int = 50000
    normality_max_dims: int = 64
    p_alpha: float = 0.01
    output_csv: str = "head_metrics.csv"
    bucket_csv: str = "head_bucket_plasticity.csv"
    seed: int = 0
    max_workers: Optional[int] = None
    dataset_name: str = DEFAULT_DATASET_NAME

    def with_overrides(self, **overrides: Any) -> "RunnerConfig":
        data = asdict(self)
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in data:
                raise ValueError(f"Unknown config override '{key}'")
            data[key] = value
        return RunnerConfig(**data)

    def resolve_worker_count(self) -> int:
        if self.max_workers is None:
            return max(1, (os_cpu_count() or 1))
        return max(1, int(self.max_workers))

    def task_kwargs(self) -> Dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "max_tokens_per_head": self.max_tokens_per_head,
            "normality_max_dims": self.normality_max_dims,
            "p_alpha": self.p_alpha,
            "seed": self.seed,
            "dataset_name": self.dataset_name,
        }


def os_cpu_count() -> Optional[int]:
    from os import cpu_count

    return cpu_count()


def load_runner_config(path: str) -> RunnerConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{path}' not found.")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML structure must be a mapping.")

    required = ["model_dir", "num_layers", "num_q_heads", "num_k_heads"]
    missing = [key for key in required if key not in data or data[key] is None]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    return RunnerConfig(**data)
