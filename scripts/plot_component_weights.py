#!/usr/bin/env python
"""
Plot average per-component positional information weights.

Consumes the component CSV emitted by analyze.py (columns:
layer, q_head, k_head, component, component_r2, component_weight). Each
head's component weights sum to 1, representing the share of linear
positional information stored in that component. This script averages
those weights across heads and renders a bar chart.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML file with default options.",
    )
    parser.add_argument(
        "--component_csv",
        type=Path,
        default=None,
        help="Path to head_component_weights.csv produced by analyze.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the bar chart (PNG, PDF, etc.). If omitted, show the figure.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of components to display (starting from 0).",
    )
    parser.add_argument(
        "--agg",
        choices=("mean", "median"),
        default=None,
        help="Aggregation applied across heads (default: mean).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title.",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="Y-axis label (defaults to 'Share of positional information').",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Component plot config must be a mapping.")
    return data


def resolve_options(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "component_csv": None,
        "output": None,
        "limit": None,
        "agg": "mean",
        "title": "Average positional information by component",
        "ylabel": "Share of positional information",
    }
    config = load_config(args.config)
    resolved: Dict[str, Any] = {}
    for key, default in defaults.items():
        arg_value = getattr(args, key, None)
        if arg_value is None:
            value = config.get(key, default)
        else:
            value = arg_value
        resolved[key] = value
    if resolved["component_csv"] is None:
        raise ValueError("component_csv must be provided via CLI or config.")
    resolved["component_csv"] = Path(resolved["component_csv"])
    if resolved["output"] is not None:
        resolved["output"] = Path(resolved["output"])
    if resolved["limit"] is not None and resolved["limit"] <= 0:
        raise ValueError("--limit must be positive if provided.")
    return resolved


def load_components(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Component CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"component", "component_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Component CSV must contain component and component_weight columns "
            f"(missing: {', '.join(sorted(missing))})"
        )
    df = df.dropna(subset=["component_weight"])
    if df.empty:
        raise ValueError("No component weights available for plotting.")
    return df


def aggregate_components(df: pd.DataFrame, agg: str, limit: int | None) -> pd.Series:
    df["component"] = df["component"].astype(int)
    if limit is not None:
        df = df[df["component"] < limit]
        if df.empty:
            raise ValueError("No components fall within the requested --limit.")
    group = df.groupby("component")["component_weight"]
    if agg == "median":
        series = group.median()
    else:
        series = group.mean()
    series = series.sort_index()
    return series


def plot_bar(series: pd.Series, opts: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    components = series.index.to_numpy()
    values = series.to_numpy()
    ax.bar(components, values, color="#1f77b4")
    ax.set_xlabel("Component index")
    ax.set_ylabel(opts["ylabel"])
    ax.set_title(opts["title"])
    ax.set_ylim(0.0, max(0.01, values.max() * 1.1))
    if len(components) <= 20:
        tick_positions = components
    else:
        tick_count = min(20, len(components))
        indices = np.linspace(0, len(components) - 1, tick_count, dtype=int)
        tick_positions = components[indices]
        tick_positions = np.unique(tick_positions)
    tick_labels = [str(int(c)) for c in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    fig.tight_layout()
    if opts["output"]:
        opts["output"].parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(opts["output"], dpi=200)
        print(f"Saved component bar chart to {opts['output']}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    opts = resolve_options(args)
    df = load_components(opts["component_csv"])
    series = aggregate_components(df, opts["agg"], opts["limit"])
    plot_bar(series, opts)


if __name__ == "__main__":
    main()
