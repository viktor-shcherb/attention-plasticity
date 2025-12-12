#!/usr/bin/env python
"""
Plot a heatmap of attention plasticity aggregated by (q_bucket, k_bucket).

The script consumes the per-bucket CSV emitted by analyze.py
(columns: layer, q_head, k_head, q_bucket, k_bucket, ap_bucket) and
shows model-level attention plasticity as a 2D histogram where
q_bucket defines the rows, k_bucket the columns, and the cell value is
the mean (or median) AP across heads.
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
        "--bucket_csv",
        type=Path,
        default=None,
        help="Path to head_bucket_plasticity.csv produced by analyze.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the heatmap (PNG, PDF, etc.). If omitted, show the figure.",
    )
    parser.add_argument(
        "--agg",
        choices=("mean", "median"),
        default=None,
        help="Aggregation applied to ap_bucket within each (q_bucket, k_bucket) pair.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="Matplotlib colormap for the heatmap.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional lower bound for the color scale.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional upper bound for the color scale.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title.",
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
        raise ValueError("Heatmap config must be a mapping.")
    return data


def resolve_options(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "bucket_csv": None,
        "output": None,
        "agg": "mean",
        "cmap": "magma",
        "vmin": None,
        "vmax": None,
        "title": "Attention Plasticity by Bucket Pair",
        "bucket_type": "uniform",
        "bucket_min_size": 1.0,
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

    if resolved["bucket_csv"] is None:
        raise ValueError("bucket_csv must be provided via CLI or config.")
    resolved["bucket_csv"] = Path(resolved["bucket_csv"])
    if resolved["output"] is not None:
        resolved["output"] = Path(resolved["output"])
    if resolved["agg"] not in ("mean", "median"):
        raise ValueError("--agg must be 'mean' or 'median'.")
    resolved["bucket_type"] = resolved["bucket_type"] or "uniform"
    if resolved["bucket_type"] not in ("uniform", "log"):
        raise ValueError("--bucket_type must be 'uniform' or 'log'.")
    resolved["bucket_min_size"] = float(resolved["bucket_min_size"])
    if resolved["bucket_min_size"] <= 0:
        raise ValueError("--bucket_min_size must be positive.")
    return resolved


def load_bucket_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Bucket CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"q_bucket", "k_bucket", "ap_bucket"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Bucket CSV must contain q_bucket, k_bucket, and ap_bucket columns "
            f"(missing: {', '.join(sorted(missing))})"
        )
    df = df.dropna(subset=["ap_bucket"])
    if df.empty:
        raise ValueError("No non-NaN ap_bucket entries to plot.")
    return df


def aggregate(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    q_levels = np.sort(df["q_bucket"].unique())
    k_levels = np.sort(df["k_bucket"].unique())
    group = df.groupby(["q_bucket", "k_bucket"])["ap_bucket"]
    if agg == "median":
        aggregated = group.median()
    else:
        aggregated = group.mean()
    matrix = (
        aggregated.reset_index()
        .pivot(index="q_bucket", columns="k_bucket", values="ap_bucket")
        .reindex(index=q_levels, columns=k_levels)
    )
    return matrix


def mask_noncausal(matrix: pd.DataFrame) -> pd.DataFrame:
    masked = matrix.copy()
    col_vals = masked.columns.to_numpy(dtype=float)
    for q_bucket in masked.index.to_numpy(dtype=float):
        mask = col_vals >= q_bucket
        masked.loc[q_bucket, mask] = np.nan
    return masked


def _bucket_starts(indices: np.ndarray, bucket_type: str, bucket_min_size: float) -> np.ndarray:
    if bucket_type == "log":
        return (2.0 ** indices) * bucket_min_size
    return indices * bucket_min_size


def _format_positions(values: np.ndarray) -> list[str]:
    labels = []
    for val in values:
        if abs(val - round(val)) < 1e-9:
            labels.append(f"{int(round(val))}")
        else:
            labels.append(f"{val:.2f}")
    return labels


def plot_heatmap(matrix: pd.DataFrame, opts: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    data = matrix.to_numpy(dtype=float)
    k_indices = matrix.columns.to_numpy(dtype=float)
    q_indices = matrix.index.to_numpy(dtype=float)
    k_positions = _bucket_starts(k_indices, opts["bucket_type"], opts["bucket_min_size"])
    q_positions = _bucket_starts(q_indices, opts["bucket_type"], opts["bucket_min_size"])
    cmap = plt.get_cmap(opts["cmap"]).copy()
    cmap.set_bad(color="white")
    masked_data = np.ma.masked_invalid(data)

    im = ax.imshow(
        masked_data,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=opts["vmin"],
        vmax=opts["vmax"],
    )
    key_labels = _format_positions(k_positions)
    query_labels = _format_positions(q_positions)

    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(query_labels)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(key_labels, rotation=45, ha="right")
    ax.set_xlabel("Key pair position")
    ax.set_ylabel("Query position")
    ax.set_title(opts["title"])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention plasticity")
    fig.tight_layout()

    if opts["output"]:
        output_path = Path(opts["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    opts = resolve_options(args)
    df = load_bucket_frame(opts["bucket_csv"])
    matrix = aggregate(df, opts["agg"])
    matrix = mask_noncausal(matrix)
    plot_heatmap(matrix, opts)


if __name__ == "__main__":
    main()
