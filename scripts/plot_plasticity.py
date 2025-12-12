#!/usr/bin/env python
"""
Plot per-bucket attention plasticity for a model.

The script expects the per-bucket CSV emitted by analyze.py (columns:
layer, q_head, k_head, q_bucket, k_bucket, ap_bucket). It aggregates
over k_bucket, draws faint lines for each head, and a bold red line for
the mean across heads. Additional options let you control bucket
type/scale, add model labels, and color heads by layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
        help="Optional YAML config file with plotting options.",
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
        help="Optional path to save the plot (e.g., plot.png). If omitted, shows the plot interactively.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha (opacity) for individual head lines.",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=None,
        help="Line width for individual head lines.",
    )
    parser.add_argument(
        "--bucket_type",
        choices=("log", "uniform"),
        default=None,
        help="Bucket spacing type: 'log' (2^(b+0.5)) or 'uniform' (min_size * (b+0.5)).",
    )
    parser.add_argument(
        "--bucket_min_size",
        type=float,
        default=None,
        help="Minimum bucket width (used for uniform buckets, multiplier for log buckets).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional model name to include in the plot title.",
    )
    parser.add_argument(
        "--color_by_layer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle: color head traces using a gradient by layer index.",
    )
    parser.add_argument(
        "--color_trend",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle: color lines by trend (increasing/decreasing/flat).",
    )
    parser.add_argument(
        "--trend_threshold",
        type=float,
        default=None,
        help="Threshold on slope magnitude to classify trends (default auto).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="Matplotlib colormap name (used when --color_by_layer is set).",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Plot config must be a mapping.")
    return data


def resolve_options(args: argparse.Namespace) -> dict:
    defaults = {
        "bucket_csv": None,
        "output": None,
        "alpha": 0.25,
        "line_width": 0.8,
        "bucket_type": "uniform",
        "bucket_min_size": 1.0,
        "model_name": None,
        "color_by_layer": False,
        "color_trend": False,
        "trend_threshold": None,
        "cmap": "viridis",
    }
    config = load_config(args.config)
    options: dict = {}
    for key, default in defaults.items():
        arg_value = getattr(args, key, None)
        if arg_value is None:
            value = config.get(key, default)
        else:
            value = arg_value
        options[key] = value
    if options["bucket_csv"] is None:
        raise ValueError("bucket_csv must be provided via CLI or config.")
    options["bucket_csv"] = Path(options["bucket_csv"])
    if options["output"] is not None:
        options["output"] = Path(options["output"])
    options["color_by_layer"] = bool(options["color_by_layer"])
    options["color_trend"] = bool(options["color_trend"])
    if options["trend_threshold"] is not None and options["trend_threshold"] <= 0:
        raise ValueError("--trend_threshold must be positive if provided.")
    return options


def head_identifier(row: pd.Series) -> str:
    return f"L{int(row['layer']):02d}/Q{int(row['q_head']):02d}/K{int(row['k_head']):02d}"


def _estimate_slope(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    if len(x_vals) < 2:
        return 0.0
    if np.allclose(x_vals, x_vals[0]):
        return 0.0
    slope, _ = np.polyfit(x_vals, y_vals, 1)
    return float(slope)


def main():
    args = parse_args()
    opts = resolve_options(args)
    bucket_csv: Path = opts["bucket_csv"]
    if not bucket_csv.exists():
        raise FileNotFoundError(f"Bucket CSV not found: {bucket_csv}")

    df_raw = pd.read_csv(bucket_csv)
    new_schema = {"layer", "q_head", "k_head", "q_bucket", "k_bucket", "ap_bucket"}
    legacy_schema = {"layer", "q_head", "k_head", "bucket", "ap_bucket"}
    columns = set(df_raw.columns)
    if new_schema.issubset(columns):
        df_work = df_raw.copy()
    elif legacy_schema.issubset(columns):
        df_work = df_raw.rename(columns={"bucket": "q_bucket"}).copy()
        df_work["k_bucket"] = df_work["q_bucket"]
    else:
        raise ValueError(
            "Bucket CSV must contain either (layer,q_head,k_head,q_bucket,k_bucket,ap_bucket) "
            "or the legacy columns (layer,q_head,k_head,bucket,ap_bucket)."
        )

    df_work = df_work.dropna(subset=["ap_bucket"])
    if df_work.empty:
        raise ValueError("No per-bucket plasticity values found in CSV.")

    df = (
        df_work.groupby(["layer", "q_head", "k_head", "q_bucket"], as_index=False)[
            "ap_bucket"
        ].mean()
    )
    df = df.rename(columns={"q_bucket": "bucket"})
    df = df.sort_values(["bucket", "layer", "q_head"])
    df["head_id"] = df.apply(head_identifier, axis=1)

    if opts["bucket_min_size"] <= 0:
        raise ValueError("--bucket_min_size must be positive.")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_label = "Bucket" if opts["bucket_type"] == "log" else "Position"
    y_label = "Attention plasticity"
    if opts["color_by_layer"] and opts["color_trend"]:
        raise ValueError("Choose either --color-by-layer or --color-trend, not both.")

    cmap = plt.get_cmap(opts["cmap"])
    max_layer = max(df["layer"].max(), 1)
    trend_threshold = opts["trend_threshold"] or 1e-4

    for head_id, sub in df.groupby("head_id"):
        sub_sorted = sub.sort_values("bucket")
        bucket_vals = sub_sorted["bucket"].to_numpy(dtype=float)
        if opts["bucket_type"] == "log":
            x_vals = (2.0 ** (bucket_vals + 0.5)) * opts["bucket_min_size"]
        else:
            x_vals = (bucket_vals + 0.5) * opts["bucket_min_size"]

        if opts["color_trend"]:
            slope = _estimate_slope(x_vals, sub_sorted["ap_bucket"].to_numpy(dtype=float))
            if slope > trend_threshold:
                color = "#2ca02c"  # green for increasing
            elif slope < -trend_threshold:
                color = "#d62728"  # red for decreasing
            else:
                color = "#7f7f7f"  # gray for flat
        elif opts["color_by_layer"]:
            layer = sub_sorted["layer"].iloc[0]
            norm = layer / max_layer
            color = cmap(norm)
        else:
            color = "C0"

        ax.plot(
            x_vals,
            sub_sorted["ap_bucket"],
            color=color,
            alpha=opts["alpha"],
            linewidth=opts["line_width"],
        )

    mean_series = (
        df.groupby("bucket")["ap_bucket"].mean().reset_index().sort_values("bucket")
    )
    bucket_vals = mean_series["bucket"].to_numpy(dtype=float)
    if opts["bucket_type"] == "log":
        mean_x = (2.0 ** (bucket_vals + 0.5)) * opts["bucket_min_size"]
    else:
        mean_x = (bucket_vals + 0.5) * opts["bucket_min_size"]
    ax.plot(
        mean_x,
        mean_series["ap_bucket"],
        color="red",
        linewidth=2.5,
        label="Mean across heads",
    )

    title = "Per-bucket attention plasticity"
    if opts["model_name"]:
        title = f"{title} — {opts['model_name']}"
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()

    if opts["output"]:
        fig.savefig(opts["output"], dpi=200)
        print(f"Saved plot to {opts['output']}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
