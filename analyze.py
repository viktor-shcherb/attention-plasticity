#!/usr/bin/env python
"""
Head-level positional analysis for qk-sniffer dumps.

Per head (layer, query_head, key_head), this script reports:
- How much the query is predictable from position alone (multioutput and 1D).
- How well the residuals of the first component and the noise subspace fit a Normal distribution.
- Attention plasticity (averaged over buckets and key pairs) for each head.

Results are written to a CSV with one row per head.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from attention_plasticity.config import RunnerConfig, load_runner_config
from attention_plasticity.dataset_cache import prepare_local_dataset
from attention_plasticity.head_analysis import analyze_head

load_dotenv()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Per-head positional analysis for qk-sniffer dumps."
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Override: base directory for the sniffer dumps.",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Override: datasets hub identifier to load vectors from.",
    )
    p.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for snapshot_download (default Hugging Face cache).",
    )
    p.add_argument(
        "--dataset_local_root",
        type=str,
        default=None,
        help="Use an already-downloaded dataset located at this path; skips snapshot_download.",
    )
    p.add_argument(
        "--download_max_workers",
        type=int,
        default=None,
        help="Override: max_workers passed to snapshot_download when fetching the dataset.",
    )
    p.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Override: number of transformer layers.",
    )
    p.add_argument(
        "--num_q_heads",
        type=int,
        default=None,
        help="Override: number of query heads per layer.",
    )
    p.add_argument(
        "--num_k_heads",
        type=int,
        default=None,
        help="Override: number of key heads per layer.",
    )
    p.add_argument(
        "--max_tokens_per_head",
        type=int,
        default=None,
        help="Override: maximum number of tokens to use per head.",
    )
    p.add_argument(
        "--normality_max_dims",
        type=int,
        default=None,
        help="Override: maximum dimensions for KS tests.",
    )
    p.add_argument(
        "--p_alpha",
        type=float,
        default=None,
        help="Override: significance threshold for KS stats.",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Override: output CSV file for per-head metrics.",
    )
    p.add_argument(
        "--bucket_csv",
        type=str,
        default=None,
        help="Override: output CSV file for per-bucket plasticities.",
    )
    p.add_argument(
        "--component_csv",
        type=str,
        default=None,
        help="Override: output CSV file for per-component positional weights.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override: random seed for subsampling and KS dims.",
    )
    p.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Override: number of worker processes (1 disables parallelism).",
    )
    return p.parse_args(argv)


def _head_specs(num_layers: int, num_q_heads: int, group_size: int) -> Iterable[Tuple[int, int, int]]:
    for layer in range(num_layers):
        for q_head in range(num_q_heads):
            k_head = q_head // group_size
            yield layer, q_head, k_head


def _analyze_head_task(payload: Tuple[int, int, int, Dict[str, object]]):
    layer, q_head, k_head, kwargs = payload
    try:
        row, bucket_rows, component_rows = analyze_head(
            layer=layer,
            q_head=q_head,
            k_head=k_head,
            **kwargs,
        )
        return layer, q_head, k_head, row, bucket_rows, component_rows, None
    except Exception as exc:  # pragma: no cover - surfaced through result handling
        return layer, q_head, k_head, None, [], [], f"{exc.__class__.__name__}: {exc}"


def _run_tasks(
    specs: Iterable[Tuple[int, int, int]],
    kwargs: Dict[str, object],
    max_workers: int,
):
    rows: List[Dict[str, object]] = []
    bucket_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []
    tasks = [(layer, q_head, k_head, kwargs) for layer, q_head, k_head in specs]
    if not tasks:
        return rows, bucket_rows, component_rows

    if max_workers == 1:
        for payload in tasks:
            result = _analyze_head_task(payload)
            _process_result(result, rows, bucket_rows, component_rows)
        return rows, bucket_rows, component_rows

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_analyze_head_task, payload): payload[:3]
            for payload in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            _process_result(result, rows, bucket_rows, component_rows)
    return rows, bucket_rows, component_rows


def _process_result(
    result,
    rows: List[Dict[str, object]],
    bucket_rows: List[Dict[str, object]],
    component_rows: List[Dict[str, object]],
):
    layer, q_head, k_head, row, buckets, components, error = result
    if error:
        print(
            f"Error at layer {layer}, q_head {q_head}, k_head {k_head}: {error}",
            file=sys.stderr,
            flush=True,
        )
    else:
        rows.append(row)
        bucket_rows.extend(buckets)
        component_rows.extend(components)
        print(
            f"Analyzed layer {layer}, q_head {q_head}, k_head {k_head}",
            flush=True,
        )


def main(argv=None):
    args = parse_args(argv)
    base_config = load_runner_config(args.config)
    overrides = {
        field: getattr(args, field)
        for field in [
            "model_dir",
            "dataset_name",
            "dataset_cache_dir",
            "dataset_local_root",
            "download_max_workers",
            "num_layers",
            "num_q_heads",
            "num_k_heads",
            "max_tokens_per_head",
            "normality_max_dims",
            "p_alpha",
            "output_csv",
            "bucket_csv",
            "component_csv",
            "seed",
            "max_workers",
        ]
        if getattr(args, field) is not None
    }
    config = base_config.with_overrides(**overrides)

    dataset_local_root = config.dataset_local_root
    if dataset_local_root:
        dataset_root_path = Path(dataset_local_root).expanduser().resolve()
        dataset_local_root = str(dataset_root_path)
        print(f"Using existing dataset snapshot at {dataset_root_path}")
    else:
        print(
            f"Downloading dataset '{config.dataset_name}' "
            f"(split directory '{config.model_dir}') via snapshot_download..."
        )
        dataset_root_path = prepare_local_dataset(
            repo_id=config.dataset_name,
            model_dir=config.model_dir,
            cache_dir=config.dataset_cache_dir,
            max_workers=config.download_max_workers,
        ).resolve()
        dataset_local_root = str(dataset_root_path)
        print(f"Dataset cached under {dataset_root_path}")

    np.random.seed(config.seed)

    if config.num_q_heads % config.num_k_heads != 0:
        raise ValueError(
            f"num_q_heads ({config.num_q_heads}) must be a multiple of num_k_heads "
            f"({config.num_k_heads}) to define a GQA mapping."
        )
    group_size = config.num_q_heads // config.num_k_heads
    worker_count = config.resolve_worker_count()

    specs = list(_head_specs(config.num_layers, config.num_q_heads, group_size))
    task_kwargs = config.task_kwargs()
    task_kwargs["dataset_local_root"] = dataset_local_root
    rows, bucket_rows, component_rows = _run_tasks(specs, task_kwargs, worker_count)

    if not rows:
        print("No heads analyzed successfully; nothing to save.")
        return

    os.makedirs(Path(config.output_csv).parent, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(config.output_csv, index=False)
    print(f"Saved per-head metrics to {config.output_csv}")
    bucket_parent = Path(config.bucket_csv).parent
    bucket_parent.mkdir(parents=True, exist_ok=True)
    component_parent = Path(config.component_csv).parent
    component_parent.mkdir(parents=True, exist_ok=True)

    if bucket_rows:
        buckets_df = pd.DataFrame(bucket_rows)
        buckets_df.to_csv(config.bucket_csv, index=False)
        print(f"Saved per-bucket plasticities to {config.bucket_csv}")
    else:
        print("No per-bucket plasticities to save.")

    if component_rows:
        components_df = pd.DataFrame(component_rows)
        components_df.to_csv(config.component_csv, index=False)
        print(f"Saved per-component positional weights to {config.component_csv}")
    else:
        print("No component weights to save.")

    # Simple summary statistics
    key_cols = [
        "q_R2_pos",
        "q_R2_scalar",
        "q_mae_ratio",
        "q_resid0_kurt",
        "q_noise_med_abs_skew",
        "q_noise_med_abs_kurt",
        "q_noise_frac_ks_p_lt_alpha",
        "ap_overall",
    ]
    existing_cols = [c for c in key_cols if c in df.columns]
    if existing_cols:
        print("\nSummary over heads (selected metrics):")
        print(df[existing_cols].describe())

    if "ap_overall" in df.columns:
        print("\nModel-level attention plasticity (ap_overall over heads):")
        print(df["ap_overall"].describe())


if __name__ == "__main__":
    main()
