"""Per-head attention analysis orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_url

from .data_utils import orientation_from_keys, stack_oriented_examples
from .plasticity import NUM_PAIRS_PER_BUCKET, compute_attention_plasticity
from .statistics import (
    compute_first_component_residual_stats,
    compute_noise_normality_stats,
    compute_scalar_position_predictability,
    fit_multioutput_linear_regressor,
    make_pos_rotation,
)


DEFAULT_DATASET_NAME = "viktoroo/sniffed-qk"


def analyze_head(
    layer: int,
    q_head: int,
    k_head: int,
    model_dir: str,
    max_tokens_per_head: int,
    normality_max_dims: int,
    p_alpha: float,
    seed: int = 0,
    num_pairs_per_bucket: int = NUM_PAIRS_PER_BUCKET,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_local_root: Optional[str] = None,
):
    """
    Perform analysis for a single (layer, query_head, key_head) and
    return a dict of per-head metrics.
    """

    split = model_dir
    config_q = f"l{layer:02d}h{q_head:02d}q"
    config_k = f"l{layer:02d}h{k_head:02d}k"
    base_path = Path(dataset_local_root).expanduser() if dataset_local_root else None

    def _resolve_data_file(config_name: str) -> str:
        if base_path:
            candidate = base_path / split / config_name / "data.parquet"
            if not candidate.exists():
                raise FileNotFoundError(f"Local parquet not found: {candidate}")
            return str(candidate)
        return hf_hub_url(
            repo_id=dataset_name,
            filename=f"{split}/{config_name}/data.parquet",
            repo_type="dataset",
        )

    url_q = _resolve_data_file(config_q)
    url_k = _resolve_data_file(config_k)
    queries = load_dataset("parquet", data_files=url_q, split="train")
    keys = load_dataset("parquet", data_files=url_k, split="train")

    # Orientation from keys, then oriented q/k
    orient = orientation_from_keys(keys)

    p_q, b_q, X_q, q_sliding = stack_oriented_examples(
        queries,
        orient,
        max_tokens=max_tokens_per_head,
        seed=seed,
        return_sliding_window=True,
    )
    p_k, b_k, X_k, k_example_ids = stack_oriented_examples(
        keys, orient, max_tokens=max_tokens_per_head, seed=seed, return_example_id=True
    )

    sliding_value = _extract_sliding_window_value(
        q_sliding, layer=layer, q_head=q_head
    )
    bucket_window_limits = None
    if sliding_value is not None:
        bucket_window_limits = _build_bucket_window_limits(
            b_q.astype(int),
            sliding_value,
        )

    n_q, d_q = X_q.shape
    n_k, d_k = X_k.shape
    if d_q != d_k:
        raise ValueError(
            f"q/k dimension mismatch at layer {layer}, q_head {q_head}, k_head {k_head}: "
            f"d_q={d_q}, d_k={d_k}"
        )

    # 1) Query predictability from position (multioutput) in original oriented space
    _, beta_q_p, r2_components, r2q_p = fit_multioutput_linear_regressor(p_q, X_q)
    beta_norm = float(np.linalg.norm(beta_q_p))

    # 2) Scalar position predictability via projection along beta_q_p
    r2_scalar, mae_baseline, mae_proj = compute_scalar_position_predictability(
        p_q, X_q, beta_q_p
    )
    mae_ratio = mae_proj / mae_baseline if mae_baseline > 0 else 1.0

    # 3) Rotate to put positional slope in first component
    H = make_pos_rotation(beta_q_p)
    X_q_rot = X_q @ H
    X_k_rot = X_k @ H

    # Fit linear model in rotated basis to extract drift and noise for all components
    alpha_rot, beta_rot, _, _ = fit_multioutput_linear_regressor(p_q, X_q_rot)
    mu_hat = alpha_rot[None, :] + p_q[:, None] * beta_rot[None, :]

    R = X_q_rot - mu_hat  # residuals
    if R.shape[0] < 2:
        resid_var = np.ones(d_q, dtype=np.float64)
    else:
        resid_var = R.var(axis=0, ddof=1)

    # 4) Residual normality for first component (queries)
    resid0_skew, resid0_kurt, resid0_ks_p = compute_first_component_residual_stats(
        p_q, X_q_rot
    )

    # 5) Noise subspace normality (queries)
    noise_med_abs_skew, noise_med_abs_kurt, noise_med_ks_p, noise_frac_lt_alpha = (
        compute_noise_normality_stats(
            X_q_rot,
            max_dims=normality_max_dims,
            alpha=p_alpha,
            seed=seed,
        )
    )

    # 6) Attention plasticity (per head, averaged over buckets and key pairs)
    ap_seed = seed + 10007 * layer + 379 * q_head
    ap_overall, ap_bucket_pairs = compute_attention_plasticity(
        p_q=p_q,
        b_q=b_q,
        X_q_rot=X_q_rot,
        alpha_rot=alpha_rot,
        beta_rot=beta_rot,
        resid_var=resid_var,
        p_k=p_k,
        b_k=b_k,
        example_ids_k=k_example_ids,
        X_k_rot=X_k_rot,
        num_pairs_per_bucket=num_pairs_per_bucket,
        seed=ap_seed,
        bucket_window_limits=bucket_window_limits,
    )

    row = {
        "layer": layer,
        "q_head": q_head,
        "k_head": k_head,
        "n_q_tokens": n_q,
        "n_k_tokens": n_k,
        "d_model": d_q,
        # Positional predictability (queries)
        "q_R2_pos": r2q_p,
        "q_beta_norm": beta_norm,
        "q_R2_scalar": r2_scalar,
        "q_mae_baseline": mae_baseline,
        "q_mae_proj": mae_proj,
        "q_mae_ratio": mae_ratio,
        # First-component residual normality (queries)
        "q_resid0_skew": resid0_skew,
        "q_resid0_kurt": resid0_kurt,
        "q_resid0_ks_p": resid0_ks_p,
        # Noise subspace normality (queries)
        "q_noise_med_abs_skew": noise_med_abs_skew,
        "q_noise_med_abs_kurt": noise_med_abs_kurt,
        "q_noise_med_ks_p": noise_med_ks_p,
        "q_noise_frac_ks_p_lt_alpha": noise_frac_lt_alpha,
        # Attention plasticity (per head)
        "ap_overall": ap_overall,
    }

    r2_positive = np.clip(r2_components, 0.0, None)
    total_info = float(r2_positive.sum())
    if total_info <= 0:
        component_weights = np.full_like(r2_positive, 1.0 / d_q)
    else:
        component_weights = r2_positive / total_info

    component_rows = [
        {
            "layer": layer,
            "q_head": q_head,
            "k_head": k_head,
            "component": int(idx),
            "component_r2": float(r2_components[idx]),
            "component_weight": float(component_weights[idx]),
        }
        for idx in range(d_q)
    ]

    bucket_rows = []
    for (q_bucket, k_bucket), value in sorted(ap_bucket_pairs.items()):
        if q_bucket <= 0:
            continue
        bucket_rows.append(
            {
                "layer": layer,
                "q_head": q_head,
                "k_head": k_head,
                "q_bucket": int(q_bucket),
                "k_bucket": int(k_bucket),
                "ap_bucket": value,
            }
        )

    return row, bucket_rows, component_rows


def _extract_sliding_window_value(
    sliding_values: np.ndarray, layer: int, q_head: int
) -> Optional[float]:
    if sliding_values.size == 0:
        return None
    unique = np.unique(sliding_values.astype(float))
    if unique.size == 0:
        return None
    if unique.size > 1:
        raise ValueError(
            f"Mixed sliding_window values for layer {layer}, q_head {q_head}"
        )
    value = unique[0]
    if np.isnan(value) or value <= 0:
        return None
    return float(value)


def _build_bucket_window_limits(
    buckets: np.ndarray,
    span_value: float,
) -> Optional[Dict[int, int]]:
    lookback = int(np.floor(span_value))
    if lookback <= 0:
        lookback = 1
    bucket_ids = np.unique(buckets.astype(int))
    if bucket_ids.size == 0:
        return None
    limits = {int(bucket): lookback for bucket in bucket_ids}
    return limits
