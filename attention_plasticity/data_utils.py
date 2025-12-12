"""Utilities for loading and orienting q/k datasets."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def fit_bucket_stats(ds) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Per-bucket mean/std/n plus effective positions (empirical + geometric).
    Used to compute orientation and diagnostics.
    """
    df = ds.to_pandas()
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for bucket, sub in df.groupby("bucket", sort=True):
        X = np.vstack(sub["vector"].values).astype(np.float64)
        out[bucket] = {
            "mean": X.mean(axis=0),
            "std": X.std(axis=0, ddof=1) if len(X) > 1 else np.zeros(X.shape[1]),
            "n": len(X),
            "p_eff_emp": float(sub["position"].mean()),
            "p_eff_geom": float(2.0 ** (float(bucket) + 0.5)),
        }
    return out


def orientation_from_keys(keys_ds) -> np.ndarray:
    """
    Orientation vector based on the sign of the average per-dimension key means
    across buckets. Ensures a roughly canonical sign across heads.
    """
    k_stats = fit_bucket_stats(keys_ds)
    mu_k = np.vstack([k_stats[b]["mean"] for b in sorted(k_stats)])
    s = np.sign(mu_k.mean(axis=0))
    s[s == 0] = 1.0
    return s


def stack_oriented_examples(
    ds,
    orient: np.ndarray,
    max_tokens: Optional[int] = None,
    seed: int = 0,
    return_sliding_window: bool = False,
    return_example_id: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Return (positions, bucket_idx, oriented_vectors) from a dataset.
    Optionally subsample to max_tokens.
    """
    df = ds.to_pandas()
    if max_tokens is not None and len(df) > max_tokens:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_tokens, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    p = df["position"].to_numpy().astype(np.float64)
    b = df["bucket"].to_numpy().astype(np.float64)
    X = np.vstack(df["vector"].values).astype(np.float64) * orient[None, :]
    outputs: list = [p, b, X]

    if return_sliding_window:
        if "sliding_window" in df.columns:
            sliding = df["sliding_window"].to_numpy(dtype=np.float64)
        else:
            sliding = np.zeros(len(df), dtype=np.float64)
        outputs.append(sliding)

    if return_example_id:
        if "example_id" in df.columns:
            example_ids = df["example_id"].to_numpy()
        else:
            example_ids = np.arange(len(df), dtype=np.int64)
        outputs.append(example_ids.astype(np.int64, copy=False))

    return tuple(outputs)
