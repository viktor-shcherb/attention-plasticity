"""Attention plasticity computation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

NUM_PAIRS_PER_BUCKET = 256


def compute_attention_plasticity(
    p_q: np.ndarray,
    b_q: np.ndarray,
    X_q_rot: np.ndarray,
    alpha_rot: np.ndarray,
    beta_rot: np.ndarray,
    resid_var: np.ndarray,
    p_k: np.ndarray,
    b_k: np.ndarray,
    X_k_rot: np.ndarray,
    num_pairs_per_bucket: int = NUM_PAIRS_PER_BUCKET,
    seed: int = 0,
    bucket_window_limits: Optional[Dict[int, int]] = None,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute attention plasticity for a head:
      - For each bucket b, sample key pairs (k1,k2) with keys from buckets < b.
      - For queries in bucket b, approximate p(k1,k2) = P(q·k1 > q·k2 | bucket b)
        using Normal noise model on q as derived in the rotated basis.
      - For each pair, define pairwise plasticity PP = 4 p(1-p).
      - Head-level plasticity is the average PP over buckets and key pairs.

    Uses:
      - Linear mean model: mu(t) = alpha_rot + beta_rot * t.
      - Noise covariance: diag(resid_var) from residuals (position drift removed).
    """
    rng = np.random.default_rng(seed)

    # Representative position per bucket
    buckets = np.unique(b_q.astype(int))
    sigma2 = resid_var.astype(np.float64)
    sigma2_safe = np.where(sigma2 <= 0, 1e-12, sigma2)

    ap_bucket: Dict[int, float] = {}

    b_q_int = b_q.astype(int)
    b_k_int = b_k.astype(int)

    for bucket in buckets:
        # Queries in this bucket
        mask_q_b = b_q_int == bucket
        if not mask_q_b.any():
            ap_bucket[bucket] = np.nan
            continue

        t_b = float(p_q[mask_q_b].mean())

        # Keys from strictly earlier buckets
        key_mask = b_k_int < bucket
        if bucket_window_limits and bucket in bucket_window_limits:
            span = max(int(bucket_window_limits[bucket]), 1)
            min_bucket = bucket - span
            key_mask = np.logical_and(key_mask, b_k_int >= min_bucket)
        key_idx = np.where(key_mask)[0]
        if key_idx.size < 2:
            ap_bucket[bucket] = np.nan
            continue

        # Mean query vector at representative position t_b (in rotated space)
        mu_tb = alpha_rot + beta_rot * t_b  # (d,)

        # Number of pairs to sample (cap by total possible distinct pairs)
        max_pairs = key_idx.size * (key_idx.size - 1) // 2
        if max_pairs <= 0:
            ap_bucket[bucket] = np.nan
            continue
        num_pairs = min(num_pairs_per_bucket, max_pairs)

        pp_vals = []
        for _ in range(num_pairs):
            # Sample two distinct keys
            i1, i2 = rng.choice(key_idx, size=2, replace=False)
            k1 = X_k_rot[i1]
            k2 = X_k_rot[i2]
            delta_k = k1 - k2  # (d,)

            # Mean and variance of logit difference d = q·(k1-k2)
            m_d = float(mu_tb @ delta_k)
            v_d = float((sigma2_safe * (delta_k ** 2)).sum())
            if v_d <= 0:
                continue

            z = m_d / np.sqrt(v_d)
            p_pair = norm.cdf(z)
            pp = 4.0 * p_pair * (1.0 - p_pair)
            pp_vals.append(pp)

        if pp_vals:
            ap_bucket[bucket] = float(np.mean(pp_vals))
        else:
            ap_bucket[bucket] = np.nan

    # Aggregate per-head plasticity as mean over buckets (ignoring NaNs)
    vals = [v for v in ap_bucket.values() if not np.isnan(v)]
    ap_overall = float(np.mean(vals)) if vals else np.nan

    return ap_overall, ap_bucket
