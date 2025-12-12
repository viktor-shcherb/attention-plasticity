"""Attention plasticity computation."""

from __future__ import annotations

from collections import defaultdict
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
    example_ids_k: np.ndarray,
    X_k_rot: np.ndarray,
    num_pairs_per_bucket: int = NUM_PAIRS_PER_BUCKET,
    seed: int = 0,
    bucket_window_limits: Optional[Dict[int, int]] = None,
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """
    Compute attention plasticity for a head:
      - For each query bucket b > 0, and each eligible key bucket c < b,
        sample key pairs (k1, k2) drawn from bucket c only.
      - For queries in bucket b, approximate p(k1,k2) = P(q·k1 > q·k2 | bucket b)
        using the Normal noise model on q in the rotated basis.
      - For each pair, define pairwise plasticity PP = 4 p(1-p).
      - Head-level plasticity is the average PP over (q_bucket, k_bucket) pairs.

    Uses:
      - Linear mean model: mu(t) = alpha_rot + beta_rot * t.
      - Noise covariance: diag(resid_var) from residuals (position drift removed).
    """
    rng = np.random.default_rng(seed)

    b_q_int = b_q.astype(int)
    b_k_int = b_k.astype(int)

    q_buckets = np.unique(b_q_int)
    k_buckets = np.unique(b_k_int)

    example_ids_k = example_ids_k.astype(np.int64)
    bucket_example_indices: Dict[int, Dict[int, np.ndarray]] = {}
    temp: Dict[int, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for idx, (bucket, example_id) in enumerate(zip(b_k_int, example_ids_k)):
        temp[int(bucket)][int(example_id)].append(idx)
    for bucket, per_example in temp.items():
        bucket_example_indices[bucket] = {
            example_id: np.asarray(indices, dtype=np.int64)
            for example_id, indices in per_example.items()
        }

    sigma2 = resid_var.astype(np.float64)
    sigma2_safe = np.where(sigma2 <= 0, 1e-12, sigma2)

    ap_pairs: Dict[Tuple[int, int], float] = {}

    for q_bucket in q_buckets:
        q_bucket = int(q_bucket)
        if q_bucket <= 0:
            # Bucket 0 cannot attend to earlier tokens yet.
            continue

        mask_q_b = b_q_int == q_bucket
        if not mask_q_b.any():
            continue

        t_b = float(p_q[mask_q_b].mean())
        min_allowed_bucket: Optional[int] = None
        if bucket_window_limits and q_bucket in bucket_window_limits:
            span = int(bucket_window_limits[q_bucket])
            span = max(span, 1)
            min_allowed_bucket = q_bucket - span

        candidate_key_buckets = []
        for kb in k_buckets:
            kb = int(kb)
            if kb >= q_bucket:
                continue
            if min_allowed_bucket is not None and kb < min_allowed_bucket:
                continue
            candidate_key_buckets.append(kb)
        if not candidate_key_buckets:
            continue

        mu_tb = alpha_rot + beta_rot * t_b  # (d,)

        for k_bucket in candidate_key_buckets:
            example_groups = [
                indices
                for indices in bucket_example_indices.get(k_bucket, {}).values()
                if indices.size >= 2
            ]
            if not example_groups:
                ap_pairs[(q_bucket, k_bucket)] = np.nan
                continue

            pair_counts = np.array(
                [group.size * (group.size - 1) // 2 for group in example_groups],
                dtype=np.int64,
            )
            total_pairs = int(pair_counts.sum())
            if total_pairs <= 0:
                ap_pairs[(q_bucket, k_bucket)] = np.nan
                continue
            num_pairs = min(num_pairs_per_bucket, total_pairs)

            cum_counts = np.cumsum(pair_counts)
            pp_vals = []
            for _ in range(num_pairs):
                r = rng.integers(total_pairs)
                group_idx = int(np.searchsorted(cum_counts, r, side="right"))
                indices = example_groups[group_idx]
                i1, i2 = rng.choice(indices, size=2, replace=False)
                k1 = X_k_rot[i1]
                k2 = X_k_rot[i2]
                delta_k = k1 - k2

                m_d = float(mu_tb @ delta_k)
                v_d = float((sigma2_safe * (delta_k ** 2)).sum())
                if v_d <= 0:
                    continue

                z = m_d / np.sqrt(v_d)
                p_pair = norm.cdf(z)
                pp_vals.append(4.0 * p_pair * (1.0 - p_pair))

            if pp_vals:
                ap_pairs[(q_bucket, k_bucket)] = float(np.mean(pp_vals))
            else:
                ap_pairs[(q_bucket, k_bucket)] = np.nan

    vals = [v for v in ap_pairs.values() if not np.isnan(v)]
    ap_overall = float(np.mean(vals)) if vals else np.nan

    return ap_overall, ap_pairs
