"""Statistical helpers for positional analysis."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import kurtosis, kstest, norm, skew


def fit_multioutput_linear_regressor(
    t: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit X ≈ alpha + beta * t (single scalar regressor) in least squares sense.

    Returns:
        alpha: (d,) intercept
        beta:  (d,) slope
        r2_components: (d,) per-dimension R^2
        r2_overall: scalar, mean of per-dimension R^2
    """
    t = t.astype(np.float64)
    X = X.astype(np.float64)

    t_mean = t.mean()
    t_c = t - t_mean
    denom = float((t_c ** 2).sum())
    if denom <= 0:
        # Degenerate: no variation in t
        d = X.shape[1]
        alpha = X.mean(axis=0)
        beta = np.zeros(d, dtype=np.float64)
        r2_components = np.zeros(d, dtype=np.float64)
        return alpha, beta, r2_components, 0.0

    beta = (t_c[:, None] * X).sum(axis=0) / denom
    alpha = X.mean(axis=0) - beta * t_mean

    X_hat = alpha[None, :] + t[:, None] * beta[None, :]
    ss_res = ((X - X_hat) ** 2).sum(axis=0)
    ss_tot = ((X - X.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_components = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    r2_components = np.clip(r2_components, -np.inf, 1.0)
    r2_overall = float(np.nanmean(r2_components))
    return alpha, beta, r2_components, r2_overall


def make_pos_rotation(beta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Given a nonzero slope vector beta in R^d, construct an orthogonal matrix H
    such that H @ e1 = beta / ||beta||.
    For row vectors X, use X_rot = X @ H.
    """
    d = beta.shape[0]
    norm_beta = np.linalg.norm(beta)
    if norm_beta < eps:
        return np.eye(d, dtype=np.float64)

    u = beta / norm_beta
    e1 = np.zeros(d, dtype=np.float64)
    e1[0] = 1.0

    if np.linalg.norm(u - e1) < eps:
        return np.eye(d, dtype=np.float64)

    # Householder reflection mapping e1 -> u
    v = u - e1
    v_len = np.linalg.norm(v)
    if v_len < eps:
        return np.eye(d, dtype=np.float64)
    v /= v_len
    H = np.eye(d, dtype=np.float64) - 2.0 * np.outer(v, v)
    return H


def compute_first_component_residual_stats(
    p: np.ndarray,
    X_rot: np.ndarray,
) -> Tuple[float, float, float]:
    """
    For rotated queries X_rot (n,d), analyze the first component:
    - Fit y0 ~ a + b * p
    - Compute residuals and their normality stats.
    Returns (skew, kurtosis, ks_pvalue).
    """
    y0 = X_rot[:, 0]
    t = p.astype(np.float64)

    t_mean = t.mean()
    t_c = t - t_mean
    denom = float((t_c ** 2).sum())
    if denom <= 0:
        # Degenerate positions
        resid = y0 - y0.mean()
    else:
        b = (t_c * y0).sum() / denom
        a = y0.mean() - b * t_mean
        y_hat = a + b * t
        resid = y0 - y_hat

    resid = resid.astype(np.float64)
    resid_mean = resid.mean()
    resid_std = resid.std(ddof=1)
    if resid_std <= 0:
        return 0.0, 0.0, 1.0

    z = (resid - resid_mean) / resid_std
    sk = float(skew(z, bias=False))
    kt = float(kurtosis(z, fisher=True, bias=False))
    _, pval = kstest(z, "norm")
    return sk, kt, float(pval)


def compute_noise_normality_stats(
    X_rot: np.ndarray,
    max_dims: int = 64,
    alpha: float = 0.01,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """
    Analyze normality of the 'noise' subspace: components j >= 1 of X_rot.
    Returns:
        med_abs_skew
        med_abs_kurt
        med_ks_p
        frac_ks_p_lt_alpha
    """
    # Drop first component
    Y = X_rot[:, 1:]
    n, d = Y.shape
    if d == 0 or n < 3:
        return 0.0, 0.0, 1.0, 0.0

    mu = Y.mean(axis=0)
    sd = Y.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (Y - mu) / sd

    sk_all = skew(Z, axis=0, bias=False)
    kt_all = kurtosis(Z, axis=0, fisher=True, bias=False)
    med_abs_skew = float(np.median(np.abs(sk_all)))
    med_abs_kurt = float(np.median(np.abs(kt_all)))

    rng = np.random.default_rng(seed)
    ndims = min(max_dims, d)
    dims = rng.choice(d, size=ndims, replace=False)

    pvals = []
    for j in dims:
        _, pval = kstest(Z[:, j], "norm")
        pvals.append(pval)
    pvals = np.asarray(pvals, dtype=np.float64)

    med_ks_p = float(np.median(pvals))
    frac_lt_alpha = float(np.mean(pvals < alpha))
    return med_abs_skew, med_abs_kurt, med_ks_p, frac_lt_alpha


def compute_scalar_position_predictability(
    p: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Using beta as positional slope, define a 1D projection s = X @ u, u=beta/||beta||.
    Fit y = log2(p+1) ~ a + b s, and return:
        R2_scalar, mae_baseline, mae_proj
    """
    t = p.astype(np.float64)
    y = np.log2(t + 1.0)

    norm_beta = np.linalg.norm(beta)
    if norm_beta <= 0:
        # No positional slope; projection carries no info
        mae_baseline = float(np.abs(y - y.mean()).mean())
        return 0.0, mae_baseline, mae_baseline

    u = beta / norm_beta
    s = X @ u  # (n,)

    # Fit y ~ a + b s
    s_mean = s.mean()
    s_c = s - s_mean
    denom = float((s_c ** 2).sum())
    if denom <= 0:
        mae_baseline = float(np.abs(y - y.mean()).mean())
        return 0.0, mae_baseline, mae_baseline

    b = (s_c * y).sum() / denom
    a = y.mean() - b * s_mean
    y_hat = a + b * s

    mae_baseline = float(np.abs(y - y.mean()).mean())
    mae_proj = float(np.abs(y - y_hat).mean())

    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        r2_scalar = 0.0
    else:
        r2_scalar = 1.0 - ss_res / ss_tot
    return float(r2_scalar), mae_baseline, mae_proj
