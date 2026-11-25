"""Matrix utilities: covariance <-> correlation, PSD projection."""

from __future__ import annotations

import numpy as np


def cov_to_corr(cov: np.ndarray, clip: bool = True) -> np.ndarray:
    """Correlation implied by a covariance matrix (works on stacks ``(..., N, N)``).

    Non-positive variances yield NaN rows/columns.  With ``clip`` the result is
    bounded to [-1, 1] (BLT endnote 8).
    """
    d = np.diagonal(cov, axis1=-2, axis2=-1)
    sd = np.sqrt(np.where(d > 0, d, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / (sd[..., :, None] * sd[..., None, :])
    if clip:
        corr = np.clip(corr, -1.0, 1.0)
    return corr


def nearest_correlation_psd(corr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix with unit diagonal onto PSD correlation matrices.

    Eigenvalue clipping at ``eps`` followed by re-normalisation to unit diagonal
    (the simple "convexity correction" used before GMV optimisation).  NaNs are
    treated as zero correlation.
    """
    c = np.nan_to_num(0.5 * (corr + corr.T), nan=0.0)
    np.fill_diagonal(c, 1.0)
    w, v = np.linalg.eigh(c)
    if w.min() >= eps:
        return c
    c = (v * np.maximum(w, eps)) @ v.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return c


def corr_to_cov(corr: np.ndarray, variances: np.ndarray) -> np.ndarray:
    sd = np.sqrt(variances)
    return corr * np.outer(sd, sd)
