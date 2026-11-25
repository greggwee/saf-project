"""Matrix-level feature transforms (BLT Appendix A)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def exp_weights(center_of_mass: float, window: int) -> np.ndarray:
    """Normalised weights ``w_k ∝ exp(-k λ)``, k = 1..window, λ = log(1 + 1/CoM) (BLT eq. A.3).

    ``w[0]`` applies to the most recent day.
    """
    lam = np.log1p(1.0 / center_of_mass)
    w = np.exp(-lam * np.arange(1, window + 1))
    return w / w.sum()


def rank_to_unit(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank transform into [-1, 1] (Gu, Kelly & Xiu 2020); NaN -> 0 (median)."""
    s = pd.Series(x)
    r = s.rank(method="average")
    n = r.notna().sum()
    out = 2.0 * (r - 1.0) / max(n - 1, 1) - 1.0
    return out.fillna(0.0).to_numpy()


def loading_matrix(
    chars: pd.DataFrame, no_rank: tuple[str, ...] = ("beta",), add_intercept: bool = False
) -> np.ndarray:
    """Factor loading matrix L (N x K) from characteristics (rows aligned to the universe)."""
    cols = []
    for c in chars.columns:
        x = chars[c].to_numpy(dtype=float)
        if c in no_rank:
            cols.append(
                np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0)
            )
        else:
            cols.append(rank_to_unit(x))
    L = np.column_stack(cols) if cols else np.empty((len(chars), 0))
    if add_intercept:
        L = np.column_stack([np.ones(len(chars)), L])
    return L


def factor_projection(cov: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Factor-driven covariance (BLT eq. A.2).

    ``RCov^F = P RCov P + Diag(RCov - P RCov P)`` with ``P = L (L'L)^+ L'``; the
    diagonal equals that of ``RCov``.  Works on ``(N, N)`` or stacks ``(T, N, N)``.
    """
    P = L @ np.linalg.pinv(L.T @ L) @ L.T
    sys_ = P @ cov @ P
    out = sys_.copy()
    idx = np.arange(cov.shape[-1])
    out[..., idx, idx] = cov[..., idx, idx]
    return out


def sector_average(corr: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    """Within-sector average off-diagonal correlation; zero for cross-sector pairs (ExpScRC)."""
    out = np.zeros_like(corr)
    for s in np.unique(sectors):
        idx = np.flatnonzero(sectors == s)
        if len(idx) < 2:
            continue
        block = corr[np.ix_(idx, idx)]
        off = block[~np.eye(len(idx), dtype=bool)]
        avg = np.nanmean(off) if np.isfinite(off).any() else np.nan
        out[np.ix_(idx, idx)] = avg
    np.fill_diagonal(out, 1.0)
    return out
