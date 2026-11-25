"""Multivariate realized kernel (Barndorff-Nielsen, Hansen, Lunde & Shephard 2011).

Used as the pilot estimator in the BMN (2019) procedure ("MRker", their eq. 21)
to obtain the eigenvectors that rotate returns into latent factors.  The Parzen
kernel guarantees a positive semi-definite estimate.

    K(X) = sum_{h=-H}^{H} k(h / (H + 1)) Gamma_h,   Gamma_h = sum_j x_j x_{j-h}',  Gamma_{-h} = Gamma_h'

with end-point jittering (average of the first/last ``m`` prices) and the
BNHLS rule-of-thumb bandwidth ``H = c* xi^{4/5} n^{3/5}``, ``c* = 3.5134``.
"""

from __future__ import annotations

import numpy as np

from corrforecast.realized.estimators.base import (
    CovEstimator,
    DailyEstimate,
    nan_mask_cov,
    register,
)
from corrforecast.realized.estimators.preaveraging import synchronized_returns
from corrforecast.realized.sampling import DayObservations

C_STAR_PARZEN = 3.5134


def parzen(x: np.ndarray | float) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype=float))
    return np.where(x <= 0.5, 1 - 6 * x**2 + 6 * x**3, np.where(x <= 1.0, 2 * (1 - x) ** 3, 0.0))


def jitter_returns(r: np.ndarray, m: int = 2) -> np.ndarray:
    """Replace the first/last prices by the average of the first/last ``m`` prices (BNHLS 2011)."""
    if m <= 1 or r.shape[0] <= 2 * m:
        return r
    p = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(r, axis=0)])
    start = p[:m].mean(axis=0)
    end = p[-m:].mean(axis=0)
    inner = p[m - 1 : -(m - 1)] if m > 1 else p
    inner = inner.copy()
    inner[0], inner[-1] = start, end
    return np.diff(inner, axis=0)


def rule_of_thumb_bandwidth(r: np.ndarray, sparse_step: int | None = None) -> float:
    """BNHLS bandwidth using the cross-sectional mean of xi_i = omega_i / sqrt(IV_i)."""
    n = r.shape[0]
    omega2 = np.sum(r**2, axis=0) / (2 * n)
    q = sparse_step or max(1, n // 20)  # ~20 sparse returns per day
    p = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(r, axis=0)])
    iv = np.mean([np.sum(np.diff(p[o::q], axis=0) ** 2, axis=0) for o in range(q)], axis=0)
    ok = iv > 0
    if not ok.any():
        return 1.0
    xi = np.sqrt(np.mean(omega2[ok] / iv[ok]))
    return float(C_STAR_PARZEN * xi**0.8 * n**0.6)


def realized_kernel(r: np.ndarray, bandwidth: float | None = None, jitter: int = 2) -> np.ndarray:
    x = jitter_returns(r, jitter)
    H = rule_of_thumb_bandwidth(r) if bandwidth is None else float(bandwidth)
    out = x.T @ x
    for h in range(1, int(np.ceil(H)) + 1):
        w = float(parzen(h / (H + 1)))
        if w == 0.0:
            break
        g = x[h:].T @ x[:-h]
        out += w * (g + g.T)
    return out


@register
class RealizedKernelEstimator(CovEstimator):
    name = "realized_kernel"

    def __init__(
        self,
        sync: str = "refresh",
        bandwidth: float | None = None,
        jitter: int = 2,
        min_obs: int = 10,
    ):
        self.sync, self.bandwidth, self.jitter, self.min_obs = (
            sync,
            bandwidth,
            int(jitter),
            int(min_obs),
        )

    def estimate(self, obs: DayObservations) -> DailyEstimate:
        r, valid = synchronized_returns(obs, self.sync, self.min_obs)
        cov = realized_kernel(r, self.bandwidth, self.jitter)
        return DailyEstimate(cov=nan_mask_cov(cov, valid), diagnostics={"n_returns": r.shape[0]})
