"""Factor-based noise-robust realized covariance of Bollerslev, Meddahi & Nyawa (2019).

Setting: efficient prices follow a strict factor model ``dX* = b dF + dE`` and the
observed prices carry microstructure noise that may itself have a factor
structure ``u_it = c_i g_t + eta_it`` (which breaks estimators that assume
cross-sectionally independent noise).  Algorithm (their eq. 15), on returns
``r`` (n x p) synchronised by refresh time:

1. Pilot: multivariate realized kernel ``MRker`` -> eigenvectors ``W`` (p x K) of the
   K largest eigenvalues.
2. Rotated factors: ``f_t = (1/p) W' r_t``.
3. Factor variances ``PRV(f_k)`` (pre-averaged RV) and loadings
   ``b_ik = MRC(r_i, f_k) / PRV(f_k)`` (modulated realized covariance).
4. Residuals ``e_it = r_it - sum_k b_ik f_kt`` and idiosyncratic variances ``PRV(e_i)``.
5. ``Sigma = b diag(PRV(f)) b' + diag(PRV(e))``, i.e.
   ``Sigma_ij = sum_k MRC(r_i,f_k) MRC(r_j,f_k) / PRV(f_k)``,
   ``Sigma_ii = sum_k MRC(r_i,f_k)^2 / PRV(f_k) + PRV(e_i)``.

The estimate is positive definite by construction.  K is fixed or chosen per
day by the Bai & Ng (2002) IC_p1 criterion on sparsely sampled (default 5-min)
returns, as in BMN's application (average K ≈ 3.3 for 384 S&P 500 stocks).
"""

from __future__ import annotations

import numpy as np

from corrforecast.realized.estimators.base import (
    CovEstimator,
    DailyEstimate,
    nan_mask_cov,
    register,
)
from corrforecast.realized.estimators.kernel import realized_kernel
from corrforecast.realized.estimators.preaveraging import mrc, mrc_cross, prv, synchronized_returns
from corrforecast.realized.sampling import DayObservations, grid_returns, parse_interval


def bai_ng_n_factors(r: np.ndarray, k_max: int = 10) -> int:
    """Bai & Ng (2002) IC_p1 on returns ``r`` (T x p) via PCA of ``r'r``."""
    T, p = r.shape
    k_max = int(min(k_max, p - 1, T - 1))
    if k_max < 1:
        return 1
    eig = np.sort(np.linalg.eigvalsh(r.T @ r))[::-1]
    total = eig.sum()
    pen = (p + T) / (p * T) * np.log(p * T / (p + T))
    ics = []
    for k in range(1, k_max + 1):
        v = max((total - eig[:k].sum()) / (p * T), 1e-300)
        ics.append(np.log(v) + k * pen)
    return int(np.argmin(ics) + 1)


def bmn_covariance(
    r: np.ndarray,
    n_factors: int,
    theta: float = 1.0 / 3.0,
    pilot: str = "mrker",
    bandwidth: float | None = None,
    floor: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """BMN estimator on synchronised returns ``r`` (n x p). Returns ``(Sigma, diagnostics)``."""
    n, p = r.shape
    K = int(max(1, min(n_factors, p)))
    if pilot == "mrker":
        pil = realized_kernel(r, bandwidth=bandwidth)
    elif pilot == "mrc":
        pil = mrc(r, theta, bias_correct=False)  # PSD without the bias correction
    elif pilot == "rcov":
        pil = r.T @ r
    else:
        raise ValueError(f"unknown pilot {pilot!r}")
    w, V = np.linalg.eigh(0.5 * (pil + pil.T))
    W = V[:, np.argsort(w)[::-1][:K]]
    f = r @ W / p  # (n, K) rotated factors f_t = W' r_t / p
    pf = prv(f, theta)
    scale = max(float(np.mean(prv(r, theta, bias_correct=False))), floor)
    pf = np.maximum(pf, floor * scale)
    cross = mrc_cross(r, f, theta)  # (p, K)
    b = cross / pf[None, :]
    e = r - f @ b.T
    pe = np.maximum(prv(e, theta), floor * scale)
    sigma = (b * pf[None, :]) @ b.T + np.diag(pe)
    return sigma, {"n_factors": K, "n_returns": n, "factor_prv": pf}


@register
class BollerslevMeddahiNyawa(CovEstimator):
    """Parameters
    ----------
    sync:
        ``"refresh"`` (refresh time across all valid assets, as in BMN) or a grid like ``"1min"``.
    n_factors:
        Integer, or ``"auto"`` for the per-day Bai–Ng choice on ``ic_sampling`` returns.
    k_max:
        Largest K considered by the information criterion.
    theta:
        Pre-averaging tuning constant (``k_n = ceil(theta sqrt(n))``).
    pilot:
        ``"mrker"`` (BMN), ``"mrc"`` or ``"rcov"``.
    min_obs:
        Assets with fewer raw observations are dropped for the day (BMN drop stocks with < 195 trades).
    """

    name = "bmn_factor"

    def __init__(
        self,
        sync: str = "refresh",
        n_factors: int | str = "auto",
        k_max: int = 10,
        theta: float = 1.0 / 3.0,
        pilot: str = "mrker",
        bandwidth: float | None = None,
        ic_sampling: str = "5min",
        min_obs: int = 10,
    ):
        self.sync, self.n_factors, self.k_max, self.theta = (
            sync,
            n_factors,
            int(k_max),
            float(theta),
        )
        self.pilot, self.bandwidth, self.ic_sampling, self.min_obs = (
            pilot,
            bandwidth,
            ic_sampling,
            int(min_obs),
        )

    def estimate(self, obs: DayObservations) -> DailyEstimate:
        r, valid = synchronized_returns(obs, self.sync, self.min_obs)
        N = obs.n_assets
        idx = np.flatnonzero(valid)
        if len(idx) < 2 or r.shape[0] < 8:
            return DailyEstimate(cov=np.full((N, N), np.nan), diagnostics={"n_returns": r.shape[0]})
        if self.n_factors == "auto":
            r_ic = grid_returns(obs, parse_interval(self.ic_sampling))[:, idx]
            K = bai_ng_n_factors(np.nan_to_num(r_ic), self.k_max)
        else:
            K = int(self.n_factors)
        sig, diag = bmn_covariance(r[:, idx], K, self.theta, self.pilot, self.bandwidth)
        cov = np.full((N, N), np.nan)
        cov[np.ix_(idx, idx)] = sig
        return DailyEstimate(cov=nan_mask_cov(cov, valid), diagnostics=diag)
