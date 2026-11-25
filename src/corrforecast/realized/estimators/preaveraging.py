"""Pre-averaging estimators.

* PRV — pre-averaged realized variance (Jacod, Li, Mykland, Podolskij & Vetter 2009)
* MRC — modulated realized covariance (Christensen, Kinnebrock & Podolskij 2010)

Both are building blocks of the Bollerslev, Meddahi & Nyawa (2019) estimator
(their eqs. 19 and 22).  With weight function g(x) = min(x, 1-x), window
k_n = ceil(theta * sqrt(n)) and pre-averaged returns
``Ybar_i = sum_{j=1}^{k_n-1} g(j/k_n) r_{i+j}``:

    MRC = n/(n-k_n+2) * 1/(psi2 k_n) * sum_i Ybar_i Ybar_i'
          - psi1/(2 n theta^2 psi2) * sum_i r_i r_i'           (bias correction)

PRV is the diagonal of MRC.  ``psi1, psi2`` are the finite-k_n versions
``psi1 = k_n sum_{i=1}^{k_n} (g(i/k_n) - g((i-1)/k_n))^2``,
``psi2 = (1/k_n) sum_{i=1}^{k_n-1} g(i/k_n)^2``.
"""

from __future__ import annotations

import numpy as np

from corrforecast.realized.estimators.base import (
    CovEstimator,
    DailyEstimate,
    nan_mask_cov,
    register,
)
from corrforecast.realized.sampling import (
    DayObservations,
    grid_returns,
    parse_interval,
    refresh_time_prices,
)


def g_triangle(x: np.ndarray) -> np.ndarray:
    return np.minimum(x, 1.0 - x)


def preaverage_weights(k_n: int) -> tuple[np.ndarray, float, float]:
    j = np.arange(1, k_n) / k_n
    w = g_triangle(j)
    grid = g_triangle(np.arange(0, k_n + 1) / k_n)
    psi1 = k_n * float(np.sum(np.diff(grid) ** 2))
    psi2 = float(np.sum(w**2)) / k_n
    return w, psi1, psi2


def window_length(n: int, theta: float) -> int:
    return max(2, int(np.ceil(theta * np.sqrt(n))))


def preaveraged_returns(r: np.ndarray, k_n: int) -> np.ndarray:
    """``Ybar_i = sum_{j=1}^{k_n-1} g(j/k_n) r_{i+j-1}`` for ``i = 0..n-k_n+1`` (rows)."""
    w, _, _ = preaverage_weights(k_n)
    n = r.shape[0]
    m = n - k_n + 2
    if m <= 0:
        raise ValueError(f"too few returns ({n}) for pre-averaging window {k_n}")
    # sliding-window weighted sum via cumulative sums of shifted copies (k_n is small, O(n k_n))
    out = np.zeros((m, r.shape[1]))
    for j, wj in enumerate(w):
        out += wj * r[j : j + m]
    return out


def mrc(r: np.ndarray, theta: float = 1.0 / 3.0, bias_correct: bool = True) -> np.ndarray:
    """Modulated realized covariance of synchronised returns ``r`` (n x p)."""
    n = r.shape[0]
    k_n = window_length(n, theta)
    _, psi1, psi2 = preaverage_weights(k_n)
    yb = preaveraged_returns(r, k_n)
    est = (n / (n - k_n + 2)) / (psi2 * k_n) * (yb.T @ yb)
    if bias_correct:
        theta_eff = k_n / np.sqrt(n)
        est -= psi1 / (2.0 * n * theta_eff**2 * psi2) * (r.T @ r)
    return est


def mrc_cross(
    r: np.ndarray, f: np.ndarray, theta: float = 1.0 / 3.0, bias_correct: bool = True
) -> np.ndarray:
    """Cross block ``MRC(r_i, f_k)`` (p x K) without forming the full (p+K)^2 matrix."""
    n = r.shape[0]
    k_n = window_length(n, theta)
    _, psi1, psi2 = preaverage_weights(k_n)
    yr, yf = preaveraged_returns(r, k_n), preaveraged_returns(f, k_n)
    est = (n / (n - k_n + 2)) / (psi2 * k_n) * (yr.T @ yf)
    if bias_correct:
        theta_eff = k_n / np.sqrt(n)
        est -= psi1 / (2.0 * n * theta_eff**2 * psi2) * (r.T @ f)
    return est


def prv(r: np.ndarray, theta: float = 1.0 / 3.0, bias_correct: bool = True) -> np.ndarray:
    """Pre-averaged realized variance of each column of ``r`` (vector of length p)."""
    n = r.shape[0]
    k_n = window_length(n, theta)
    _, psi1, psi2 = preaverage_weights(k_n)
    yb = preaveraged_returns(r, k_n)
    est = (n / (n - k_n + 2)) / (psi2 * k_n) * np.sum(yb**2, axis=0)
    if bias_correct:
        theta_eff = k_n / np.sqrt(n)
        est -= psi1 / (2.0 * n * theta_eff**2 * psi2) * np.sum(r**2, axis=0)
    return est


def synchronized_returns(
    obs: DayObservations, sync: str, min_obs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(r, valid)`` with ``r`` (n x p) synchronised returns, zero columns for invalid assets.

    ``sync`` is ``"refresh"`` (refresh time across all valid assets) or a grid interval like ``"1min"``.
    """
    valid = obs.n_obs() >= max(min_obs, 2)
    idx = np.flatnonzero(valid)
    if sync == "refresh":
        _, p = refresh_time_prices([obs.times[i] for i in idx], [obs.logp[i] for i in idx])
        r = np.zeros((max(len(p) - 1, 0), obs.n_assets))
        if len(p) > 1:
            r[:, idx] = np.diff(p, axis=0)
    else:
        r = grid_returns(obs, parse_interval(sync))
        r = np.where(valid[None, :], r, 0.0)
    return r, valid


@register
class MRCEstimator(CovEstimator):
    """Stand-alone MRC estimator (a common noise-robust benchmark; not PSD after bias correction)."""

    name = "mrc"

    def __init__(
        self,
        sync: str = "refresh",
        theta: float = 1.0 / 3.0,
        bias_correct: bool = True,
        min_obs: int = 10,
    ):
        self.sync, self.theta, self.bias_correct, self.min_obs = (
            sync,
            float(theta),
            bool(bias_correct),
            int(min_obs),
        )

    def estimate(self, obs: DayObservations) -> DailyEstimate:
        r, valid = synchronized_returns(obs, self.sync, self.min_obs)
        cov = mrc(r, self.theta, self.bias_correct)
        return DailyEstimate(cov=nan_mask_cov(cov, valid), diagnostics={"n_returns": r.shape[0]})
