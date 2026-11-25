"""Plain realized covariance and negative semicovariance on a calendar-time grid.

This is the BLT baseline: previous-tick midquotes on a 15-minute grid (26
intraday returns; with the overnight return n = 27, BLT eq. (1) and (A.1)).
With finer ``sampling`` it becomes the naive high-frequency estimator that is
biased by microstructure noise and the Epps effect — the motivation for the
denoising estimators.
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


def realized_cov_and_semicov(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``RCov = r'r`` and ``RCov^- = (r^-)'(r^-)`` with ``r^- = min(r, 0)`` (BLT eq. A.1)."""
    rn = np.minimum(r, 0.0)
    return r.T @ r, rn.T @ rn


@register
class RealizedCov(CovEstimator):
    """Parameters
    ----------
    sampling:
        Grid interval (``"15min"``, ``"5min"``, ``"1min"``, ``"30s"``) or ``"refresh"`` for
        refresh-time sampling across the whole universe.
    min_obs:
        Assets with fewer raw observations that day are set to NaN.
    """

    name = "realized_cov"
    provides_semicov = True

    def __init__(self, sampling: str = "15min", min_obs: int = 1):
        self.sampling = sampling
        self.min_obs = int(min_obs)

    def estimate(self, obs: DayObservations) -> DailyEstimate:
        valid = obs.n_obs() >= max(self.min_obs, 1)
        if self.sampling == "refresh":
            idx = np.flatnonzero(valid)
            _, p = refresh_time_prices([obs.times[i] for i in idx], [obs.logp[i] for i in idx])
            r = np.full((max(len(p) - 1, 0), obs.n_assets), np.nan)
            r[:, idx] = np.diff(p, axis=0)
        else:
            r = grid_returns(obs, parse_interval(self.sampling))
        r = np.where(valid[None, :], r, 0.0)
        cov, semi = realized_cov_and_semicov(r)
        return DailyEstimate(
            cov=nan_mask_cov(cov, valid),
            semicov_neg=nan_mask_cov(semi, valid),
            diagnostics={"n_returns": r.shape[0]},
        )
