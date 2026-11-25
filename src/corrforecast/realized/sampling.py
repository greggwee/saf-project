"""Intraday sampling and synchronisation schemes.

* :func:`previous_tick` — calendar-time sampling on a fixed grid (BLT use a
  15-minute grid of midquotes).
* :func:`refresh_times` — the refresh-time scheme of Barndorff-Nielsen, Hansen,
  Lunde & Shephard (2011), which is the Generalized Sampling Time used by
  Aït-Sahalia, Fan & Xiu (2010) and the synchronisation used by Bollerslev,
  Meddahi & Nyawa (2019).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DayObservations:
    """All intraday observations of one trading day, aligned to the universe order.

    ``times[i]`` are seconds since the session open (sorted, unique) and
    ``logp[i]`` the corresponding log prices of asset ``assets[i]``.
    Assets without data have empty arrays.
    """

    date: pd.Timestamp
    assets: np.ndarray
    times: list[np.ndarray]
    logp: list[np.ndarray]
    session_seconds: float

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def n_obs(self) -> np.ndarray:
        return np.array([len(t) for t in self.times])

    @classmethod
    def from_frame(
        cls,
        date: pd.Timestamp,
        frame: pd.DataFrame,
        assets: np.ndarray,
        session_open: str = "09:30",
        session_close: str = "16:00",
    ) -> DayObservations:
        open_td = pd.Timedelta(session_open + ":00")
        sess = (pd.Timedelta(session_close + ":00") - open_td).total_seconds()
        times: list[np.ndarray] = []
        logp: list[np.ndarray] = []
        empty = np.empty(0)
        if len(frame):
            secs = (
                ((frame["timestamp"] - frame["timestamp"].dt.normalize()) - open_td)
                .dt.total_seconds()
                .to_numpy()
            )
            ids = frame["asset_id"].to_numpy()
            lp = np.log(frame["price"].to_numpy(dtype=float))
            order = np.lexsort((secs, ids))
            ids, secs, lp = ids[order], secs[order], lp[order]
            uniq, start = np.unique(ids, return_index=True)
            bounds = dict(
                zip(uniq.tolist(), zip(start, [*start[1:], len(ids)], strict=True), strict=True)
            )
        else:
            bounds = {}
        for a in assets:
            b = bounds.get(a.item() if hasattr(a, "item") else a)
            if b is None:
                times.append(empty)
                logp.append(empty)
                continue
            t, p = secs[b[0] : b[1]], lp[b[0] : b[1]]
            # keep the last observation of duplicated timestamps
            keep = np.append(t[1:] != t[:-1], True)
            times.append(t[keep])
            logp.append(p[keep])
        return cls(
            date=pd.Timestamp(date),
            assets=np.asarray(assets),
            times=times,
            logp=logp,
            session_seconds=sess,
        )


def parse_interval(spec: str | float | int) -> float:
    """``"15min"`` / ``"5s"`` / ``900`` -> seconds."""
    if isinstance(spec, (int, float)):
        return float(spec)
    return pd.Timedelta(spec).total_seconds()


def previous_tick(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Last observation at or before each grid point (first observation before the first tick)."""
    if len(times) == 0:
        return np.full(len(grid), np.nan)
    idx = np.searchsorted(times, grid, side="right") - 1
    return values[np.clip(idx, 0, len(values) - 1)]


def grid_returns(obs: DayObservations, interval_seconds: float) -> np.ndarray:
    """Calendar-time log returns, shape ``(n_intervals, n_assets)``; NaN columns for assets without data."""
    n_int = round(obs.session_seconds / interval_seconds)
    grid = np.linspace(0.0, obs.session_seconds, n_int + 1)
    prices = np.column_stack(
        [previous_tick(t, p, grid) for t, p in zip(obs.times, obs.logp, strict=True)]
    )
    return np.diff(prices, axis=0)


def refresh_times(times: list[np.ndarray]) -> np.ndarray:
    """Refresh times: tau_1 = max_i t_{i,1}; tau_{j+1} = max_i (first t_{i,k} > tau_j).

    Every asset trades at least once in each interval (tau_{j-1}, tau_j].
    Returns an empty array if any asset has no observation.
    """
    if any(len(t) == 0 for t in times):
        return np.empty(0)
    if len(times) == 1:
        return times[0].copy()
    if len(times) <= 8:
        return _refresh_times_pointer(times)
    out = []
    tau = max(t[0] for t in times)
    while True:
        out.append(tau)
        nxt = -np.inf
        for t in times:
            k = np.searchsorted(t, tau, side="right")
            if k >= len(t):
                return np.asarray(out)
            nxt = max(nxt, t[k])
        tau = nxt


def _refresh_times_pointer(times: list[np.ndarray]) -> np.ndarray:
    """Few-asset refresh times: precompute the successor map on the union of times, then chase it."""
    u = np.unique(np.concatenate(times))
    succ = np.full(len(u), -np.inf)
    for t in times:
        k = np.searchsorted(t, u, side="right")
        nx = np.where(k < len(t), t[np.minimum(k, len(t) - 1)], np.inf)
        succ = np.maximum(succ, nx)
    # index of successor in u; len(u) marks "no successor"
    nxt_idx = np.where(np.isfinite(succ), np.searchsorted(u, succ), len(u)).tolist()
    i = int(np.searchsorted(u, max(t[0] for t in times)))
    path = []
    n_u = len(u)
    while i < n_u:
        path.append(i)
        i = nxt_idx[i]
    return u[path]


def refresh_time_prices(
    times: list[np.ndarray], logp: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Synchronised log prices at the refresh times: ``(tau, prices[n_tau, n_assets])``."""
    tau = refresh_times(times)
    if len(tau) == 0:
        return tau, np.empty((0, len(times)))
    prices = np.column_stack([previous_tick(t, p, tau) for t, p in zip(times, logp, strict=True)])
    return tau, prices
