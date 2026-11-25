"""Helpers shared by the applications."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from corrforecast.features.panel import MonthlyMatrices


def prediction_matrices(
    preds: pd.DataFrame, model: str, n_assets: int
) -> dict[pd.Timestamp, np.ndarray]:
    """``{origin month t: predicted correlation matrix for t+1}`` (unit diagonal, NaN if missing)."""
    out = {}
    for month, g in preds.groupby("month"):
        m = np.full((n_assets, n_assets), np.nan)
        i, j, v = g["i"].to_numpy(), g["j"].to_numpy(), g[model].to_numpy(dtype=float)
        m[i, j] = v
        m[j, i] = v
        np.fill_diagonal(m, 1.0)
        out[pd.Timestamp(month)] = m
    return out


def panel_matrix(
    monthly: pd.DataFrame, value: str, months: pd.DatetimeIndex, assets: np.ndarray
) -> pd.DataFrame:
    """(month x asset) matrix of a monthly-panel column."""
    return monthly.pivot(index="month", columns="asset_id", values=value).reindex(
        index=months, columns=assets
    )


def har_variance_forecasts(
    mm: MonthlyMatrices, test_years: list[int], window_years: int = 5
) -> dict[pd.Timestamp, np.ndarray]:
    """Monthly HAR forecasts of next-month realized variance (BLT §5.3: HAR for all volatility forecasts).

    Pooled OLS ``RV_{t+1} = a + b_d RV^d_t + b_w RV^w_t + b_m RV^m_t`` re-estimated each test year on
    target years ``Y-window..Y-1``. Returns ``{origin month t: RV-hat_{t+1}}`` (floored at a small positive value).
    """
    months = mm.months
    out: dict[pd.Timestamp, np.ndarray] = {}
    X = np.stack([mm.rv_d, mm.rv_w, mm.rv_m], axis=-1)  # (T, N, 3)
    for Y in test_years:
        rows_x, rows_y = [], []
        for t in range(len(months) - 1):
            ty = months[t + 1].year
            if Y - window_years <= ty <= Y - 1:
                rows_x.append(X[t])
                rows_y.append(mm.rv_m[t + 1])
        if not rows_x:
            continue
        Xa = np.concatenate(rows_x)
        ya = np.concatenate(rows_y)
        ok = np.isfinite(Xa).all(1) & np.isfinite(ya)
        A = np.column_stack([np.ones(ok.sum()), Xa[ok]])
        beta, *_ = np.linalg.lstsq(A, ya[ok], rcond=None)
        for t in range(len(months) - 1):
            if months[t + 1].year == Y:
                f = beta[0] + X[t] @ beta[1:]
                floor = 0.05 * np.nanmedian(mm.rv_m[t])
                out[months[t]] = np.maximum(np.nan_to_num(f, nan=floor), floor)
    return out


def capped_value_weights(caps: np.ndarray, cap: float) -> np.ndarray:
    """Value weights with no single weight above ``cap`` (excess redistributed proportionally)."""
    w = caps / caps.sum()
    if cap * len(w) < 1:  # infeasible -> equal weights
        return np.full(len(w), 1.0 / len(w))
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        w[free] += excess * w[free] / w[free].sum()
    return w


def utility_gain(
    r_base: np.ndarray, r_new: np.ndarray, gamma: float, periods_per_year: int = 12
) -> float:
    """Annualised fee Delta solving sum U(r_base) = sum U(r_new - Delta) with quadratic utility (BLT eqs. 13-14)."""

    def U(r: np.ndarray) -> float:
        return float(np.sum((1 + r) - gamma / (2 * (1 + gamma)) * (1 + r) ** 2))

    target = U(r_base)
    # U is increasing in r for r < 1/gamma, so bracket the monthly fee tightly around zero
    for width in (0.05, 0.1, 0.2):
        try:
            d = brentq(lambda x: U(r_new - x) - target, -width, width)
            break
        except ValueError:
            continue
    else:
        return float("nan")
    return float(d * periods_per_year)


def newey_west_tstat(x: np.ndarray, lags: int = 0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(s / n))
