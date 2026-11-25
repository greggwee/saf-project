"""Risk targeting (BLT §5.3) and Global Minimum Variance portfolios (BLT §5.4).

Covariance forecasts: ``RCov-hat_{t+1} = sqrt(RV-hat) RC-hat sqrt(RV-hat)`` (eq. 10) where the
variances always come from the same HAR model, so differences across models isolate the
correlation forecasts.  Realized covariance is the month-(t+1) RCov.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corrforecast.applications.common import (
    har_variance_forecasts,
    newey_west_tstat,
    panel_matrix,
    prediction_matrices,
    utility_gain,
)
from corrforecast.config import ApplicationsConfig
from corrforecast.features.panel import MonthlyMatrices
from corrforecast.realized.psd import corr_to_cov, nearest_correlation_psd


def long_short_weights(x: np.ndarray, n_q: int) -> np.ndarray:
    """Equal-weighted long top / short bottom quantile of characteristic ``x`` (zero-cost)."""
    w = np.zeros(len(x))
    ok = np.isfinite(x)
    if ok.sum() < 2 * n_q:
        n_q = int(max(2, ok.sum() // 2))
    lab = np.full(len(x), -1)
    lab[ok] = pd.qcut(pd.Series(x[ok]).rank(method="first"), n_q, labels=False).to_numpy()
    hi, lo = lab == n_q - 1, lab == 0
    w[hi], w[lo] = 1.0 / hi.sum(), -1.0 / lo.sum()
    return w


def _forecast_covs(preds, mm, acfg) -> tuple[dict, dict]:
    years = sorted(pd.to_datetime(preds["target_month"]).dt.year.unique().tolist())
    rv_hat = har_variance_forecasts(mm, years)
    pm = {m: prediction_matrices(preds, m, len(mm.assets)) for m in acfg.models}
    return rv_hat, pm


def run_risk_targeting(
    preds: pd.DataFrame,
    mm: MonthlyMatrices,
    monthly: pd.DataFrame,
    acfg: ApplicationsConfig,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    rv_hat, pm = _forecast_covs(preds, mm, acfg)
    chars = [c for c in acfg.long_short_characteristics if c in monthly.columns]
    cm = {c: panel_matrix(monthly, c, mm.months, mm.assets) for c in chars}
    rows = []
    for t, rv in rv_hat.items():
        t1 = mm.index(t) + 1
        if t1 >= len(mm.months):
            continue
        realized = mm.cov_m[t1]
        for c in chars:
            w = long_short_weights(cm[c].loc[t].to_numpy(dtype=float), acfg.long_short_quantiles)
            den = w @ realized @ w
            for m in acfg.models:
                if t not in pm[m]:
                    continue
                fc = corr_to_cov(np.nan_to_num(pm[m][t], nan=0.0), rv)
                rows.append(
                    {"month": mm.months[t1], "strategy": c, "model": m, "ratio": (w @ fc @ w) / den}
                )
    ts = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    summ = []
    for (c, m), g in ts.groupby(["strategy", "model"]):
        r = g["ratio"].to_numpy()
        boots = [rng.choice(r, len(r)).mean() for _ in range(n_boot)]
        summ.append(
            {
                "strategy": c,
                "model": m,
                "AvgRatio": r.mean(),
                "ci_lo": np.percentile(boots, 2.5),
                "ci_hi": np.percentile(boots, 97.5),
                "n": len(r),
            }
        )
    return {"risk_targeting_timeseries": ts, "risk_targeting": pd.DataFrame(summ)}


def gmv_weights(cov: np.ndarray, beta_neutral_m: np.ndarray | None = None) -> np.ndarray:
    """GMV weights (eq. 12); with ``beta_neutral_m`` adds the constraint ``w' Cov m = 0`` (eq. 15)."""
    ones = np.ones(cov.shape[0])
    A = ones[:, None] if beta_neutral_m is None else np.column_stack([ones, cov @ beta_neutral_m])
    b = np.array([1.0]) if beta_neutral_m is None else np.array([1.0, 0.0])
    SiA = np.linalg.solve(cov, A)
    return SiA @ np.linalg.solve(A.T @ SiA, b)


def run_gmv(
    preds: pd.DataFrame,
    mm: MonthlyMatrices,
    monthly: pd.DataFrame,
    market: pd.DataFrame | None,
    acfg: ApplicationsConfig,
) -> dict[str, pd.DataFrame]:
    rv_hat, pm = _forecast_covs(preds, mm, acfg)
    ret_m = panel_matrix(monthly, "ret", mm.months, mm.assets)
    cap_m = panel_matrix(monthly, "mktcap", mm.months, mm.assets)
    rf = market.set_index("month")["rf"] if market is not None else pd.Series(dtype=float)
    rows = []
    for t, rv in rv_hat.items():
        t1 = mm.index(t) + 1
        if t1 >= len(mm.months):
            continue
        r_next = np.nan_to_num(ret_m.iloc[t1].to_numpy(dtype=float))
        realized = mm.cov_m[t1]
        caps = np.nan_to_num(cap_m.loc[t].to_numpy(dtype=float))
        mvec = caps / caps.sum() if caps.sum() > 0 else np.full(len(caps), 1 / len(caps))
        for m in acfg.models:
            if t not in pm[m]:
                continue
            fc = corr_to_cov(nearest_correlation_psd(pm[m][t]), rv)  # convexity correction
            for kind, mv in (("GMV", None), ("BetaNeutralGMV", mvec)):
                w = gmv_weights(fc, mv)
                rows.append(
                    {
                        "month": mm.months[t1],
                        "model": m,
                        "portfolio": kind,
                        "ret": float(w @ r_next),
                        "realized_vol": float(np.sqrt(max(w @ realized @ w, 0.0))),
                        "realized_beta": float((w @ realized @ mvec) / (mvec @ realized @ mvec)),
                        "rf": float(rf.get(mm.months[t1], 0.0)),
                    }
                )
    ts = pd.DataFrame(rows)
    summ = []
    base = acfg.models[0]
    for (kind, m), g in ts.groupby(["portfolio", "model"]):
        ex = g["ret"] - g["rf"]
        row = {
            "portfolio": kind,
            "model": m,
            "mean_ret_ann_%": 1200 * g["ret"].mean(),
            "sd_ann_%": 100 * np.sqrt(12) * g["ret"].std(),
            "sharpe": np.sqrt(12) * ex.mean() / ex.std(),
            "avg_realized_vol_%": 100 * g["realized_vol"].mean(),
            "avg_beta": g["realized_beta"].mean(),
            "beta_t": newey_west_tstat(g["realized_beta"].to_numpy(), 6),
        }
        if m != base:
            gb = ts[(ts["portfolio"] == kind) & (ts["model"] == base)].set_index("month")["ret"]
            gm = g.set_index("month")["ret"]
            idx = gb.index.intersection(gm.index)
            for gam in acfg.risk_aversions:
                row[f"utility_gain_g{gam:g}_%"] = 100 * utility_gain(
                    gb[idx].to_numpy(), gm[idx].to_numpy(), gam
                )
        summ.append(row)
    return {"gmv_timeseries": ts, "gmv": pd.DataFrame(summ)}
