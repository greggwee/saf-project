"""Correlation-augmented pairs trading (BLT §5.1, following Chen, Chen, Chen & Li 2019).

For stock i at month-end t:

* historical correlation ``H = mean(RC_{t-12..t-1})``; partners = 20 stocks with highest ``H_i.``;
  ``RC^h_i = mean_j H_ij`` over partners;
* ``RetDiff_i = beta_i (PRet_i - rf) - (Ret_i - rf)`` (eq. 7) with ``PRet`` the equal-weighted
  partner return in month t and ``beta_i`` from daily returns over months t-12..t-1;
* ``dRC^theta_i = mean_j RC-hat_{ij,t+1} - RC^h_i``.

Unconditional strategy: RetDiff quintiles across all stocks.  Conditional strategy
(model theta): keep the top ``dRC^theta`` quintile, then RetDiff quintiles within it.
Long the top RetDiff quintile, short the bottom, hold over month t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corrforecast.applications.common import (
    capped_value_weights,
    newey_west_tstat,
    panel_matrix,
    prediction_matrices,
)
from corrforecast.config import ApplicationsConfig
from corrforecast.features.panel import MonthlyMatrices


def _quantile_labels(x: np.ndarray, q: int) -> np.ndarray:
    lab = np.full(len(x), -1)
    ok = np.isfinite(x)
    if ok.sum() >= q:
        lab[ok] = pd.qcut(pd.Series(x[ok]).rank(method="first"), q, labels=False).to_numpy()
    return lab


def pair_signals(
    mm: MonthlyMatrices,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    rf: pd.Series,
    acfg: ApplicationsConfig,
    months: list[pd.Timestamp],
) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    """Partners, historical correlation, RetDiff for each origin month."""
    assets = mm.assets
    N = len(assets)
    ret_m = panel_matrix(monthly, "ret", mm.months, assets)
    dret = daily.pivot(index="date", columns="asset_id", values="ret").reindex(columns=assets)
    dper = dret.index.to_period("M")
    out = {}
    L = acfg.pairs_lookback_months
    k = min(acfg.pairs_n_partners, N - 1)
    for t in months:
        ti = mm.index(t)
        if ti < L:
            continue
        H = np.nanmean(mm.rc_m[ti - L : ti], axis=0)
        np.fill_diagonal(H, -np.inf)
        partners = np.argsort(-np.nan_to_num(H, nan=-np.inf), axis=1)[:, :k]
        rch = np.take_along_axis(H, partners, axis=1).mean(1)
        r_t = ret_m.loc[t].to_numpy(dtype=float)
        pret = np.nanmean(r_t[partners], axis=1)
        lo = (t - pd.offsets.MonthEnd(L)).to_period("M")
        hi = (t - pd.offsets.MonthEnd(1)).to_period("M")
        win = dret[(dper >= lo) & (dper <= hi)].to_numpy(dtype=float)
        beta = np.full(N, np.nan)
        for i in range(N):
            x = np.nanmean(win[:, partners[i]], axis=1)
            y = win[:, i]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 20 and np.var(x[ok]) > 0:
                beta[i] = np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok], ddof=1)
        rft = float(rf.get(t, 0.0))
        out[t] = {
            "partners": partners,
            "rc_hist": rch,
            "retdiff": beta * (pret - rft) - (r_t - rft),
            "beta": beta,
        }
    return out


def run_pairs_trading(
    preds: pd.DataFrame,
    mm: MonthlyMatrices,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    market: pd.DataFrame | None,
    acfg: ApplicationsConfig,
) -> dict[str, pd.DataFrame]:
    assets = mm.assets
    rf = market.set_index("month")["rf"] if market is not None else pd.Series(dtype=float)
    months = sorted(pd.to_datetime(preds["month"].unique()))
    sig = pair_signals(mm, daily, monthly, rf, acfg, months)
    pm = {m: prediction_matrices(preds, m, len(assets)) for m in acfg.models}
    ret_m = panel_matrix(monthly, "ret", mm.months, assets)
    cap_m = panel_matrix(monthly, "mktcap", mm.months, assets)
    q = acfg.pairs_n_quantiles
    controls = [c for c in ("beta", "size", "bm", "mom") if c in monthly.columns]
    ctrl = {c: panel_matrix(monthly, c, mm.months, assets) for c in controls}
    rows, stock_rows = [], []
    for t, s in sig.items():
        t1 = t + pd.offsets.MonthEnd(1)
        if t1 not in ret_m.index:
            continue
        r_next = ret_m.loc[t1].to_numpy(dtype=float)
        caps = cap_m.loc[t].to_numpy(dtype=float)
        samples = {"Unconditional": np.ones(len(assets), dtype=bool)}
        for m in acfg.models:
            if t not in pm[m]:
                continue
            pred = np.take_along_axis(pm[m][t], s["partners"], axis=1).mean(1)
            d_rc = pred - s["rc_hist"]
            samples[m] = _quantile_labels(d_rc, q) == q - 1
        for name, mask in samples.items():
            lab = np.full(len(assets), -1)
            lab[mask] = _quantile_labels(s["retdiff"][mask], q)
            sel_all = mask & (lab >= 0) & np.isfinite(r_next)
            rec = pd.DataFrame(
                {
                    "month": t1,
                    "sample": name,
                    "ret_next": r_next[sel_all],
                    "retdiff_q": lab[sel_all] + 1.0,
                }
            )
            for c in controls:
                rec[c] = ctrl[c].loc[t].to_numpy(dtype=float)[sel_all]
            stock_rows.append(rec)
            row = {"month": t1, "sample": name}
            for b in range(q):
                sel = (lab == b) & np.isfinite(r_next)
                if not sel.any():
                    continue
                row[f"EW_Q{b + 1}"] = float(np.mean(r_next[sel]))
                w = capped_value_weights(
                    np.nan_to_num(caps[sel], nan=0.0) + 1e-12, acfg.pairs_vw_cap
                )
                row[f"VW_Q{b + 1}"] = float(w @ r_next[sel])
            rows.append(row)
    ts = pd.DataFrame(rows)
    summ = []
    for name, g in ts.groupby("sample"):
        for scheme in ("EW", "VW"):
            r = {"sample": name, "scheme": scheme}
            for b in range(q):
                c = f"{scheme}_Q{b + 1}"
                r[f"Q{b + 1}_ann_%"] = 1200 * g[c].mean() if c in g else np.nan
            hml = g.get(f"{scheme}_Q{q}") - g.get(f"{scheme}_Q1")
            r["HML_ann_%"] = 1200 * hml.mean()
            r["HML_t"] = newey_west_tstat(hml.to_numpy(), 0)
            summ.append(r)
    stocks = pd.concat(stock_rows, ignore_index=True) if stock_rows else pd.DataFrame()
    fm = []
    for name, g in stocks.groupby("sample") if len(stocks) else []:
        for with_ctrl in (False, True):
            xs = ["retdiff_q", *(controls if with_ctrl else [])]
            res = fama_macbeth(g, "ret_next", xs)
            res.insert(0, "controls", with_ctrl)
            res.insert(0, "sample", name)
            fm.append(res)
    return {
        "pairs_timeseries": ts,
        "pairs_summary": pd.DataFrame(summ),
        "pairs_fama_macbeth": pd.concat(fm, ignore_index=True) if fm else pd.DataFrame(),
    }


def fama_macbeth(df: pd.DataFrame, y: str, xs: list[str], nw_lags: int = 0) -> pd.DataFrame:
    """Fama–MacBeth: monthly cross-sectional OLS, time-series mean of coefficients (in % for returns)."""
    coefs, r2 = [], []
    for _, g in df.groupby("month"):
        g = g[[y, *xs]].dropna()
        if len(g) <= len(xs) + 2:
            continue
        X = np.column_stack([np.ones(len(g)), g[xs].to_numpy(float)])
        b, *_ = np.linalg.lstsq(X, g[y].to_numpy(float), rcond=None)
        resid = g[y].to_numpy(float) - X @ b
        tss = np.sum((g[y] - g[y].mean()) ** 2)
        n, k = X.shape
        r2.append(1 - (resid @ resid / (n - k)) / (tss / (n - 1)) if tss > 0 else np.nan)
        coefs.append(b)
    if not coefs:
        return pd.DataFrame()
    C = np.array(coefs)
    names = ["intercept", *xs]
    return pd.DataFrame(
        {
            "term": names,
            "coef_%": 100 * C.mean(0),
            "t": [newey_west_tstat(C[:, k], nw_lags) for k in range(C.shape[1])],
            "adj_R2_%": 100 * np.nanmean(r2),
            "n_months": len(C),
        }
    )
