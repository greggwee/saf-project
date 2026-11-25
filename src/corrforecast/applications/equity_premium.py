"""Equity premium predictions with average correlations (BLT §5.2, Pollet & Wilson 2010).

``AvgCorr^{theta,EW}_t`` is the mean predicted correlation for month t+1 across all pairs,
``AvgCorr^{theta,VW}_t`` the omega-weighted mean (eq. 9).  The ``RC`` variant uses the
realized correlations of month t.  Predictive regressions of the month-t+1 market excess
return on ``AvgCorr_t`` with and without the Welch–Goyal controls (Table 6).
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from corrforecast.config import ApplicationsConfig
from corrforecast.evaluation.metrics import value_weights


def average_correlations(
    preds: pd.DataFrame, models: tuple[str, ...], winsor_q: float = 0.90
) -> pd.DataFrame:
    w = value_weights(preds, winsor_q)
    out = {}
    g = preds.assign(_w=w)
    for m in models:
        out[f"{m}_EW"] = g.groupby("month")[m].mean()
        out[f"{m}_VW"] = g.assign(_x=g[m] * g["_w"]).groupby("month")["_x"].sum()
    # realized correlation of month t is the previous row's target (target of origin t-1)
    prev = g.assign(month=g["target_month"])  # realized RC_{t+1} indexed at t+1
    out["RC_EW"] = prev.groupby("month")["target"].mean()
    out["RC_VW"] = prev.assign(_x=prev["target"] * prev["_w"]).groupby("month")["_x"].sum()
    return pd.DataFrame(out)


def run_equity_premium(
    preds: pd.DataFrame,
    market: pd.DataFrame,
    acfg: ApplicationsConfig,
    winsor_q: float = 0.90,
    hac_lags: int = 0,
) -> dict[str, pd.DataFrame]:
    ac = average_correlations(preds, acfg.models, winsor_q)
    mk = market.set_index("month")
    controls = [c for c in acfg.equity_premium_controls if c in mk.columns]
    rows = []
    for scheme in ("EW", "VW"):
        for src in ("RC", *acfg.models):
            x = ac[f"{src}_{scheme}"].rename("AvgCorr")
            df = pd.concat([x, mk[controls]], axis=1, sort=True)
            df["y"] = mk["mkt_excess"].shift(-1).reindex(df.index)  # r_{t+1}
            for with_ctrl in (False, True):
                cols = ["AvgCorr", *(controls if with_ctrl else [])]
                d = df[[*cols, "y"]].dropna()
                if len(d) <= len(cols) + 2:
                    continue
                res = sm.OLS(d["y"], sm.add_constant(d[cols])).fit(
                    cov_type="HAC" if hac_lags else "nonrobust",
                    **({"cov_kwds": {"maxlags": hac_lags}} if hac_lags else {}),
                )
                rows.append(
                    {
                        "scheme": scheme,
                        "predictor": src,
                        "controls": with_ctrl,
                        "coef": res.params["AvgCorr"],
                        "t": res.tvalues["AvgCorr"],
                        "adj_R2_%": 100 * res.rsquared_adj,
                        "n": int(res.nobs),
                    }
                )
    return {"equity_premium": pd.DataFrame(rows), "avg_corr": ac.reset_index(names="month")}
