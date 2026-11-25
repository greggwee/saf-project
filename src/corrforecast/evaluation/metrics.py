"""Statistical forecast evaluation (BLT §4.1).

* :func:`r2_oos` — out-of-sample R^2 relative to the benchmark (eqs. 4, 5).
* :func:`dm_test` — the cross-sectional modified Diebold–Mariano test (eq. 6):
  per-stock averages of pair loss differentials, then a t-statistic across stocks.
* Value weights ``omega_ij,t`` are products of month-t market caps, winsorised at the
  90th cross-sectional percentile each month (endnote 11), normalised to sum to one
  within each month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def value_weights(df: pd.DataFrame, winsor_q: float = 0.90) -> pd.Series:
    """Per-row VW weights from ``cap_i, cap_j`` (normalised within each origin month)."""
    caps = pd.concat(
        [
            df[["month", "i", "cap_i"]].rename(columns={"i": "a", "cap_i": "cap"}),
            df[["month", "j", "cap_j"]].rename(columns={"j": "a", "cap_j": "cap"}),
        ]
    )
    caps = caps.drop_duplicates(["month", "a"])
    cap_q = caps.groupby("month")["cap"].quantile(winsor_q)
    ci = np.minimum(df["cap_i"].to_numpy(), df["month"].map(cap_q).to_numpy())
    cj = np.minimum(df["cap_j"].to_numpy(), df["month"].map(cap_q).to_numpy())
    w = pd.Series(ci * cj, index=df.index)
    return w / w.groupby(df["month"]).transform("sum")


def r2_oos(
    y: np.ndarray, pred: np.ndarray, bench: np.ndarray, weights: np.ndarray | None = None
) -> float:
    w = np.ones_like(y) if weights is None else weights
    return float(1.0 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - bench) ** 2))


def dm_test(
    df: pd.DataFrame, model_a: str, model_b: str, weights: pd.Series | None = None
) -> float:
    """Modified DM t-statistic; positive means ``model_a`` beats ``model_b`` (lower squared error).

    ``d_ij,t = e_b^2 - e_a^2``; ``d_k`` is the mean over all (pair, month) observations
    involving stock k; DM = mean(d_k) / (sd(d_k)/sqrt(K)).  With ``weights`` the
    loss differentials are multiplied by ``omega_ij,t * n_pairs_t`` (mean weight one).
    """
    y = df["target"].to_numpy(dtype=float)
    d = (y - df[model_b].to_numpy(dtype=float)) ** 2 - (y - df[model_a].to_numpy(dtype=float)) ** 2
    if weights is not None:
        n_t = df.groupby("month")["i"].transform("size").to_numpy()
        d = d * weights.to_numpy() * n_t
    long = pd.DataFrame(
        {"k": np.concatenate([df["i"].to_numpy(), df["j"].to_numpy()]), "d": np.concatenate([d, d])}
    )
    dk = long.groupby("k")["d"].mean().to_numpy()
    se = dk.std(ddof=1) / np.sqrt(len(dk))
    return float(dk.mean() / se) if se > 0 else float("nan")


def r2_table(
    preds: pd.DataFrame,
    models: list[str],
    benchmark: str,
    winsor_q: float = 0.90,
    subperiods: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """R^2_OOS (in %) of every model relative to the benchmark: full sample and sub-periods (Tables 2, 7)."""
    rows = []
    periods: list[tuple[str, pd.Series]] = [("full", pd.Series(True, index=preds.index))]
    for a, b in subperiods or []:
        yr = preds["target_month"].dt.year
        periods.append((f"{a}-{b}", (yr >= a) & (yr <= b)))
    for label, mask in periods:
        sub = preds[mask]
        if sub.empty:
            continue
        w = value_weights(sub, winsor_q).to_numpy()
        y, bench = sub["target"].to_numpy(float), sub[benchmark].to_numpy(float)
        for m in models:
            p = sub[m].to_numpy(float)
            rows.append(
                {
                    "period": label,
                    "model": m,
                    "R2_EW_%": 100 * r2_oos(y, p, bench),
                    "R2_VW_%": 100 * r2_oos(y, p, bench, w),
                    "n_obs": len(sub),
                }
            )
    return pd.DataFrame(rows)


def dm_table(
    preds: pd.DataFrame,
    models: list[str],
    benchmark: str,
    value_weighted: bool = False,
    winsor_q: float = 0.90,
) -> pd.DataFrame:
    """Pairwise DM statistics; entry (row, col) > 0 means row model outperforms column model (Table 3)."""
    w = value_weights(preds, winsor_q) if value_weighted else None
    cols = [benchmark] + [m for m in models if m != benchmark]
    out = pd.DataFrame(index=[m for m in models if m != benchmark], columns=cols[:-1], dtype=float)
    for r_i, row in enumerate(out.index):
        for col in cols[: r_i + 1]:
            out.loc[row, col] = dm_test(preds, row, col, w)
    return out
