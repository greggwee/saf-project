"""Compare forecasts from different runs on a *common* target.

Changing the intraday estimator changes both the features and the realized
correlation being forecast, so R^2's computed against each run's own target are
not comparable.  :func:`compare_runs` re-scores every (run, model) against the
realized monthly correlations of one reference run, with that run's benchmark
model as the denominator, on the intersection of (month, pair) observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corrforecast.evaluation.metrics import dm_test, r2_oos, value_weights
from corrforecast.features.panel import load_monthly_matrices
from corrforecast.models.rolling import load_predictions


def compare_runs(
    output_dir: Path,
    target_run: str,
    runs: list[str],
    benchmark: str = "HAR",
    models: list[str] | None = None,
    winsor_q: float = 0.90,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    mm = load_monthly_matrices(output_dir / target_run / "features")
    base = load_predictions(output_dir / target_run)
    key = ["month", "i", "j"]
    merged = base[[*key, "target_month", "cap_i", "cap_j", benchmark]].rename(
        columns={benchmark: "__bench"}
    )
    cols = []
    for run in runs:
        p = load_predictions(output_dir / run)
        use = [
            m
            for m in (
                models
                or [
                    c
                    for c in p.columns
                    if c not in (*key, "target_month", "target", "cap_i", "cap_j")
                ]
            )
            if m in p.columns
        ]
        p = p[[*key, *use]].rename(columns={m: f"{run}:{m}" for m in use})
        merged = merged.merge(p, on=key, how="inner")
        cols += [f"{run}:{m}" for m in use]
    k = mm.months.get_indexer(merged["target_month"])
    merged["target"] = mm.rc_m[k, merged["i"].to_numpy(), merged["j"].to_numpy()]
    merged = merged[np.isfinite(merged["target"])].reset_index(drop=True)
    w = value_weights(merged, winsor_q).to_numpy()
    y, b = merged["target"].to_numpy(float), merged["__bench"].to_numpy(float)
    rows = []
    for c in cols:
        p = merged[c].to_numpy(float)
        rows.append(
            {
                "run": c.split(":", 1)[0],
                "model": c.split(":", 1)[1],
                "R2_EW_%": 100 * r2_oos(y, p, b),
                "R2_VW_%": 100 * r2_oos(y, p, b, w),
                "DM_vs_bench": dm_test(merged.rename(columns={c: "__m"}), "__m", "__bench"),
                "MSE": float(np.mean((y - p) ** 2)),
                "n_obs": len(merged),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["target_run"] = target_run
    return out
