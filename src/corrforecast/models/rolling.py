"""Rolling-window estimation (BLT §3.1).

At the end of each year t the models are refit and used to forecast every
month of year t+1 (the test year).  Samples are indexed by **target** year:

* penalised models: train on target years t-4..t-1, choose the penalty on year t;
* OLS models: estimate on target years t-4..t (same information set, no tuning);
* test on target year t+1.

Outputs (``<run>/forecasts``): ``predictions/year=<Y>/part.parquet`` with
``month, target_month, i, j, target, cap_i, cap_j`` and one column per model,
and ``coefficients.parquet`` (standardised coefficients per model and year).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from corrforecast.config import ModelConfig, ModelSpec, RunConfig
from corrforecast.features.names import ALL_FEATURES, resolve_features
from corrforecast.features.panel import load_monthly_matrices, load_pairs
from corrforecast.models.estimators import make_model

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Split:
    test_year: int
    train_years: tuple[int, ...]
    val_years: tuple[int, ...]

    @property
    def ols_years(self) -> tuple[int, ...]:
        return self.train_years + self.val_years


def make_splits(mcfg: ModelConfig) -> list[Split]:
    out = []
    for y in range(mcfg.first_test_year, mcfg.last_test_year + 1, mcfg.refit_frequency_years):
        val = tuple(range(y - mcfg.val_years, y))
        train = tuple(range(val[0] - mcfg.train_years, val[0]))
        out.append(Split(test_year=y, train_years=train, val_years=val))
    return out


def _model_params(spec: ModelSpec, mcfg: ModelConfig) -> dict:
    p = {
        "n_alphas": mcfg.lasso_n_alphas,
        "alpha_min_ratio": mcfg.lasso_alpha_min_ratio,
        "refit_on_train_plus_val": mcfg.refit_on_train_plus_val,
    }
    p.update(spec.params)
    return p


def replace_target(
    panel: pd.DataFrame, target_features_dir: Path, assets: np.ndarray
) -> pd.DataFrame:
    """Swap ``target`` for the realized monthly correlation stored in another run."""
    mm = load_monthly_matrices(target_features_dir)
    if not np.array_equal(mm.assets, assets):
        raise ValueError(f"target run {target_features_dir} has a different universe")
    k = mm.months.get_indexer(panel["target_month"])
    ok = k >= 0
    tgt = np.full(len(panel), np.nan)
    tgt[ok] = mm.rc_m[k[ok], panel["i"].to_numpy()[ok], panel["j"].to_numpy()[ok]]
    panel = panel.assign(target=tgt)
    return panel[panel["target"].notna()].reset_index(drop=True)


def run_rolling(cfg: RunConfig, test_years: list[int] | None = None) -> Path:
    mcfg = cfg.models
    feat_dir = cfg.run_dir / "features"
    out = cfg.run_dir / "forecasts"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    coef_rows = []
    specs = mcfg.specs
    target_dir = None if mcfg.target_run is None else cfg.output_dir / mcfg.target_run / "features"
    assets = np.load(feat_dir / "monthly" / "assets.npy", allow_pickle=False)
    for split in make_splits(mcfg):
        if test_years is not None and split.test_year not in test_years:
            continue
        years = (*split.train_years, *split.val_years, split.test_year)
        panel = load_pairs(feat_dir, years, columns=[*ALL_FEATURES, "cap_i", "cap_j"])
        if target_dir is not None:
            panel = replace_target(panel, target_dir, assets)
        ty = panel["target_month"].dt.year
        tr, va, te = ty.isin(split.train_years), ty.isin(split.val_years), ty == split.test_year
        if not te.any():
            log.warning("no test data for %d; skipping", split.test_year)
            continue
        if not tr.any():
            raise ValueError(
                f"no training data for test year {split.test_year}; check feature burn-in"
            )
        y = panel["target"].to_numpy(dtype=np.float64)
        preds = panel.loc[
            te, ["month", "target_month", "i", "j", "target", "cap_i", "cap_j"]
        ].reset_index(drop=True)
        for spec in specs:
            feats = list(resolve_features(spec.features))
            X = panel[feats].to_numpy(dtype=np.float64)
            model = make_model(spec.method, **_model_params(spec, mcfg))
            if model.needs_validation:
                model.fit(X[tr.to_numpy()], y[tr.to_numpy()], X[va.to_numpy()], y[va.to_numpy()])
            else:
                m = (tr | va).to_numpy()
                model.fit(X[m], y[m])
            p = model.predict(X[te.to_numpy()])
            if mcfg.clip_predictions:  # "insanity filter" (BLT endnote 10)
                p = np.clip(p, -1.0, 1.0)
            preds[spec.name] = p.astype(np.float32)
            res = model.result
            assert res is not None
            row = {"model": spec.name, "test_year": split.test_year, **res.hyper}
            row.update({f"coef_std:{f}": c for f, c in zip(feats, res.coef_std, strict=True)})
            row["intercept"] = res.intercept
            coef_rows.append(row)
            log.info("test %d %-12s fitted (%s)", split.test_year, spec.name, res.hyper)
        d = out / "predictions" / f"year={split.test_year}"
        d.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(d / "part.parquet", index=False)
    pd.DataFrame(coef_rows).to_parquet(out / "coefficients.parquet", index=False)
    return out


def load_predictions(run_dir: Path) -> pd.DataFrame:
    import pyarrow.dataset as ds

    df = (
        ds.dataset(
            Path(run_dir) / "forecasts" / "predictions", format="parquet", partitioning="hive"
        )
        .to_table()
        .to_pandas()
    )
    df = df.drop(columns=[c for c in ("year",) if c in df.columns])
    for c in ("month", "target_month"):
        df[c] = pd.to_datetime(df[c])
    return df.sort_values(["month", "i", "j"]).reset_index(drop=True)


def feature_contributions(coefs: pd.DataFrame, model: str) -> pd.DataFrame:
    """Scaled absolute standardised coefficients per test year (BLT Figure 4); rows: features."""
    sub = coefs[coefs["model"] == model].set_index("test_year")
    cols = [c for c in sub.columns if c.startswith("coef_std:")]
    a = sub[cols].abs().fillna(0.0)
    share = a.div(a.sum(axis=1).replace(0, np.nan), axis=0)
    share.columns = [c.split(":", 1)[1] for c in cols]
    return share.T
