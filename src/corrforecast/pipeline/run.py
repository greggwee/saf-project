"""Stage orchestration. Each stage reads the previous stage's artifacts from ``<output_dir>/<run name>``.

realized  -> <run>/daily/            daily RCov and RCov^- matrices
features  -> <run>/features/         pair-month panel + monthly matrices
forecast  -> <run>/forecasts/        rolling out-of-sample predictions, coefficients
evaluate  -> <run>/results/          R2_OOS, DM tables, feature contributions
apps      -> <run>/results/          pairs trading, equity premium, risk targeting, GMV
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd

from corrforecast.config import RunConfig, dump_config
from corrforecast.data.loaders import InputData, load_inputs
from corrforecast.evaluation.metrics import dm_table, r2_table
from corrforecast.features.build import build_features
from corrforecast.features.panel import load_monthly_matrices
from corrforecast.models.rolling import feature_contributions, load_predictions, run_rolling
from corrforecast.realized.compute import compute_daily_matrices
from corrforecast.realized.store import DailyMatrixStore

log = logging.getLogger(__name__)

STAGES = ("realized", "features", "forecast", "evaluate", "apps")


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _inputs(cfg: RunConfig) -> InputData:
    return load_inputs(
        cfg.data, cfg.features.factor_characteristics, cfg.applications.equity_premium_controls
    )


def _write_tables(results_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(results_dir / f"{name}.csv", index=not isinstance(df.index, pd.RangeIndex))


def stage_evaluate(cfg: RunConfig) -> dict[str, pd.DataFrame]:
    preds = load_predictions(cfg.run_dir)
    models = [s.name for s in cfg.models.specs]
    bench = cfg.models.benchmark
    q = cfg.evaluation.mktcap_winsor_quantile
    tables = {
        "r2_oos": r2_table(
            preds, [m for m in models if m != bench], bench, q, list(cfg.evaluation.subperiods)
        ),
        "dm_ew": dm_table(preds, models, bench, False, q),
        "dm_vw": dm_table(preds, models, bench, True, q),
    }
    coefs = pd.read_parquet(cfg.run_dir / "forecasts" / "coefficients.parquet")
    for spec in cfg.models.specs:
        if spec.method in ("lasso", "enet"):
            tables[f"feature_contributions_{spec.name}"] = feature_contributions(coefs, spec.name)
    _write_tables(cfg.run_dir / "results", tables)
    return tables


def stage_apps(cfg: RunConfig, inputs: InputData) -> dict[str, pd.DataFrame]:
    from corrforecast.applications.equity_premium import run_equity_premium
    from corrforecast.applications.pairs_trading import run_pairs_trading
    from corrforecast.applications.portfolios import run_gmv, run_risk_targeting

    preds = load_predictions(cfg.run_dir)
    mm = load_monthly_matrices(cfg.run_dir / "features")
    acfg = cfg.applications
    missing = [m for m in acfg.models if m not in preds.columns]
    if missing:
        raise ValueError(f"applications.models {missing} not among forecasted models")
    tables: dict[str, pd.DataFrame] = {}
    if "pairs_trading" in acfg.enabled:
        tables.update(
            run_pairs_trading(preds, mm, inputs.daily, inputs.monthly, inputs.market, acfg)
        )
    if "equity_premium" in acfg.enabled:
        if inputs.market is None:
            log.warning("no market file configured; skipping equity premium application")
        else:
            tables.update(
                run_equity_premium(
                    preds, inputs.market, acfg, cfg.evaluation.mktcap_winsor_quantile
                )
            )
    if "risk_targeting" in acfg.enabled:
        tables.update(run_risk_targeting(preds, mm, inputs.monthly, acfg, seed=cfg.seed))
    if "gmv" in acfg.enabled:
        tables.update(run_gmv(preds, mm, inputs.monthly, inputs.market, acfg))
    _write_tables(cfg.run_dir / "results", tables)
    return tables


def run(cfg: RunConfig, stages: tuple[str, ...] = STAGES, overwrite: bool = False) -> None:
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; valid: {STAGES}")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, cfg.run_dir / "config.resolved.yaml")
    stage_times: dict[str, dict[str, float]] = {}
    manifest = {"git_rev": _git_rev(), "python": platform.python_version(), "stages": stage_times}
    inputs = _inputs(cfg) if {"realized", "features", "apps"} & set(stages) else None
    for stage in STAGES:
        if stage not in stages:
            continue
        t0 = time.time()
        log.info("=== stage %s ===", stage)
        if stage == "realized":
            compute_daily_matrices(cfg, inputs, overwrite=overwrite)  # type: ignore[arg-type]
        elif stage == "features":
            build_features(cfg, inputs, DailyMatrixStore(cfg.run_dir / "daily"))  # type: ignore[arg-type]
        elif stage == "forecast":
            run_rolling(cfg)
        elif stage == "evaluate":
            stage_evaluate(cfg)
        elif stage == "apps":
            stage_apps(cfg, inputs)  # type: ignore[arg-type]
        stage_times[stage] = {"seconds": round(time.time() - t0, 2)}
    with open(cfg.run_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
