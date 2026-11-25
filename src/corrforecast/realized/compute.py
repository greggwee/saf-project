"""Stage 1: daily realized covariance and semicovariance matrices.

For each trading day d (BLT eq. 1 and A.1, with the overnight return included
as in Hansen & Lunde 2005):

    RCov_d   = 252 * ( IntradayCov_d   + r_on r_on' )
    RCov^-_d = 252 * ( IntradaySemi_d  + r_on^- r_on^-' )

``IntradayCov_d`` comes from the configured estimator (15-min RCov in the
baseline, or a noise-robust estimator); ``IntradaySemi_d`` from the estimator
itself if it provides one, else from ``realized.semicov_estimator``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from corrforecast.config import RunConfig
from corrforecast.data.loaders import InputData, IntradayReader
from corrforecast.realized.estimators import CovEstimator, build_estimator
from corrforecast.realized.sampling import DayObservations
from corrforecast.realized.store import DailyMatrixStore

log = logging.getLogger(__name__)


@dataclass
class _DayJob:
    obs: DayObservations
    overnight: np.ndarray


def _estimate_day(
    main: CovEstimator,
    semi: CovEstimator | None,
    job: _DayJob,
    annualization: float,
    include_overnight: bool,
) -> tuple[np.ndarray, np.ndarray]:
    est = main.estimate(job.obs)
    cov = est.cov
    semicov = est.semicov_neg
    if semicov is None:
        if semi is None:
            raise ValueError(
                f"{main!r} provides no semicovariance and no semicov_estimator is configured"
            )
        semicov = semi.estimate(job.obs).semicov_neg
        if semicov is None:
            raise ValueError(f"semicov_estimator {semi!r} does not produce semicovariances")
    if include_overnight:
        r = np.nan_to_num(job.overnight, nan=0.0)
        rn = np.minimum(r, 0.0)
        cov = cov + np.outer(r, r)
        semicov = semicov + np.outer(rn, rn)
    return annualization * cov, annualization * semicov


def overnight_matrix(inputs: InputData) -> pd.DataFrame:
    """(date x asset) overnight log returns from the daily panel's ``ret_overnight``."""
    if "ret_overnight" in inputs.daily.columns:
        return inputs.daily.pivot(index="date", columns="asset_id", values="ret_overnight").reindex(
            columns=inputs.assets
        )
    raise ValueError(
        "daily panel lacks `ret_overnight`; either provide it (see docs/data_contract.md) or set "
        "realized.include_overnight: false"
    )


class _Worker:
    """Picklable callable for process pools."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.main = build_estimator(cfg.realized.estimator.name, cfg.realized.estimator.params)
        self.semi = (
            None
            if self.main.provides_semicov
            else build_estimator(
                cfg.realized.semicov_estimator.name, cfg.realized.semicov_estimator.params
            )
        )

    def __call__(self, job: _DayJob) -> tuple[np.ndarray, np.ndarray]:
        return _estimate_day(
            self.main,
            self.semi,
            job,
            self.cfg.realized.annualization,
            self.cfg.realized.include_overnight,
        )


def compute_daily_matrices(
    cfg: RunConfig, inputs: InputData, overwrite: bool = False
) -> DailyMatrixStore:
    store = DailyMatrixStore(cfg.run_dir / "daily")
    store.init(
        inputs.assets,
        {
            "estimator": cfg.realized.estimator.model_dump(),
            "semicov_estimator": cfg.realized.semicov_estimator.model_dump(),
            "include_overnight": cfg.realized.include_overnight,
            "annualization": cfg.realized.annualization,
        },
    )
    reader = IntradayReader(
        cfg.data.intraday, inputs.assets, cfg.data.session_open, cfg.data.session_close
    )
    on = overnight_matrix(inputs) if cfg.realized.include_overnight else None
    worker = _Worker(cfg)
    days = inputs.trading_days
    pool = ProcessPoolExecutor(cfg.realized.n_jobs) if cfg.realized.n_jobs > 1 else None
    try:
        for year, ydays in pd.Series(days, index=days).groupby(days.year):
            if store.has_year(int(year)) and not overwrite:
                log.info("daily matrices for %d exist; skipping", year)
                continue
            jobs = []
            for d, frame in reader.iter_days(pd.DatetimeIndex(ydays.values)):
                obs = DayObservations.from_frame(
                    d, frame, inputs.assets, cfg.data.session_open, cfg.data.session_close
                )
                r_on = (
                    on.loc[d].to_numpy()
                    if on is not None and d in on.index
                    else np.zeros(len(inputs.assets))
                )
                jobs.append(_DayJob(obs, r_on))
            results = (
                list(pool.map(worker, jobs, chunksize=4)) if pool else [worker(j) for j in jobs]
            )
            cov = np.stack([r[0] for r in results])
            semi = np.stack([r[1] for r in results])
            store.write_year(
                int(year), pd.DatetimeIndex(ydays.values), {"cov": cov, "semicov": semi}
            )
            log.info("wrote daily matrices for %d (%d days)", year, len(ydays))
    finally:
        if pool:
            pool.shutdown()
    return store
