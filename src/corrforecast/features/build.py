"""Build the monthly pair-level feature panel from daily matrices.

For every month-end t and every stock pair (i < j) we compute the 25 BLT
features (Appendix A) and the target ``RC_{ij,t+1}``, the realized correlation of
the next calendar month.  Output (``<run>/features``):

* ``pairs/year=<Y>/part.parquet`` — one row per (month, pair): ``month, i, j``,
  the 25 features, ``target``, ``cap_i, cap_j`` (month-t market caps), ``same_sector``.
* ``monthly/*.npy`` — month-level matrices reused by the applications:
  ``cov_m`` (N x N monthly RCov), ``rc_m`` (realized correlation), ``rv_d/rv_w/rv_m``
  (N realized variances), and ``months.npy``.

Conventions (all configurable in ``FeatureConfig``):

* daily = last trading day of month t; weekly = mean of the last 5 daily RCov;
  monthly = mean over the calendar month (or the last 21 days);
* correlations of averaged covariances, i.e. ``RC^h = corr(mean RCov)``;
  semicorrelations normalise ``RCov^-`` by its own diagonal (negative semivariances);
* ExpRC uses exactly ``exp_window`` (500) lagged days; months without a full
  window are skipped (BLT use 2000–2002 as burn-in).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from corrforecast.config import FeatureConfig, RunConfig
from corrforecast.data.loaders import InputData
from corrforecast.features import names
from corrforecast.features.transforms import (
    exp_weights,
    factor_projection,
    loading_matrix,
    sector_average,
)
from corrforecast.realized.psd import cov_to_corr
from corrforecast.realized.store import DailyMatrixStore

log = logging.getLogger(__name__)


@dataclass
class MonthInputs:
    """Low-frequency cross-sectional information at month-end t, aligned to the universe."""

    L: np.ndarray
    sectors: np.ndarray
    mktcap: np.ndarray


def month_inputs(
    monthly: pd.DataFrame, assets: np.ndarray, month: pd.Timestamp, fcfg: FeatureConfig
) -> MonthInputs:
    """Latest available characteristics at or before ``month`` for each asset."""
    hist = monthly[monthly["month"] <= month]
    last = (
        hist.sort_values("month").groupby("asset_id").tail(1).set_index("asset_id").reindex(assets)
    )
    chars = last[list(fcfg.factor_characteristics)]
    L = loading_matrix(chars, fcfg.characteristics_no_rank, fcfg.factor_add_intercept)
    sectors = last[fcfg.sector_column].astype(str).fillna("NA").to_numpy()
    return MonthInputs(L=L, sectors=sectors, mktcap=last["mktcap"].to_numpy(dtype=float))


def monthly_matrices(
    cov: np.ndarray,
    semi: np.ndarray,
    month_pos: np.ndarray,
    end: int,
    fcfg: FeatureConfig,
    mi: MonthInputs,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """All feature correlation matrices for one month.

    ``cov``/``semi`` are buffers of daily matrices, ``end`` the buffer index of the
    month's last trading day and ``month_pos`` the buffer indices of the month's days.
    Returns ``(features, extras)`` or ``None`` if the exponential window is incomplete.
    """
    W = fcfg.exp_window
    if end + 1 < W or fcfg.weekly_days > W:
        return None
    clip = fcfg.clip_to_unit
    with np.errstate(invalid="ignore"):
        win = {
            "d": slice(end, end + 1),
            "w": slice(max(0, end - fcfg.weekly_days + 1), end + 1),
            "m": month_pos
            if fcfg.monthly_window == "calendar"
            else slice(max(0, end - 20), end + 1),
        }
        c = {h: np.nanmean(cov[s], axis=0) for h, s in win.items()}
        n = {h: np.nanmean(semi[s], axis=0) for h, s in win.items()}
        out: dict[str, np.ndarray] = {}
        for h in names.HAR_HORIZONS:
            out[f"RC_{h}"] = cov_to_corr(c[h], clip)
            out[f"RC_{h}_neg"] = cov_to_corr(n[h], clip)
            out[f"FRC_{h}"] = cov_to_corr(factor_projection(c[h], mi.L), clip)
        # most recent day first; one NaN-cleaned copy reused for all centers of mass
        lagged_c = np.nan_to_num(cov[end - W + 1 : end + 1][::-1])
        lagged_s = np.nan_to_num(semi[end - W + 1 : end + 1][::-1])
        for h, com in zip(names.EXP_HORIZONS, fcfg.exp_centers_of_mass, strict=True):
            w = exp_weights(com, W)
            ec = cov_to_corr(np.tensordot(w, lagged_c, axes=1), clip)
            es = cov_to_corr(np.tensordot(w, lagged_s, axes=1), clip)
            out[f"ExpRC_{h}"], out[f"ExpRC_{h}_neg"] = ec, es
            out[f"ExpScRC_{h}"] = sector_average(ec, mi.sectors)
            out[f"ExpScRC_{h}_neg"] = sector_average(es, mi.sectors)
    extras = {
        "cov_m": c["m"],
        "rv_d": np.diagonal(c["d"]).copy(),
        "rv_w": np.diagonal(c["w"]).copy(),
        "rv_m": np.diagonal(c["m"]).copy(),
    }
    return out, extras


class _PanelWriter:
    def __init__(self, root: Path):
        self.root = root
        self.frames: dict[int, list[pd.DataFrame]] = {}

    def add(self, frame: pd.DataFrame) -> None:
        y = int(frame["month"].iloc[0].year)
        self.frames.setdefault(y, []).append(frame)

    def flush(self, before_year: int | None = None) -> None:
        for y in sorted(list(self.frames)):
            if before_year is not None and y >= before_year:
                continue
            d = self.root / f"year={y}"
            d.mkdir(parents=True, exist_ok=True)
            pd.concat(self.frames.pop(y), ignore_index=True).to_parquet(
                d / "part.parquet", index=False
            )


def build_features(cfg: RunConfig, inputs: InputData, store: DailyMatrixStore) -> Path:
    fcfg = cfg.features
    out_dir = cfg.run_dir / "features"
    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
    (out_dir / "monthly").mkdir(parents=True, exist_ok=True)
    assets = store.assets
    if not np.array_equal(assets, inputs.assets):
        raise ValueError("daily store universe differs from the loaded inputs")
    N = len(assets)
    iu, ju = np.triu_indices(N, 1)
    writer = _PanelWriter(out_dir / "pairs")
    keep_days = fcfg.exp_window + 40

    buf_dates = pd.DatetimeIndex([])
    buf_c = np.empty((0, N, N), dtype=np.float32)
    buf_s = np.empty((0, N, N), dtype=np.float32)
    pending: tuple[pd.Timestamp, pd.DataFrame] | None = None
    monthly_store: dict[str, list[np.ndarray]] = {
        k: [] for k in ("cov_m", "rc_m", "rv_d", "rv_w", "rv_m")
    }
    month_list: list[pd.Timestamp] = []

    for year in store.years():
        d, c = store.load("cov", pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
        _, s = store.load("semicov", pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
        buf_dates = buf_dates.append(d)
        buf_c = np.concatenate([buf_c, c.astype(np.float32)])
        buf_s = np.concatenate([buf_s, s.astype(np.float32)])
        if len(buf_dates) > keep_days + len(d):
            cut = len(buf_dates) - keep_days - len(d)
            buf_dates, buf_c, buf_s = buf_dates[cut:], buf_c[cut:], buf_s[cut:]
        per = buf_dates.to_period("M")
        for mo in pd.PeriodIndex(d.to_period("M").unique()):
            pos = np.flatnonzero(per == mo)
            end = int(pos[-1])
            month = mo.to_timestamp(how="end").normalize()
            mi = month_inputs(inputs.monthly, assets, month, fcfg)
            res = monthly_matrices(buf_c, buf_s, pos, end, fcfg, mi)
            # realized monthly correlation is needed as a target even when features are incomplete
            with np.errstate(invalid="ignore"):
                cov_m = np.nanmean(buf_c[pos], axis=0).astype(np.float64)
            rc_m = cov_to_corr(cov_m, fcfg.clip_to_unit)
            month_list.append(month)
            monthly_store["cov_m"].append(cov_m)
            monthly_store["rc_m"].append(rc_m)
            dg = np.diagonal(cov_m)
            last = np.diagonal(buf_c[end]).astype(np.float64)
            wk = np.diagonal(
                np.nanmean(buf_c[max(0, end - fcfg.weekly_days + 1) : end + 1], axis=0)
            ).astype(np.float64)
            monthly_store["rv_d"].append(last)
            monthly_store["rv_w"].append(wk)
            monthly_store["rv_m"].append(dg.copy())
            if pending is not None:
                _pm, pf = pending
                pf["target"] = rc_m[iu, ju].astype(np.float32)
                writer.add(pf)
                pending = None
            if res is None:
                continue
            feats, _ = res
            frame = pd.DataFrame(
                {"month": month, "i": iu.astype(np.int32), "j": ju.astype(np.int32)}
            )
            for f in names.ALL_FEATURES:
                frame[f] = feats[f][iu, ju].astype(np.float32)
            frame["cap_i"] = mi.mktcap[iu]
            frame["cap_j"] = mi.mktcap[ju]
            frame["same_sector"] = mi.sectors[iu] == mi.sectors[ju]
            pending = (month, frame)
        writer.flush(before_year=year)
    if pending is not None:  # last month: no realized target yet
        pf = pending[1]
        pf["target"] = np.float32(np.nan)
        writer.add(pf)
    writer.flush()
    mdir = out_dir / "monthly"
    np.save(mdir / "months.npy", np.asarray(month_list, dtype="datetime64[D]"))
    np.save(mdir / "assets.npy", assets)
    for k, v in monthly_store.items():
        np.save(
            mdir / f"{k}.npy",
            np.stack(v).astype(np.float32 if k in ("cov_m", "rc_m") else np.float64),
        )
    log.info("built features for %d months", len(month_list))
    return out_dir
