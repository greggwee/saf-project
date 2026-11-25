"""Read the pair-month feature panel and month-level matrices written by :mod:`build`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


def load_pairs(
    features_dir: Path,
    target_years: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    dropna_target: bool = True,
) -> pd.DataFrame:
    """Load pair rows whose *target month* falls in ``target_years``.

    Rows are indexed by forecast origin ``month`` (features known at month-end t);
    ``target_month`` = t + 1.  Selecting by target year is what makes the
    train/validation/test split free of look-ahead: a model refit at the end of
    year t only ever sees targets dated <= December of year t.
    """
    dset = ds.dataset(Path(features_dir) / "pairs", format="parquet", partitioning="hive")
    flt = None
    if target_years is not None:
        yrs = sorted(set(target_years))
        origin_years = sorted({y for y in yrs} | {y - 1 for y in yrs})
        flt = ds.field("year").isin(origin_years)
    cols = None if columns is None else list(dict.fromkeys(["month", "i", "j", "target", *columns]))
    df = dset.to_table(filter=flt, columns=cols).to_pandas()
    if "year" in df.columns:
        df = df.drop(columns="year")
    df["month"] = pd.to_datetime(df["month"])
    df["target_month"] = (df["month"] + pd.offsets.MonthEnd(1)).dt.normalize()
    if target_years is not None:
        df = df[df["target_month"].dt.year.isin(list(target_years))]
    if dropna_target:
        df = df[df["target"].notna()]
    return df.sort_values(["month", "i", "j"]).reset_index(drop=True)


@dataclass
class MonthlyMatrices:
    months: pd.DatetimeIndex
    assets: np.ndarray
    cov_m: np.ndarray
    rc_m: np.ndarray
    rv_d: np.ndarray
    rv_w: np.ndarray
    rv_m: np.ndarray

    def index(self, month: pd.Timestamp) -> int:
        return int(self.months.get_loc(pd.Timestamp(month)))


def load_monthly_matrices(features_dir: Path) -> MonthlyMatrices:
    d = Path(features_dir) / "monthly"
    arr = {k: np.load(d / f"{k}.npy") for k in ("cov_m", "rc_m", "rv_d", "rv_w", "rv_m")}
    return MonthlyMatrices(
        months=pd.DatetimeIndex(np.load(d / "months.npy")),
        assets=np.load(d / "assets.npy"),
        **{k: v.astype(np.float64) for k, v in arr.items()},
    )
