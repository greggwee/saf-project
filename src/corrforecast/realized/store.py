"""On-disk store for daily N x N matrices.

Layout (one directory per run stage)::

    <root>/assets.npy              universe order (row/column order of every matrix)
    <root>/meta.json               estimator configuration and provenance
    <root>/<kind>/<year>.npy       float32 array (n_days, N, N), memory-mappable
    <root>/dates/<year>.npy        datetime64[D] array (n_days,)

``kind`` is ``cov`` (daily realized covariance incl. overnight, annualised) or
``semicov`` (negative realized semicovariance).  Years are written atomically so
an interrupted run can resume.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KINDS = ("cov", "semicov")


class DailyMatrixStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ------------------------------------------------------------------ write
    def init(self, assets: np.ndarray, meta: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "assets.npy").exists():
            old = np.load(self.root / "assets.npy", allow_pickle=False)
            if not np.array_equal(old, assets):
                raise ValueError(
                    f"store at {self.root} was built for a different universe; delete it or change run name"
                )
        np.save(self.root / "assets.npy", np.asarray(assets))
        with open(self.root / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def write_year(self, year: int, dates: pd.DatetimeIndex, mats: dict[str, np.ndarray]) -> None:
        for kind, arr in mats.items():
            d = self.root / kind
            d.mkdir(exist_ok=True)
            tmp = d / f".{year}.tmp.npy"
            np.save(tmp, arr.astype(np.float32))
            os.replace(tmp, d / f"{year}.npy")
        (self.root / "dates").mkdir(exist_ok=True)
        tmp = self.root / "dates" / f".{year}.tmp.npy"
        np.save(tmp, np.asarray(dates.values, dtype="datetime64[D]"))
        os.replace(tmp, self.root / "dates" / f"{year}.npy")

    # ------------------------------------------------------------------- read
    @property
    def assets(self) -> np.ndarray:
        return np.load(self.root / "assets.npy", allow_pickle=False)

    def years(self) -> list[int]:
        d = self.root / "dates"
        return sorted(int(p.stem) for p in d.glob("[0-9]*.npy")) if d.exists() else []

    def has_year(self, year: int) -> bool:
        return (self.root / "dates" / f"{year}.npy").exists() and all(
            (self.root / k / f"{year}.npy").exists() for k in KINDS
        )

    def dates(self) -> pd.DatetimeIndex:
        parts = [np.load(self.root / "dates" / f"{y}.npy") for y in self.years()]
        return pd.DatetimeIndex(np.concatenate(parts)) if parts else pd.DatetimeIndex([])

    def load(
        self, kind: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """Load ``(dates, array[n_days, N, N])`` for ``start <= date <= end`` (float64)."""
        ys = [
            y
            for y in self.years()
            if (start is None or y >= start.year) and (end is None or y <= end.year)
        ]
        ds_, arrs = [], []
        for y in ys:
            d = pd.DatetimeIndex(np.load(self.root / "dates" / f"{y}.npy"))
            a = np.load(self.root / kind / f"{y}.npy", mmap_mode="r")
            keep = np.ones(len(d), dtype=bool)
            if start is not None:
                keep &= d >= start
            if end is not None:
                keep &= d <= end
            ds_.append(d[keep])
            arrs.append(np.asarray(a[keep], dtype=np.float64))
        if not arrs:
            n = len(self.assets)
            return pd.DatetimeIndex([]), np.empty((0, n, n))
        return pd.DatetimeIndex(np.concatenate([d.values for d in ds_])), np.concatenate(arrs)
