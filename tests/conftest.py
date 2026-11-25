from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from corrforecast.config import RunConfig
from corrforecast.data.synthetic import SyntheticSpec, write_synthetic


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    write_synthetic(
        out,
        SyntheticSpec(n_assets=20, start="2001-01-01", end="2005-12-31", steps_per_day=26, seed=7),
    )
    return out


def make_config(data_dir: Path, out_dir: Path, **overrides) -> RunConfig:
    raw = {
        "name": "test",
        "output_dir": str(out_dir),
        "data": {
            "intraday": str(data_dir / "intraday"),
            "daily": str(data_dir / "daily.parquet"),
            "monthly": str(data_dir / "monthly.parquet"),
            "market": str(data_dir / "market.parquet"),
            "start": "2001-01-01",
            "end": "2005-12-31",
        },
        # few characteristics so that K < N and the factor projection is non-trivial
        "features": {"exp_window": 60, "factor_characteristics": ["beta", "size", "bm"]},
        "models": {
            "train_years": 2,
            "val_years": 1,
            "first_test_year": 2004,
            "last_test_year": 2005,
        },
        "applications": {
            "pairs_n_partners": 4,
            "long_short_quantiles": 3,
            "long_short_characteristics": ["beta", "size", "bm"],
        },
    }
    for k, v in overrides.items():
        raw[k] = {**raw.get(k, {}), **v} if isinstance(v, dict) else v
    return RunConfig.model_validate(raw)
