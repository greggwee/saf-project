"""Typed run configuration.

Every pipeline run is fully described by one YAML file that is parsed into
:class:`RunConfig`.  Defaults reproduce the choices in Bollerslev, Li & Tang
(2026) (henceforth BLT); every deviation must be made explicitly in YAML so
that runs are reproducible and diffable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    """Locations of the cleaned input data (produced by the data team).

    See ``docs/data_contract.md`` for the exact schema of every file.
    """

    intraday: Path = Field(
        description="Parquet file or (hive-partitioned) directory of intraday prices."
    )
    daily: Path = Field(description="Parquet file with the daily stock panel.")
    monthly: Path = Field(
        description="Parquet file with the monthly stock panel (characteristics, sector, mktcap, ret)."
    )
    market: Path | None = Field(
        default=None,
        description="Parquet file with monthly market data (needed for applications only).",
    )
    universe: Path | None = Field(
        default=None,
        description="Optional CSV/Parquet with an `asset_id` column fixing the stock universe.",
    )
    start: str = "2000-01-01"
    end: str = "2020-12-31"
    session_open: str = "09:30"
    session_close: str = "16:00"
    universe_rule: Literal["balanced", "all"] = "balanced"


class EstimatorConfig(_Strict):
    """A daily (co)variance estimator from the registry in ``corrforecast.realized.estimators``."""

    name: str = "realized_cov"
    params: dict[str, Any] = Field(default_factory=dict)


class RealizedConfig(_Strict):
    estimator: EstimatorConfig = EstimatorConfig(name="realized_cov", params={"sampling": "15min"})
    # Estimator used for the negative semicovariances when the main estimator does
    # not produce them (neither denoising paper defines a semicovariance).
    semicov_estimator: EstimatorConfig = EstimatorConfig(
        name="realized_cov", params={"sampling": "15min"}
    )
    include_overnight: bool = True
    annualization: float = 252.0
    n_jobs: int = 1


class FeatureConfig(_Strict):
    weekly_days: int = 5
    monthly_window: Literal["calendar", "21d"] = "calendar"
    exp_centers_of_mass: tuple[float, ...] = (1.0, 5.0, 21.0, 63.0)
    exp_window: int = 500
    factor_characteristics: tuple[str, ...] = (
        "beta",
        "size",
        "bm",
        "rev",
        # Stambaugh, Yu & Yuan (2012) 11 mispricing anomalies
        "nsi",
        "cei",
        "acc",
        "noa",
        "agr",
        "inta",
        "oscore",
        "mom",
        "gpf",
        "roa",
        "dis",
    )
    characteristics_no_rank: tuple[str, ...] = ("beta",)
    factor_add_intercept: bool = False
    sector_column: str = "sector"
    clip_to_unit: bool = True


class ModelSpec(_Strict):
    name: str
    method: Literal["ols", "lasso", "ridge", "enet", "pcr", "fnn"]
    features: str | list[str] = Field(
        description="Named feature set (see features.names) or explicit list."
    )
    params: dict[str, Any] = Field(default_factory=dict)


def _default_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="HAR", method="ols", features="har"),
        ModelSpec(name="SHAR", method="ols", features="shar"),
        ModelSpec(name="SHAR-F", method="ols", features="shar_f"),
        ModelSpec(name="SHAR-F-Exp", method="ols", features="all"),
        ModelSpec(name="LASSO", method="lasso", features="all"),
    ]


class ModelConfig(_Strict):
    benchmark: str = "HAR"
    specs: list[ModelSpec] = Field(default_factory=_default_models)
    train_years: int = 4
    val_years: int = 1
    first_test_year: int = 2008
    last_test_year: int = 2020
    refit_frequency_years: int = 1
    lasso_n_alphas: int = 100
    lasso_alpha_min_ratio: float = 1e-4
    refit_on_train_plus_val: bool = False
    clip_predictions: bool = True
    # Name of another run (under the same output_dir) whose realized monthly correlations are
    # used as the forecast TARGET, e.g. fit denoised features to the 15-minute target or the
    # 15-minute features to a denoised target. None = use this run's own target.
    target_run: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        if self.benchmark not in names:
            raise ValueError(f"benchmark {self.benchmark!r} not among model specs {names}")
        return self


class EvaluationConfig(_Strict):
    mktcap_winsor_quantile: float = 0.90
    subperiods: list[tuple[int, int]] = Field(
        default_factory=lambda: [(2008, 2011), (2012, 2015), (2016, 2020)]
    )


class ApplicationsConfig(_Strict):
    enabled: tuple[str, ...] = ("pairs_trading", "equity_premium", "risk_targeting", "gmv")
    models: tuple[str, ...] = ("HAR", "LASSO")
    pairs_n_partners: int = 20
    pairs_lookback_months: int = 12
    pairs_n_quantiles: int = 5
    pairs_vw_cap: float = 0.10
    risk_aversions: tuple[float, ...] = (2.0, 5.0, 10.0)
    equity_premium_controls: tuple[str, ...] = (
        "dp",
        "ep",
        "bm",
        "ntis",
        "tbl",
        "tms",
        "dfy",
        "svar",
    )
    long_short_characteristics: tuple[str, ...] = (
        "acc",
        "agr",
        "beta",
        "bm",
        "cei",
        "dis",
        "gpf",
        "inta",
        "mom",
        "noa",
        "nsi",
        "oscore",
        "roa",
        "rev",
        "size",
    )
    long_short_quantiles: int = 10


class RunConfig(_Strict):
    name: str
    output_dir: Path = Path("artifacts")
    seed: int = 0
    data: DataConfig
    realized: RealizedConfig = RealizedConfig()
    features: FeatureConfig = FeatureConfig()
    models: ModelConfig = ModelConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    applications: ApplicationsConfig = ApplicationsConfig()

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not v or any(c in v for c in r"\/:*?\"<>| "):
            raise ValueError("run name must be a non-empty filesystem-safe token")
        return v

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.name


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Load a YAML config. Relative paths are interpreted relative to the working directory.

    A config may start with ``extends: <other.yaml>`` (resolved relative to the file); the
    parent is loaded first and this file is deep-merged on top, so variants only state
    what differs from the baseline.  ``overrides`` is deep-merged last.
    """
    raw = _read_with_extends(Path(path))
    if overrides:
        raw = _deep_merge(raw, overrides)
    return RunConfig.model_validate(raw)


def _read_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular `extends` chain: {[*_seen, path]}")
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_read_with_extends(path.parent / parent, (*_seen, path)), raw)


def _deep_merge(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def dump_config(cfg: RunConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=False)
