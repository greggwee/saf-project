"""Estimator interface and registry.

A daily estimator maps one day of intraday observations to an estimate of the
*intraday* integrated covariance matrix (no overnight return, not annualised).
Overnight returns and annualisation are handled uniformly by
:mod:`corrforecast.realized.compute` so that estimators stay comparable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from corrforecast.realized.sampling import DayObservations


@dataclass
class DailyEstimate:
    cov: np.ndarray  # (N, N), NaN rows/cols for assets without usable data
    semicov_neg: np.ndarray | None = None  # (N, N) negative realized semicovariance, if provided
    diagnostics: dict[str, Any] = field(default_factory=dict)


class CovEstimator(ABC):
    """Base class. Subclasses declare ``name`` and accept their params as keyword arguments."""

    name: ClassVar[str]
    provides_semicov: ClassVar[bool] = False

    @abstractmethod
    def estimate(self, obs: DayObservations) -> DailyEstimate: ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


REGISTRY: dict[str, type[CovEstimator]] = {}


def register(cls: type[CovEstimator]) -> type[CovEstimator]:
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate estimator name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def build_estimator(name: str, params: dict[str, Any] | None = None) -> CovEstimator:
    # import for registration side effects
    from corrforecast.realized.estimators import bmn, kernel, preaveraging, qmle, realized_cov  # noqa: F401

    if name not in REGISTRY:
        raise KeyError(f"unknown estimator {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**(params or {}))


def nan_mask_cov(cov: np.ndarray, valid: np.ndarray) -> np.ndarray:
    cov = cov.copy()
    cov[~valid, :] = np.nan
    cov[:, ~valid] = np.nan
    return cov
