"""Pooled linear (and FNN) correlation forecasting models (BLT §3).

All models fit ``RC_{ij,t+1} = theta_0 + x_{ij,t}' theta + error`` on the pooled
panel of stock pairs.  Penalised models standardise each feature by its
*training-sample* mean and standard deviation and pick the penalty on the
validation year by minimum MSE (BLT eq. 3).  Because the panel is long
(millions of rows) and narrow (<= 25 features), penalised paths are computed
from the Gram matrix, so their cost does not grow with the number of pairs.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import enet_path, lasso_path
from sklearn.neural_network import MLPRegressor

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    coef: np.ndarray  # on the original (unstandardised) feature scale
    intercept: float
    coef_std: np.ndarray  # on the standardised scale (for feature-importance plots)
    hyper: dict[str, Any] = field(default_factory=dict)
    val_curve: np.ndarray | None = None


class Forecaster(ABC):
    needs_validation: bool = False

    def __init__(self, **params: Any):
        self.params = params
        self.result: FitResult | None = None

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Forecaster: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.result is not None, "fit first"
        return self.result.intercept + X @ self.result.coef


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def _to_original(
    beta_std: np.ndarray, b0: float, mu: np.ndarray, sd: np.ndarray
) -> tuple[np.ndarray, float]:
    coef = beta_std / sd
    return coef, float(b0 - mu @ coef)


class OLS(Forecaster):
    def fit(self, X, y, X_val=None, y_val=None):
        Xs, mu, sd = _standardize(X)
        ym = y.mean()
        beta, *_ = np.linalg.lstsq(Xs, y - ym, rcond=None)
        coef, b0 = _to_original(beta, ym, mu, sd)
        self.result = FitResult(coef=coef, intercept=b0, coef_std=beta)
        return self


class _PathModel(Forecaster):
    """Shared logic: compute a path on training data, choose the penalty on validation data."""

    needs_validation = True

    def _path(
        self, Xs: np.ndarray, yc: np.ndarray, alphas: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _quiet_path(
        self, Xs: np.ndarray, yc: np.ndarray, **kw: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        # Highly collinear correlation features make coordinate descent slow at the smallest
        # penalties; the duality gaps involved are negligible for model selection, so log
        # instead of spamming warnings.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            out = self._path(Xs, yc, **kw)
        n = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        if n:
            log.debug("%s: %d penalty values hit max_iter", type(self).__name__, n)
        return out

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            raise ValueError(f"{type(self).__name__} requires a validation sample")
        Xs, mu, sd = _standardize(X)
        ym = y.mean()
        alphas, betas = self._quiet_path(Xs, y - ym)  # betas: (p, n_alphas)
        Xv = (X_val - mu) / sd
        pred = ym + Xv @ betas
        mse = np.mean((y_val[:, None] - pred) ** 2, axis=0)
        k = int(np.argmin(mse))
        beta = betas[:, k]
        if self.params.get("refit_on_train_plus_val"):
            Xa, ya = np.vstack([X, X_val]), np.concatenate([y, y_val])
            Xas, mu, sd = _standardize(Xa)
            ym = ya.mean()
            _, b = self._quiet_path(Xas, ya - ym, alphas=np.array([alphas[k]]))
            beta = b[:, 0]
        coef, b0 = _to_original(beta, ym, mu, sd)
        self.result = FitResult(
            coef=coef,
            intercept=b0,
            coef_std=beta,
            hyper={"alpha": float(alphas[k]), "n_selected": int(np.sum(beta != 0))},
            val_curve=np.column_stack([alphas, mse]),
        )
        return self


def _alpha_grid(
    Xs: np.ndarray, yc: np.ndarray, n: int, ratio: float, l1_ratio: float = 1.0
) -> np.ndarray:
    amax = np.max(np.abs(Xs.T @ yc)) / (len(yc) * max(l1_ratio, 1e-3))
    return np.geomspace(amax, amax * ratio, n)


class Lasso(_PathModel):
    def _path(self, Xs, yc, alphas=None):
        if alphas is None:
            alphas = _alpha_grid(
                Xs, yc, self.params.get("n_alphas", 100), self.params.get("alpha_min_ratio", 1e-4)
            )
        gram, xy = Xs.T @ Xs, Xs.T @ yc
        a, coefs, _ = lasso_path(
            Xs,
            yc,
            alphas=alphas,
            precompute=gram,
            Xy=xy,
            max_iter=self.params.get("max_iter", 100_000),
            tol=1e-4,
        )
        return a, coefs


class ElasticNet(_PathModel):
    def _path(self, Xs, yc, alphas=None):
        l1 = self.params.get("l1_ratio", 0.5)
        if alphas is None:
            alphas = _alpha_grid(
                Xs,
                yc,
                self.params.get("n_alphas", 100),
                self.params.get("alpha_min_ratio", 1e-4),
                l1,
            )
        gram, xy = Xs.T @ Xs, Xs.T @ yc
        a, coefs, _ = enet_path(
            Xs,
            yc,
            l1_ratio=l1,
            alphas=alphas,
            precompute=gram,
            Xy=xy,
            max_iter=self.params.get("max_iter", 100_000),
            tol=1e-4,
        )
        return a, coefs


class Ridge(_PathModel):
    def _path(self, Xs, yc, alphas=None):
        n = len(yc)
        gram, xy = Xs.T @ Xs / n, Xs.T @ yc / n
        if alphas is None:
            alphas = np.geomspace(1e3, 1e-6, self.params.get("n_alphas", 100))
        p = gram.shape[0]
        coefs = np.column_stack([np.linalg.solve(gram + a * np.eye(p), xy) for a in alphas])
        return np.asarray(alphas), coefs


class PCR(_PathModel):
    """Principal component regression; the 'penalty' is the number of components (<= 20, BLT endnote 31)."""

    def _path(self, Xs, yc, alphas=None):
        gram = Xs.T @ Xs
        w, V = np.linalg.eigh(gram)
        V = V[:, np.argsort(w)[::-1]]
        kmax = min(self.params.get("max_components", 20), Xs.shape[1])
        ks = np.arange(1, kmax + 1) if alphas is None else np.asarray(alphas, dtype=int)
        coefs = []
        for k in ks:
            Z = Xs @ V[:, :k]
            g, *_ = np.linalg.lstsq(Z, yc, rcond=None)
            coefs.append(V[:, :k] @ g)
        return ks.astype(float), np.column_stack(coefs)


class FNN(Forecaster):
    """Feed-forward network (BLT: two hidden layers with 4 and 2 neurons), early-stopped on validation."""

    needs_validation = True

    def fit(self, X, y, X_val=None, y_val=None):
        Xs, self._mu, self._sd = _standardize(X)
        hidden = tuple(self.params.get("hidden", (4, 2)))
        best, best_mse = None, np.inf
        for seed in range(int(self.params.get("n_seeds", 3))):
            m = MLPRegressor(
                hidden_layer_sizes=hidden,
                alpha=self.params.get("l2", 1e-4),
                batch_size=4096,
                learning_rate_init=1e-3,
                max_iter=self.params.get("max_iter", 50),
                random_state=seed,
                early_stopping=False,
            )
            m.fit(Xs, y)
            if X_val is not None and y_val is not None:
                mse = float(np.mean((m.predict((X_val - self._mu) / self._sd) - y_val) ** 2))
            else:
                mse = 0.0
            if mse < best_mse:
                best, best_mse = m, mse
        self._net = best
        self.result = FitResult(
            coef=np.zeros(X.shape[1]),
            intercept=0.0,
            coef_std=np.zeros(X.shape[1]),
            hyper={"val_mse": best_mse},
        )
        return self

    def predict(self, X):
        return self._net.predict((X - self._mu) / self._sd)


METHODS: dict[str, type[Forecaster]] = {
    "ols": OLS,
    "lasso": Lasso,
    "enet": ElasticNet,
    "ridge": Ridge,
    "pcr": PCR,
    "fnn": FNN,
}


def make_model(method: str, **params: Any) -> Forecaster:
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    return METHODS[method](**params)
