"""Statistical checks of the daily covariance estimators on simulated noisy data."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from corrforecast.data.synthetic import SyntheticSpec, simulate
from corrforecast.realized.estimators import build_estimator
from corrforecast.realized.estimators.bmn import bai_ng_n_factors, bmn_covariance
from corrforecast.realized.estimators.preaveraging import mrc, preaverage_weights, prv
from corrforecast.realized.estimators.qmle import (
    dst_coefficients,
    qmle_batch,
    qmle_integrated_variance,
)
from corrforecast.realized.estimators.realized_cov import realized_cov_and_semicov
from corrforecast.realized.psd import cov_to_corr, nearest_correlation_psd
from corrforecast.realized.sampling import DayObservations


def test_realized_semicov_definition(rng) -> None:
    r = rng.normal(size=(27, 4))
    cov, semi = realized_cov_and_semicov(r)
    np.testing.assert_allclose(cov, r.T @ r)
    i, j = 0, 1
    expected = np.sum(r[:, i] * (r[:, i] < 0) * r[:, j] * (r[:, j] < 0))
    assert np.isclose(semi[i, j], expected)
    assert np.all(semi >= 0)
    c = cov_to_corr(semi)
    assert np.all((c >= 0) & (c <= 1))


def test_dst_diagonalises_ma1_covariance() -> None:
    n, v, w = 12, 0.7, 0.3
    T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    omega = v * np.eye(n) + w * T
    D = dst_coefficients(np.eye(n))  # columns: transform of unit vectors = eigenbasis
    lam = 2 - 2 * np.cos(np.pi * np.arange(1, n + 1) / (n + 1))
    np.testing.assert_allclose(D @ omega @ D.T, np.diag(v + w * lam), atol=1e-12)


def test_qmle_matches_numerical_optimum(rng) -> None:
    n = 300
    y = rng.normal(scale=0.01, size=n + 1)
    y = np.diff(y) + rng.normal(scale=0.003, size=n)
    z2 = dst_coefficients(y) ** 2
    lam = 2 - 2 * np.cos(np.pi * np.arange(1, n + 1) / (n + 1))

    def nll(th):
        L = np.exp(th[0]) + np.exp(th[1]) * lam
        return 0.5 * np.sum(np.log(L) + z2 / L)

    ref = minimize(
        nll,
        x0=np.log([1e-5, 1e-5]),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000},
    )
    v, w, ok = qmle_batch(z2[None, :], n)
    assert ok[0]
    assert nll(np.log([v[0], w[0]])) <= ref.fun + 1e-8


def test_qmle_ragged_batch_equals_individual(rng) -> None:
    series = [rng.normal(size=k) for k in (40, 75, 120)]
    zs = np.zeros((3, 120))
    for i, s in enumerate(series):
        zs[i, : len(s)] = dst_coefficients(s) ** 2
    v, _w, _ = qmle_batch(zs, np.array([40, 75, 120]))
    for i, s in enumerate(series):
        iv, _a2 = qmle_integrated_variance(s)
        assert np.isclose(iv, len(s) * v[i], rtol=1e-6)


def test_qmle_is_robust_to_noise(rng) -> None:
    n, iv_true, noise = 23400, 1e-4, 5e-4
    x = np.cumsum(rng.normal(scale=np.sqrt(iv_true / n), size=n + 1))
    y = np.diff(x + rng.normal(scale=noise, size=n + 1))
    iv, a2 = qmle_integrated_variance(y)
    assert np.sum(y**2) > 50 * iv_true  # naive RV is swamped by noise
    assert abs(iv / iv_true - 1) < 0.15
    assert abs(np.sqrt(a2) / noise - 1) < 0.05


def test_preaveraging_constants_converge() -> None:
    _, psi1, psi2 = preaverage_weights(400)
    assert abs(psi1 - 1.0) < 1e-3 and abs(psi2 - 1 / 12) < 1e-3


def test_prv_and_mrc_debias_noise(rng) -> None:
    n, p = 23400, 3
    sig = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]) * 1e-4
    x = np.vstack(
        [np.zeros(p), np.cumsum(rng.multivariate_normal(np.zeros(p), sig / n, size=n), axis=0)]
    )
    r = np.diff(x + rng.normal(scale=3e-4, size=x.shape), axis=0)
    est = mrc(r, theta=1.0)
    assert np.allclose(est, sig, atol=0.25e-4)
    np.testing.assert_allclose(prv(r, theta=1.0), np.diag(est))


def _noisy_day(seed: int, n_assets: int = 10):
    spec = SyntheticSpec(
        n_assets=n_assets,
        start="2001-01-02",
        end="2001-01-12",
        steps_per_day=2340,
        obs="poisson",
        mean_obs_per_day=900,
        noise_std=6e-4,
        noise_factor_std=4e-4,
        overnight_share=0.0,
        seed=seed,
    )
    d = simulate(spec)
    days = d["days"]
    out = []
    for t, day in enumerate(days):
        fr = d["intraday"][d["intraday"]["date"] == day]
        out.append((DayObservations.from_frame(day, fr, d["assets"]), d["true_cov"][t]))
    return out


@pytest.fixture(scope="module")
def noisy_days():
    return _noisy_day(3)


def _corr_mae(est_name, params, days) -> float:
    e = build_estimator(est_name, params)
    iu = np.triu_indices(days[0][1].shape[0], 1)
    errs = [
        np.mean(np.abs(cov_to_corr(e.estimate(o).cov)[iu] - cov_to_corr(tc)[iu])) for o, tc in days
    ]
    return float(np.mean(errs))


def test_denoisers_beat_naive_high_frequency_rcov(noisy_days) -> None:
    naive = _corr_mae("realized_cov", {"sampling": "10s"}, noisy_days)
    sparse = _corr_mae("realized_cov", {"sampling": "15min"}, noisy_days)
    qmle = _corr_mae("qmle_afx", {"sync": "pairwise_refresh"}, noisy_days)
    qmle_r = _corr_mae("qmle_afx", {"sync": "refresh"}, noisy_days)
    bmn = _corr_mae("bmn_factor", {"n_factors": "auto"}, noisy_days)
    assert qmle < naive and qmle_r < naive and bmn < naive
    assert qmle < sparse and bmn < sparse


def test_bmn_is_positive_definite(rng) -> None:
    r = rng.normal(size=(500, 30)) * 1e-3
    sig, diag = bmn_covariance(r, 3)
    assert np.linalg.eigvalsh(sig).min() > 0
    assert diag["n_factors"] == 3


def test_bai_ng_recovers_factor_count(rng) -> None:
    T, p, k = 400, 60, 3
    F = rng.normal(size=(T, k))
    B = rng.normal(size=(p, k))
    r = F @ B.T + 0.3 * rng.normal(size=(T, p))
    assert bai_ng_n_factors(r, 8) == k


def test_nearest_correlation_psd(rng) -> None:
    a = rng.uniform(-1, 1, size=(8, 8))
    c = (a + a.T) / 2
    np.fill_diagonal(c, 1.0)
    p = nearest_correlation_psd(c)
    assert np.linalg.eigvalsh(p).min() > -1e-10
    np.testing.assert_allclose(np.diag(p), 1.0)


def test_estimators_mark_missing_assets_nan() -> None:
    day = pd.Timestamp("2020-01-02")
    ts = pd.date_range(day + pd.Timedelta("09:30:00"), day + pd.Timedelta("16:00:00"), freq="30s")
    rng = np.random.default_rng(0)
    fr = pd.concat(
        [
            pd.DataFrame(
                {
                    "asset_id": a,
                    "timestamp": ts,
                    "price": np.exp(np.cumsum(rng.normal(0, 1e-3, len(ts)))),
                }
            )
            for a in (1, 2, 3)
        ]
    )
    obs = DayObservations.from_frame(day, fr, np.array([1, 2, 3, 4]))
    for name in ("realized_cov", "qmle_afx", "bmn_factor", "mrc", "realized_kernel"):
        cov = (
            build_estimator(name, {"n_factors": 1} if name == "bmn_factor" else {})
            .estimate(obs)
            .cov
        )
        assert np.all(np.isnan(cov[3])) and np.all(np.isfinite(cov[:3, :3])), name
