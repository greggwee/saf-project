import numpy as np
import pandas as pd

from corrforecast.applications.common import capped_value_weights, utility_gain
from corrforecast.applications.portfolios import gmv_weights, long_short_weights
from corrforecast.evaluation.metrics import dm_test, r2_oos, value_weights


def _panel(rng, months=6, n=8):
    i, j = np.triu_indices(n, 1)
    rows = []
    for t in pd.date_range("2010-01-31", periods=months, freq="ME"):
        y = rng.uniform(0, 0.6, len(i))
        rows.append(
            pd.DataFrame(
                {
                    "month": t,
                    "i": i,
                    "j": j,
                    "target": y,
                    "good": y + rng.normal(0, 0.05, len(i)),
                    "bad": y + rng.normal(0, 0.2, len(i)),
                    "cap_i": rng.uniform(1, 100, len(i)),
                    "cap_j": rng.uniform(1, 100, len(i)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_r2_oos_basic(rng) -> None:
    y = rng.normal(size=100)
    b = y + rng.normal(size=100)
    assert r2_oos(y, b, b) == 0
    assert r2_oos(y, y, b) == 1


def test_value_weights_sum_to_one_per_month(rng) -> None:
    df = _panel(rng)
    w = value_weights(df)
    np.testing.assert_allclose(w.groupby(df["month"]).sum(), 1.0)
    assert (w > 0).all()


def test_dm_sign_and_antisymmetry(rng) -> None:
    df = _panel(rng)
    t = dm_test(df, "good", "bad")
    assert t > 2
    assert np.isclose(dm_test(df, "bad", "good"), -t)
    assert dm_test(df, "good", "bad", value_weights(df)) > 0


def test_gmv_weights(rng) -> None:
    A = rng.normal(size=(6, 6))
    S = A @ A.T + np.eye(6)
    w = gmv_weights(S)
    assert np.isclose(w.sum(), 1)
    # first-order condition: S w proportional to 1
    g = S @ w
    assert np.allclose(g, g[0])
    m = rng.uniform(0, 1, 6)
    m /= m.sum()
    wb = gmv_weights(S, m)
    assert np.isclose(wb.sum(), 1) and np.isclose(wb @ S @ m, 0, atol=1e-12)


def test_long_short_and_caps(rng) -> None:
    w = long_short_weights(rng.normal(size=50), 10)
    assert np.isclose(w.sum(), 0) and np.isclose(w[w > 0].sum(), 1)
    cw = capped_value_weights(np.array([100.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 0.10)
    assert np.isclose(cw.sum(), 1) and cw.max() <= 0.10 + 1e-12


def test_utility_gain_zero_for_identical(rng) -> None:
    r = rng.normal(0.008, 0.04, 120)
    assert abs(utility_gain(r, r, 5)) < 1e-10
    assert utility_gain(r, r + 0.001, 5) > 0
