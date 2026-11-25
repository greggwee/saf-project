import numpy as np
import pandas as pd

from corrforecast.features.names import ALL_FEATURES, FEATURE_SETS
from corrforecast.features.transforms import (
    exp_weights,
    factor_projection,
    loading_matrix,
    rank_to_unit,
    sector_average,
)


def test_25_features_and_nested_sets() -> None:
    assert len(ALL_FEATURES) == 25 and len(set(ALL_FEATURES)) == 25
    assert [len(FEATURE_SETS[k]) for k in ("har", "shar", "shar_f", "all")] == [3, 6, 9, 25]
    assert (
        set(FEATURE_SETS["har"])
        < set(FEATURE_SETS["shar"])
        < set(FEATURE_SETS["shar_f"])
        < set(FEATURE_SETS["all"])
    )


def test_exp_weights_center_of_mass() -> None:
    for com in (1, 5, 21, 63):
        w = exp_weights(com, 5000)
        assert np.isclose(w.sum(), 1)
        assert np.isclose(np.sum(np.arange(5000) * w), com, rtol=1e-6)  # lags 0,1,2,...


def test_factor_projection_keeps_diagonal_and_is_low_rank(rng) -> None:
    N, K = 30, 4
    A = rng.normal(size=(N, N))
    cov = A @ A.T
    L = rng.normal(size=(N, K))
    f = factor_projection(cov, L)
    np.testing.assert_allclose(np.diag(f), np.diag(cov))
    off = f - np.diag(np.diag(f))
    P = L @ np.linalg.pinv(L.T @ L) @ L.T
    np.testing.assert_allclose(off, (P @ cov @ P) - np.diag(np.diag(P @ cov @ P)), atol=1e-9)
    assert np.linalg.matrix_rank(P @ cov @ P) == K
    # stacked input works too
    np.testing.assert_allclose(factor_projection(np.stack([cov, cov]), L)[1], f)


def test_rank_transform_and_loading_matrix() -> None:
    x = np.array([3.0, np.nan, 1.0, 2.0])
    np.testing.assert_allclose(rank_to_unit(x), [1.0, 0.0, -1.0, 0.0])
    L = loading_matrix(
        pd.DataFrame({"beta": [1.2, 0.8, np.nan], "size": [5, 1, 3]}), ("beta",), add_intercept=True
    )
    np.testing.assert_allclose(L, [[1, 1.2, 1], [1, 0.8, -1], [1, 1.0, 0]])


def test_sector_average() -> None:
    c = np.array([[1, 0.2, 0.4, 0.9], [0.2, 1, 0.6, 0.9], [0.4, 0.6, 1, 0.9], [0.9, 0.9, 0.9, 1]])
    s = sector_average(c, np.array(["a", "a", "a", "b"]))
    assert np.isclose(s[0, 1], 0.4) and np.isclose(s[1, 2], 0.4)
    assert s[0, 3] == 0 and s[3, 3] == 1
