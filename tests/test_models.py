import numpy as np

from corrforecast.config import ModelConfig
from corrforecast.models.estimators import make_model
from corrforecast.models.rolling import make_splits


def _data(rng, n=4000, p=10):
    X = rng.normal(size=(n, p)) * rng.uniform(0.1, 3, p) + rng.normal(size=p)
    beta = np.zeros(p)
    beta[[0, 3]] = [0.5, -0.2]
    y = 0.1 + X @ beta + rng.normal(scale=0.5, size=n)
    return X, y, beta


def test_ols_matches_lstsq(rng) -> None:
    X, y, _ = _data(rng)
    m = make_model("ols").fit(X, y)
    ref, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)
    np.testing.assert_allclose(m.result.coef, ref[1:], atol=1e-10)
    assert np.isclose(m.result.intercept, ref[0])


def test_lasso_selects_sparse_truth_on_validation(rng) -> None:
    X, y, beta = _data(rng)
    Xv, yv, _ = _data(np.random.default_rng(99))
    yv = 0.1 + Xv @ beta + np.random.default_rng(1).normal(scale=0.5, size=len(yv))
    m = make_model("lasso", n_alphas=50, alpha_min_ratio=1e-4).fit(X, y, Xv, yv)
    sel = np.flatnonzero(m.result.coef_std)
    assert {0, 3} <= set(sel)
    np.testing.assert_allclose(m.result.coef[[0, 3]], beta[[0, 3]], atol=0.05)
    # standardisation uses training moments only: predictions are affine in X
    assert np.allclose(m.predict(X[:5]), m.result.intercept + X[:5] @ m.result.coef)


def test_other_methods_run(rng) -> None:
    X, y, _ = _data(rng, n=800)
    for meth in ("ridge", "enet", "pcr"):
        m = make_model(meth, n_alphas=20).fit(X[:600], y[:600], X[600:], y[600:])
        assert np.isfinite(m.predict(X)).all()


def test_splits_follow_paper_and_never_peek() -> None:
    splits = make_splits(ModelConfig(first_test_year=2008, last_test_year=2020))
    assert len(splits) == 13
    s = splits[0]
    assert (
        s.test_year == 2008 and s.val_years == (2007,) and s.train_years == (2003, 2004, 2005, 2006)
    )
    assert s.ols_years == (2003, 2004, 2005, 2006, 2007)
    for s in splits:
        assert max(s.ols_years) < s.test_year
        assert not set(s.train_years) & set(s.val_years)
