"""End-to-end run of every stage on synthetic data that satisfies the data contract."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_config
from corrforecast.features.names import ALL_FEATURES
from corrforecast.features.panel import load_monthly_matrices, load_pairs
from corrforecast.models.rolling import load_predictions
from corrforecast.pipeline.run import run
from corrforecast.realized.psd import cov_to_corr
from corrforecast.realized.store import DailyMatrixStore


@pytest.fixture(scope="module")
def baseline_run(synthetic_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    cfg = make_config(synthetic_dir, tmp_path_factory.mktemp("runs"))
    run(cfg)
    return cfg


def test_daily_store_matches_direct_computation(baseline_run, synthetic_dir: Path) -> None:
    store = DailyMatrixStore(baseline_run.run_dir / "daily")
    dates, cov = store.load("cov")
    _, semi = store.load("semicov")
    assert cov.shape[1:] == (20, 20) and len(dates) == len(store.dates())
    # recompute day 10 by hand: 15-min grid (26 returns) + overnight, annualised
    intr = pd.read_parquet(synthetic_dir / "intraday")
    daily = pd.read_parquet(synthetic_dir / "daily.parquet")
    d = dates[10]
    px = intr[intr["timestamp"].dt.normalize() == d].pivot(
        index="timestamp", columns="asset_id", values="price"
    )
    r = np.diff(np.log(px.to_numpy()), axis=0)
    assert r.shape[0] == 26
    on = (
        daily[daily["date"] == d]
        .set_index("asset_id")["ret_overnight"]
        .reindex(px.columns)
        .to_numpy()
    )
    expected = 252 * (r.T @ r + np.outer(on, on))
    np.testing.assert_allclose(cov[10], expected, rtol=1e-4)
    rn, onn = np.minimum(r, 0), np.minimum(on, 0)
    np.testing.assert_allclose(
        semi[10], 252 * (rn.T @ rn + np.outer(onn, onn)), rtol=1e-4, atol=1e-10
    )


def test_feature_panel(baseline_run) -> None:
    fdir = baseline_run.run_dir / "features"
    panel = load_pairs(fdir)
    assert set(ALL_FEATURES) <= set(panel.columns)
    assert panel[list(ALL_FEATURES)].abs().max().max() <= 1.0 + 1e-6
    assert len(panel.groupby(["month"]).size().unique()) == 1  # balanced: 190 pairs every month
    assert (panel.groupby("month").size() == 20 * 19 // 2).all()
    # target(t) == realized monthly correlation of t+1
    mm = load_monthly_matrices(fdir)
    row = panel.iloc[1000]
    k = mm.index(row["target_month"])
    assert np.isclose(row["target"], mm.rc_m[k][row["i"], row["j"]], atol=1e-6)
    # RC_m(t) is the target of t-1
    k0 = mm.index(row["month"])
    assert np.isclose(row["RC_m"], mm.rc_m[k0][row["i"], row["j"]], atol=1e-6)
    # sector features are zero across sectors
    assert (panel.loc[~panel["same_sector"], "ExpScRC_d"] == 0).all()


def test_monthly_rc_is_corr_of_mean_daily_cov(baseline_run) -> None:
    store = DailyMatrixStore(baseline_run.run_dir / "daily")
    dates, cov = store.load("cov")
    mm = load_monthly_matrices(baseline_run.run_dir / "features")
    k = 20
    m = mm.months[k]
    sel = dates.to_period("M") == m.to_period("M")
    np.testing.assert_allclose(mm.rc_m[k], cov_to_corr(cov[sel].mean(0)), atol=1e-5)


def test_predictions_and_results(baseline_run) -> None:
    preds = load_predictions(baseline_run.run_dir)
    assert set(preds["target_month"].dt.year) == {2004, 2005}
    for m in ("HAR", "SHAR", "SHAR-F", "SHAR-F-Exp", "LASSO"):
        assert preds[m].between(-1, 1).all()
    res = baseline_run.run_dir / "results"
    for f in (
        "r2_oos",
        "dm_ew",
        "dm_vw",
        "feature_contributions_LASSO",
        "pairs_summary",
        "equity_premium",
        "risk_targeting",
        "gmv",
    ):
        assert (res / f"{f}.csv").exists(), f
    r2 = pd.read_csv(res / "r2_oos.csv")
    assert set(r2["model"]) == {"SHAR", "SHAR-F", "SHAR-F-Exp", "LASSO"}


def test_no_lookahead_in_rolling(baseline_run, tmp_path: Path) -> None:
    """Corrupting test-year targets must not change any fitted coefficient."""
    import shutil

    from corrforecast.models.rolling import run_rolling

    cfg = baseline_run
    clone = cfg.model_copy(update={"output_dir": tmp_path, "name": "clone"})
    shutil.copytree(cfg.run_dir / "features", clone.run_dir / "features")
    for p in (clone.run_dir / "features" / "pairs").glob("year=*/part.parquet"):
        df = pd.read_parquet(p)
        tm = (pd.to_datetime(df["month"]) + pd.offsets.MonthEnd(1)).dt.year
        df.loc[tm == 2005, "target"] = 0.99
        df.to_parquet(p, index=False)
    run_rolling(clone, test_years=[2005])
    a = pd.read_parquet(cfg.run_dir / "forecasts" / "coefficients.parquet")
    b = pd.read_parquet(clone.run_dir / "forecasts" / "coefficients.parquet")
    a = a[a["test_year"] == 2005].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[b.columns], b, check_exact=False, rtol=1e-10)


@pytest.mark.slow
def test_denoised_run_compare_and_target_swap(baseline_run, synthetic_dir: Path) -> None:
    from corrforecast.evaluation.compare import compare_runs

    out = baseline_run.output_dir
    cfg = make_config(
        synthetic_dir,
        out,
        name="qmle",
        realized={"estimator": {"name": "qmle_afx", "params": {"sync": "refresh"}}},
        models={"target_run": baseline_run.name},
    )
    run(cfg, ("realized", "features", "forecast", "evaluate"))
    assert (cfg.run_dir / "results" / "r2_oos.csv").exists()
    # with target_run, this run's targets are the baseline run's realized correlations
    a = load_predictions(cfg.run_dir).set_index(["month", "i", "j"])["target"]
    b = load_predictions(baseline_run.run_dir).set_index(["month", "i", "j"])["target"]
    pd.testing.assert_series_equal(
        a.sort_index(), b.reindex(a.index).sort_index(), check_dtype=False
    )
    cmp = compare_runs(out, baseline_run.name, [baseline_run.name, "qmle"], models=["HAR", "LASSO"])
    assert set(cmp["run"]) == {baseline_run.name, "qmle"} and len(cmp) == 4
    har_self = cmp[(cmp["run"] == baseline_run.name) & (cmp["model"] == "HAR")]
    assert np.isclose(har_self["R2_EW_%"].iloc[0], 0.0)
