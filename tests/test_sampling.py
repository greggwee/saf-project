import itertools

import numpy as np
import pandas as pd

from corrforecast.realized.sampling import (
    DayObservations,
    grid_returns,
    previous_tick,
    refresh_time_prices,
    refresh_times,
)


def test_previous_tick_backfills_and_holds() -> None:
    t = np.array([10.0, 20.0, 30.0])
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        previous_tick(t, v, np.array([0.0, 10.0, 15.0, 30.0, 99.0])), [1, 1, 1, 3, 3]
    )


def test_refresh_times_definition(rng) -> None:
    for _ in range(50):
        ts = [
            np.unique(rng.uniform(0, 1000, rng.integers(5, 60))) for _ in range(rng.integers(2, 12))
        ]
        tau = refresh_times(ts)
        assert tau[0] == max(t[0] for t in ts)
        # every asset trades in every interval (tau_{j-1}, tau_j]
        for a, b in itertools.pairwise(tau):
            for t in ts:
                assert np.any((t > a) & (t <= b))
        # and the next refresh time would not exist
        assert any(np.searchsorted(t, tau[-1], side="right") >= len(t) for t in ts)


def test_refresh_time_prices_shape(rng) -> None:
    ts = [np.sort(rng.uniform(0, 100, 30)) for _ in range(3)]
    lp = [np.cumsum(rng.normal(size=30)) for _ in range(3)]
    tau, p = refresh_time_prices(ts, lp)
    assert p.shape == (len(tau), 3)


def test_grid_returns_15min_gives_26_intervals() -> None:
    day = pd.Timestamp("2020-01-02")
    ts = pd.date_range(day + pd.Timedelta("09:30:00"), day + pd.Timedelta("16:00:00"), freq="1min")
    frame = pd.DataFrame(
        {"asset_id": 1, "timestamp": ts, "price": np.exp(np.linspace(0, 0.01, len(ts)))}
    )
    obs = DayObservations.from_frame(day, frame, np.array([1, 2]))
    r = grid_returns(obs, 900)
    assert r.shape == (26, 2)
    assert np.isclose(np.nansum(r[:, 0]), 0.01)
    assert np.all(np.isnan(r[:, 1]))
