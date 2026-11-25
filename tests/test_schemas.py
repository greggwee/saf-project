import numpy as np
import pandas as pd
import pytest

from corrforecast.data.schemas import DAILY, INTRADAY, DataContractError, validate


def test_valid_intraday_passes() -> None:
    df = pd.DataFrame(
        {
            "asset_id": [1, 1],
            "timestamp": pd.to_datetime(["2020-01-02 09:30", "2020-01-02 09:31"]),
            "price": [10.0, 10.1],
        }
    )
    assert validate(df, INTRADAY) is df


def test_collects_all_problems() -> None:
    df = pd.DataFrame(
        {
            "asset_id": [1, 1],
            "timestamp": pd.to_datetime(["2020-01-02 09:30"] * 2),
            "price": [10.0, -1.0],
        }
    )
    with pytest.raises(DataContractError) as e:
        validate(df, INTRADAY)
    msg = str(e.value)
    assert "violate check" in msg and "duplicate" in msg


def test_missing_required_and_extra_columns() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "asset_id": [1]})
    with pytest.raises(DataContractError, match="ret"):
        validate(df, DAILY)
    df["ret"] = 0.01
    with pytest.raises(DataContractError, match="beta"):
        validate(df, DAILY, extra_required=("beta",))
    df["beta"] = np.nan
    validate(df, DAILY, extra_required=("beta",))
