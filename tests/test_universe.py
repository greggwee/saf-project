import pandas as pd
import pytest

from saf.universe import (clip_segments, crsp_to_taq_symbol, databento_candidates,
                          get_monthly_universe, load_segments)

NAN = float("nan")


def _seg(**cols):
    base = {"permno": [1], "crsp_ticker": ["AAA"], "crsp_tradingsymbol": ["AAA"], "shareclass": [NAN],
            "valid_from": ["2016-01-01"], "valid_to": ["2016-12-31"],
            "member_from": ["2010-01-01"], "member_to": ["2020-12-31"]}
    base.update(cols)
    df = pd.DataFrame(base)
    for c in ["valid_from", "valid_to", "member_from", "member_to"]:
        df[c] = pd.to_datetime(df[c])
    return df


@pytest.mark.parametrize("args, expected", [
    (("BRK", "BRKB", "B"), ("BRK", "B")),
    (("AAPL", "AAPL", NAN), ("AAPL", "")),
    (("XYZ", NAN, NAN), ("XYZ", "")),
    (("GOOG", "GOOGL", "A"), ("GOOGL", "")),   # trading symbol is not ticker + class
])
def test_crsp_to_taq_symbol(args, expected):
    assert crsp_to_taq_symbol(*args) == expected


@pytest.mark.parametrize("args, expected", [
    (("BRK", "BRKB", "B"), ["BRK.B"]),              # manual override
    (("BF", "BFB", "B"), ["BF.B"]),
    (("FOX", "FOXA", "A"), ["FOX.A", "FOXA", "FOX"]),
    (("AAPL", "AAPL", NAN), ["AAPL"]),
    (("XYZ", NAN, NAN), ["XYZ"]),
])
def test_databento_candidates(args, expected):
    assert databento_candidates(*args) == expected


def test_load_segments_csv_types(tmp_path):
    p = tmp_path / "seg.csv"
    p.write_text("permno,gvkey,crsp_ticker,crsp_tradingsymbol,shareclass,valid_from,valid_to,member_from,member_to\n"
                 "10078,012136,SUNW,SUNW,,1992-08-20,2004-06-09,1992-08-20,2010-01-28\n")
    seg = load_segments(p)
    assert seg.loc[0, "gvkey"] == "012136"                 # leading zero kept
    assert seg.loc[0, "valid_to"] == pd.Timestamp("2004-06-09")
    assert pd.api.types.is_datetime64_any_dtype(seg["member_from"])


def test_clip_segments_window_is_half_open():
    seg = _seg(permno=[2, 1, 3], crsp_ticker=["B", "A", NAN], crsp_tradingsymbol=["B", "A", "C"],
               shareclass=[NAN] * 3, valid_from=["2016-03-01", "2015-01-01", "2016-01-01"],
               valid_to=["2016-03-31", "2016-06-30", "2016-12-31"],
               member_from=["2010-01-01"] * 3, member_to=["2020-01-01"] * 3)
    out = clip_segments(seg, pd.Timestamp("2016-01-01"), pd.Timestamp("2016-04-01"))
    assert out["permno"].tolist() == [1, 2]               # sorted, segment without ticker dropped
    assert out.loc[0, "req_start"] == pd.Timestamp("2016-01-01")   # clipped to lo
    assert out.loc[0, "req_end"] == pd.Timestamp("2016-04-01")     # clipped to hi
    assert out.loc[1, "req_end"] == pd.Timestamp("2016-04-01")     # valid_to + 1 day (exclusive)


def test_clip_segments_drops_empty_window():
    out = clip_segments(_seg(valid_to=["2015-12-31"]), pd.Timestamp("2016-01-01"), pd.Timestamp("2017-01-01"))
    assert out.empty


def test_get_monthly_universe_requires_validity_and_membership():
    seg = _seg(permno=[1, 2, 3], crsp_ticker=["A", "B", "C"], crsp_tradingsymbol=["A", "B", "C"],
               shareclass=[NAN] * 3,
               valid_from=["2016-06-30", "2016-01-01", "2016-01-01"],
               valid_to=["2016-12-31", "2016-05-31", "2016-12-31"],
               member_from=["2010-01-01", "2010-01-01", "2016-07-01"],
               member_to=["2020-01-01", "2020-01-01", "2020-01-01"])
    assert get_monthly_universe(seg, "2016-06")["permno"].tolist() == [1]
