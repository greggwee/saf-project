import pandas as pd

from saf import wrds_checks as wc

T = pd.Timestamp


def test_month_ends_uses_last_trading_day():
    days = pd.to_datetime(["2008-01-30", "2008-01-31", "2008-02-28", "2008-02-29", "2008-03-28"])
    assert list(wc.month_ends(days, "2008-01-01")) == [T("2008-01-31"), T("2008-02-29"), T("2008-03-28")]


def test_members_open_end_and_bounds():
    msp = pd.DataFrame({"permno": [1, 2, 3], "mbrstartdt": [T("2000-01-01"), T("2008-02-01"), T("2000-01-01")],
                        "mbrenddt": [pd.NaT, T("2030-01-01"), T("2008-01-31")]})
    m = wc.members(msp, pd.DatetimeIndex([T("2008-01-31"), T("2008-02-29")]))
    assert m.groupby("date")["permno"].apply(sorted).tolist() == [[1, 3], [1, 2]]


def test_duplicates_are_reported():
    df = pd.DataFrame({"gvkey": ["a", "a", "b"], "datadate": [T("2020-03-31")] * 3})
    assert len(wc.duplicate_keys(df, ("gvkey", "datadate"))) == 2


def _fixture():
    d = T("2020-01-31")
    mem = pd.DataFrame({"date": [d, d, d], "permno": [1, 2, 3]})
    dsf = pd.DataFrame({"permno": [1, 2], "dlycaldt": [d, d], "shrout": [1.0, 1.0], "dlyfacprc": [1.0, 1.0]})
    ccm = pd.DataFrame({"gvkey": ["g1", "g2", "g2"], "lpermno": [1.0, 2.0, 2.0],   # float8 on WRDS
                        "linkdt": [T("2000-01-01"), T("2000-01-01"), T("2019-01-01")],
                        "linkenddt": [pd.NaT, T("2010-01-01"), pd.NaT]})
    fundq = pd.DataFrame({"gvkey": ["g1", "g2"], "datadate": [T("2019-12-31"), T("2018-12-31")]})
    funda = pd.DataFrame({"gvkey": ["g1", "g2"], "datadate": [T("2019-12-31"), T("2018-12-31")]})
    hgic = pd.DataFrame({"gvkey": ["g1"], "indfrom": [T("2000-01-01")], "indthru": [pd.NaT]})
    return mem, dsf, ccm, fundq, funda, hgic


def test_coverage_table():
    cov, ml = wc.coverage_table(*_fixture())
    r = cov.iloc[0]
    # permno 3: no dsf row, no link; permno 2 linked via its second link; old fundq, recent funda
    assert (r.members, r.dsf, r.link, r.fundq, r.funda, r.gics) == (3, 2, 2, 1, 2, 1)
    assert wc.unlinked_members(ml)["permno"].tolist() == [3]


def test_nvda_units():
    days = pd.bdate_range("2024-06-03", "2024-06-14")
    dsf = pd.DataFrame({"permno": wc.NVDA_PERMNO, "dlycaldt": days, "shrout": 2_456_884.0, "dlyfacprc": 1.0})
    dsf.loc[dsf["dlycaldt"] == T("2024-06-10"), "dlyfacprc"] = 10.0
    res = wc.nvda_units(dsf)
    assert [ok for _, _, ok in res[:3]] == [True, True, True]
