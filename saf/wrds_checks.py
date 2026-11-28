"""Sanity checks on the raw WRDS pull, written to ``data/meta/CHECKS.md`` (+ ``checks_monthly.csv``).

Pure functions on DataFrames (testable without WRDS) and :func:`run_checks` which reads the pull.
Interval convention: a start/end pair is inclusive and a missing end date means "still open".
"""
from pathlib import Path

import pandas as pd

from saf.wrds_pull import BY_NAME, meta_dir, read_table

SAMPLE_START = "2008-01-01"
MEMBERS_MIN, MEMBERS_MAX = 500, 505
NVDA_PERMNO = 86580
NVDA_SHROUT_2024_06_07 = 2_456_884          # thousands of shares
NVDA_SPLIT_DAY = "2024-06-10"               # 10-for-1: dlyfacprc = 10
FUNDQ_WINDOW = pd.Timedelta(days=366)       # a quarterly report within the last year
FUNDA_WINDOW = pd.Timedelta(days=731)       # an annual report within the last two years


def duplicate_keys(df, keys):
    """All rows whose ``keys`` appear more than once (duplicates are reported, not dropped)."""
    return df[df.duplicated(list(keys), keep=False)].sort_values(list(keys))


def month_ends(trading_days, start=SAMPLE_START, end=None):
    """Last trading day of each month in [start, end]."""
    d = pd.Series(pd.to_datetime(pd.unique(trading_days))).sort_values()
    d = d[(d >= pd.Timestamp(start)) & ((d <= pd.Timestamp(end)) if end is not None else True)]
    return pd.DatetimeIndex(d.groupby(d.dt.to_period("M")).max().values)


def _active(df, start_col, end_col, dates):
    """Cross join of interval rows with ``dates`` kept where start <= date <= end (end NaT = open)."""
    x = df.merge(pd.DataFrame({"date": dates}), how="cross")
    end = x[end_col].fillna(pd.Timestamp.max)
    return x[(x[start_col] <= x["date"]) & (x["date"] <= end)]


def members(msp, dates):
    """DataFrame [date, permno] of S&P 500 members at each date."""
    m = _active(msp[["permno", "mbrstartdt", "mbrenddt"]], "mbrstartdt", "mbrenddt", dates)
    return m[["date", "permno"]].drop_duplicates().reset_index(drop=True)


def members_missing_dsf(mem, dsf):
    """Members without a dsf row on that exact day: DataFrame [date, permno]."""
    have = dsf[["permno", "dlycaldt"]].rename(columns={"dlycaldt": "date"}).drop_duplicates()
    x = mem.merge(have, on=["date", "permno"], how="left", indicator=True)
    return x.loc[x["_merge"] == "left_only", ["date", "permno"]].reset_index(drop=True)


def member_links(mem, ccm):
    """Members with their CCM gvkey valid at that date: [date, permno, gvkey] (gvkey NaN = no link)."""
    lk = ccm.rename(columns={"lpermno": "permno"})[["permno", "gvkey", "linkdt", "linkenddt"]]
    lk = lk.dropna(subset=["permno"]).astype({"permno": "int64"})   # lpermno is float8 in CCM
    x = mem.merge(lk, on="permno", how="left")
    ok = (x["linkdt"] <= x["date"]) & (x["date"] <= x["linkenddt"].fillna(pd.Timestamp.max))
    linked = x[ok][["date", "permno", "gvkey"]].drop_duplicates(["date", "permno"])
    return mem.merge(linked, on=["date", "permno"], how="left")


def has_recent_report(ml, comp, window):
    """Boolean Series aligned on ``ml``: gvkey has a ``comp`` datadate in (date - window, date]."""
    x = ml.reset_index().dropna(subset=["gvkey"]).merge(comp[["gvkey", "datadate"]], on="gvkey")
    x = x[(x["datadate"] <= x["date"]) & (x["datadate"] > x["date"] - window)]
    return ml.index.isin(x["index"].unique())


def has_gics(ml, hgic):
    """Boolean Series aligned on ``ml``: gvkey has a co_hgic interval covering the date."""
    x = ml.reset_index().dropna(subset=["gvkey"]).merge(hgic[["gvkey", "indfrom", "indthru"]], on="gvkey")
    x = x[(x["indfrom"] <= x["date"]) & (x["date"] <= x["indthru"].fillna(pd.Timestamp.max))]
    return ml.index.isin(x["index"].unique())


def coverage_table(mem, dsf, ccm, fundq, funda, hgic):
    """Per month-end: members, members with dsf row, CCM link, recent fundq/funda, GICS."""
    ml = member_links(mem, ccm)
    ml["dsf"] = ~ml.set_index(["date", "permno"]).index.isin(
        members_missing_dsf(mem, dsf).set_index(["date", "permno"]).index)
    ml["link"] = ml["gvkey"].notna()
    ml["fundq"] = has_recent_report(ml, fundq, FUNDQ_WINDOW)
    ml["funda"] = has_recent_report(ml, funda, FUNDA_WINDOW)
    ml["gics"] = has_gics(ml, hgic)
    t = ml.groupby("date").agg(members=("permno", "size"), dsf=("dsf", "sum"), link=("link", "sum"),
                               fundq=("fundq", "sum"), funda=("funda", "sum"), gics=("gics", "sum"))
    return t.reset_index(), ml


def unlinked_members(ml):
    """Members without a CCM link: permno with first/last month-end and count."""
    u = ml[ml["gvkey"].isna()]
    return (u.groupby("permno")["date"].agg(first="min", last="max", months="size")
            .reset_index().sort_values("first"))


def nvda_units(dsf):
    """Unit checks on NVDA: shrout in thousands and dlyfacprc on the 2024 split."""
    n = dsf[dsf["permno"] == NVDA_PERMNO].set_index("dlycaldt")
    out = []
    d0 = pd.Timestamp("2024-06-07")
    if d0 in n.index:
        v = float(n.loc[d0, "shrout"])
        out.append(("NVDA shrout 2024-06-07 (thousands)", f"{v:,.0f}",
                    abs(v / NVDA_SHROUT_2024_06_07 - 1) < 0.01))
    else:
        out.append(("NVDA shrout 2024-06-07 (thousands)", "missing", False))
    split = pd.Timestamp(NVDA_SPLIT_DAY)
    if split in n.index:
        out.append((f"NVDA dlyfacprc {NVDA_SPLIT_DAY}", f"{n.loc[split, 'dlyfacprc']}",
                    float(n.loc[split, "dlyfacprc"]) == 10))
    else:
        out.append((f"NVDA dlyfacprc {NVDA_SPLIT_DAY}", "missing", False))
    normal = n.drop(index=[split], errors="ignore")["dlyfacprc"]
    out.append(("NVDA dlyfacprc on other days", f"{(normal != 1).sum()} days != 1",
                bool((normal.dropna() == 1).mean() > 0.99)))
    out.append(("All universe: share of days with dlyfacprc == 1",
                f"{(dsf['dlyfacprc'] == 1).mean():.4%}", bool((dsf["dlyfacprc"] == 1).mean() > 0.99)))
    return out


# ---------------------------------------------------------------- report
def _md(df):
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else
                                       (v.date().isoformat() if isinstance(v, pd.Timestamp) else str(v))
                                       for v in r) + " |")
    return "\n".join(lines) + "\n"


def run_checks(data_dir, end=None):
    """Run every check on the pull in ``data_dir``; write meta/CHECKS.md and meta/checks_monthly.csv."""
    out_dir = Path(data_dir)
    meta = meta_dir(out_dir)
    msp = read_table(out_dir, "crsp_msp500list_v2")
    ccm = read_table(out_dir, "crsp_ccmxpf_lnkhist")
    dsf = read_table(out_dir, "crsp_dsf_v2", columns=["permno", "dlycaldt", "shrout", "dlyfacprc"])
    msf_keys = read_table(out_dir, "crsp_msf_v2", columns=["permno", "mthcaldt"])
    funda = read_table(out_dir, "comp_funda", columns=["gvkey", "datadate"])
    fundq = read_table(out_dir, "comp_fundq", columns=["gvkey", "datadate"])
    hgic = read_table(out_dir, "comp_co_hgic")

    s = ["# Checks on the WRDS raw pull", ""]

    # 1. unique keys
    s += ["## 1. Unique keys (duplicates kept in the files)", ""]
    for name, df in [("crsp_dsf_v2", dsf), ("crsp_msf_v2", msf_keys), ("comp_funda", funda),
                     ("comp_fundq", fundq)]:
        keys = BY_NAME[name].keys
        dup = duplicate_keys(df[list(keys)], keys)
        s.append(f"- `{name}` ({', '.join(keys)}): {len(df):,} rows, {len(dup):,} rows in duplicated keys")
        if len(dup):
            s += ["", _md(dup.head(50))]
    s.append("")

    dates = month_ends(dsf["dlycaldt"], SAMPLE_START, end)
    mem = members(msp, dates)
    cov, ml = coverage_table(mem, dsf, ccm, fundq, funda, hgic)
    cov.to_csv(meta / "checks_monthly.csv", index=False)

    # 2. members per month-end
    bad = cov[(cov["members"] < MEMBERS_MIN) | (cov["members"] > MEMBERS_MAX)]
    s += [f"## 2. Members per month-end ({dates.min().date()} -> {dates.max().date()}, "
          f"{len(dates)} month-ends)", "",
          f"min {cov['members'].min()}, max {cov['members'].max()}; "
          f"{len(bad)} month-ends outside [{MEMBERS_MIN}, {MEMBERS_MAX}]", "", _md(bad[["date", "members"]])]

    # 3. dsf row for every member on every month-end
    miss = members_missing_dsf(mem, dsf)
    s += ["## 3. Members without a dsf_v2 row on the month-end", "",
          f"{len(miss)} (member, month-end) pairs missing", "", _md(miss.head(100))]

    # 4. CCM link + fundamentals coverage
    linked = cov["link"].sum()
    s += ["## 4. CCM link and Compustat coverage", "",
          f"- members linked: min {(cov['link'] / cov['members']).min():.2%} per month-end",
          f"- linked members with fundq datadate in the last year: "
          f"{cov['fundq'].sum() / linked:.2%} overall, min {(cov['fundq'] / cov['link']).min():.2%} per month-end",
          f"- linked members with funda datadate in the last two years: "
          f"{cov['funda'].sum() / linked:.2%} overall, min {(cov['funda'] / cov['link']).min():.2%} per month-end",
          "", "Members without a link:", "", _md(unlinked_members(ml))]

    # 5. GICS
    s += ["## 5. GICS coverage (co_hgic interval covering the month-end)", "",
          f"min {(cov['gics'] / cov['members']).min():.2%}, "
          f"overall {cov['gics'].sum() / cov['members'].sum():.2%} of members", ""]

    # 6. units
    s += ["## 6. Units", "", "| check | value | ok |", "|---|---|---|"]
    s += [f"| {a} | {b} | {'yes' if ok else 'NO'} |" for a, b, ok in nvda_units(dsf)]
    s += ["", "Per-month coverage table: `checks_monthly.csv`."]

    (meta / "CHECKS.md").write_text("\n".join(s) + "\n")
    return cov
