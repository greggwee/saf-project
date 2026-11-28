"""Point-in-time S&P 500 universe: segment loading, monthly universe, symbol mapping.

A *segment* is one row of ``data/sp500_pit_identifiers_segments.csv``: a PERMNO over a period
during which it is an index member and its ticker/CUSIP are stable, so symbols are taken as-of.

Key columns used here:
    permno, crsp_ticker, crsp_tradingsymbol, shareclass  (identifiers)
    valid_from, valid_to                                 (identifier stability, inclusive)
    member_from, member_to                               (index membership, inclusive)
"""
from pathlib import Path

import pandas as pd

DATE_COLS = ["valid_from", "valid_to", "member_from", "member_to"]
STR_COLS = {"gvkey": str, "cusip8": str, "cusip9": str, "crsp_ticker": str,
            "crsp_tradingsymbol": str, "shareclass": str}

# Manual CRSP ticker -> Databento raw_symbol fixes (Nasdaq convention, e.g. BRK.B).
# Value = candidates tried in order.
SYMBOL_OVERRIDES = {
    "BRK": ["BRK.B"],   # CRSP: ticker BRK, tradingsymbol BRKB, share class B
    "BF": ["BF.B"],
}

_EXCEL_EPOCH = "1899-12-30"


# ---------------------------------------------------------------- segments
def load_segments(path):
    """Read the segments file (CSV or Excel) with identifiers as strings and parsed dates.

    Excel-serialized dates (day counts) are converted too. Rows are returned unfiltered.
    """
    path = Path(path)
    if path.suffix in (".xlsx", ".xls"):
        seg = pd.read_excel(path, dtype=STR_COLS)
    else:
        seg = pd.read_csv(path, dtype=STR_COLS)
    for col in DATE_COLS:
        if pd.api.types.is_numeric_dtype(seg[col]):
            seg[col] = pd.to_datetime(seg[col], unit="D", origin=_EXCEL_EPOCH)
        else:
            seg[col] = pd.to_datetime(seg[col])
    return seg


def clip_segments(seg, lo, hi):
    """Restrict segments to the request window ``[lo, hi)``.

    Adds ``req_start`` (inclusive) and ``req_end`` (exclusive, = valid_to + 1 day, capped at hi),
    drops segments without a CRSP ticker or with an empty window, sorted by (permno, req_start).
    """
    s = seg.dropna(subset=["crsp_ticker"]).copy()
    s["req_start"] = s["valid_from"].clip(lower=lo)
    s["req_end"] = (s["valid_to"] + pd.Timedelta(days=1)).clip(upper=hi)
    s = s[s["req_start"] < s["req_end"]]
    return s.sort_values(["permno", "req_start"]).reset_index(drop=True)


def get_monthly_universe(seg, month):
    """Segments valid and index members at some point during ``month`` (e.g. "2016-06").

    The whole month is downloaded; membership is re-applied at event date downstream.
    """
    month = pd.Period(month, freq="M")
    start, end = month.start_time.normalize(), month.end_time.normalize()
    mask = ((seg["valid_from"] <= end) & (seg["valid_to"] >= start)
            & (seg["member_from"] <= end) & (seg["member_to"] >= start))
    return seg.loc[mask].copy()


# ---------------------------------------------------------------- symbols
def _is_class_symbol(ticker, tradingsymbol, shareclass):
    """True when CRSP encodes a share class as ticker + class letter (BRK, BRKB, B)."""
    return (isinstance(ticker, str) and isinstance(tradingsymbol, str)
            and isinstance(shareclass, str) and tradingsymbol == f"{ticker}{shareclass}")


def crsp_to_taq_symbol(ticker, tradingsymbol, shareclass):
    """CRSP identifiers -> TAQ ``(sym_root, sym_suffix)``, e.g. (BRK, BRKB, B) -> ("BRK", "B").

    Fallback used when the WRDS CRSP-TAQM link table has no match.
    """
    if _is_class_symbol(ticker, tradingsymbol, shareclass):
        return ticker, shareclass
    sym = tradingsymbol if isinstance(tradingsymbol, str) else ticker
    return sym, ""


def databento_candidates(ticker, tradingsymbol, shareclass):
    """Ordered Databento raw_symbol candidates (Nasdaq convention) for a CRSP segment."""
    if ticker in SYMBOL_OVERRIDES:
        return list(SYMBOL_OVERRIDES[ticker])
    c = []
    if _is_class_symbol(ticker, tradingsymbol, shareclass):
        c.append(f"{ticker}.{shareclass}")       # BRK + B -> BRK.B
    for s in (tradingsymbol, ticker):
        if isinstance(s, str) and s and s not in c:
            c.append(s)
    return c
