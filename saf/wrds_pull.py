"""Raw WRDS pull for feature engineering (everything except TAQ intraday data).

Each table is described by a :class:`TableSpec`. The SELECT list is built from the live schema
(``describe_table``) so that requested columns are checked before any data is pulled and NUMERIC
columns are cast to float8 in SQL (otherwise pandas receives ``Decimal`` objects).

Output under ``data_dir`` (the repo's ``data/`` folder), one subfolder per source:
``crsp/``, ``compustat/``, ``ff/`` hold ``<name>.parquet`` or ``<name>_<YYYY>.parquet`` (zstd) for
yearly tables; ``meta/`` holds ``manifest.csv`` (file, rows, seconds, pulled_utc, sql),
``_schemas.txt``, ``README.md`` and the checks. Existing files are skipped, so an interrupted run
resumes where it stopped.
"""
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Universe = every S&P 500 member still in the index on or after this date (two years of
# look-back before the 2008-01 feature sample, for the 500-day realized windows).
UNIVERSE_FROM = "2006-01-01"
UNIVERSE_PERMNOS = f"SELECT DISTINCT permno FROM crspq.msp500list_v2 WHERE mbrenddt >= '{UNIVERSE_FROM}'"
# LS and J are kept: LS links Eaton and Dayforce; J (secondary share class: GOOG, NWS, FOX, UA, ...)
# links index members whose primary class is a separate permno of the same company.
CCM_FILTER = "linktype IN ('LU', 'LC', 'LS') AND linkprim IN ('P', 'C', 'J')"
UNIVERSE_GVKEYS = (f"SELECT DISTINCT gvkey FROM crspq.ccmxpf_lnkhist "
                   f"WHERE {CCM_FILTER} AND lpermno IN ({UNIVERSE_PERMNOS})")
COMP_FILTER = "indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'"
COMMON_STOCK = "sharetype = 'NS' AND securitytype = 'EQTY' AND securitysubtype = 'COM'"

SOURCE_DIRS = {"crspq": "crsp", "crsp": "crsp", "comp": "compustat", "ff": "ff"}
META_DIR = "meta"           # manifest, schemas, README and checks of the pull
FRED_DIR = "fred"

_FLOAT_TYPES = ("NUMERIC", "DECIMAL")
_DATE_TYPES = ("DATE", "TIMESTAMP")


@dataclass(frozen=True)
class TableSpec:
    """One WRDS table to pull.

    columns  : requested columns, ``None`` = every column of the table
    where    : extra SQL conditions, AND-ed together
    date_col : column used for the period filter and yearly chunks
    start    : first date kept (inclusive), ISO string
    to_end   : also cap ``date_col`` at the run's end date ("-> latest" in the spec)
    yearly   : one file per calendar year
    keys     : expected unique key (checked, duplicates are kept)
    """
    name: str
    lib: str
    table: str
    columns: tuple = None
    where: tuple = ()
    date_col: str = None
    start: str = None
    to_end: bool = False
    yearly: bool = False
    keys: tuple = ()

    @property
    def folder(self):
        """Subfolder of ``data/`` for this table's source (crsp, compustat, ff)."""
        return SOURCE_DIRS.get(self.lib, self.lib)


TABLES = (
    TableSpec("crsp_msp500list_v2", "crspq", "msp500list_v2",
              ("permno", "mbrstartdt", "mbrenddt", "mbrflg")),
    TableSpec("crsp_stksecurityinfohist", "crspq", "stksecurityinfohist", None,
              where=(f"permno IN ({UNIVERSE_PERMNOS})",)),
    TableSpec("crsp_ccmxpf_lnkhist", "crspq", "ccmxpf_lnkhist",
              ("gvkey", "lpermno", "lpermco", "linktype", "linkprim", "linkdt", "linkenddt"),
              where=(CCM_FILTER,)),
    TableSpec("crsp_dsf_v2", "crspq", "dsf_v2",
              ("permno", "permco", "dlycaldt", "dlyret", "dlyretx", "dlyprc", "dlyvol", "shrout",
               "dlycap", "dlyfacprc", "dlycumfacpr", "dlycumfacshr", "sharetype", "securitytype",
               "securitysubtype", "primaryexch", "siccd", "ticker", "cusip"),
              where=(f"permno IN ({UNIVERSE_PERMNOS})",),
              date_col="dlycaldt", start="2004-01-01", to_end=True, yearly=True,
              keys=("permno", "dlycaldt")),
    # All common stocks (not only the universe): indmom and chpmia need industry averages.
    TableSpec("crsp_msf_v2", "crspq", "msf_v2",
              ("permno", "permco", "mthcaldt", "mthret", "mthretx", "mthprc", "shrout", "mthcap",
               "siccd", "primaryexch", "issuertype", "usincflg"),
              where=(COMMON_STOCK,),
              date_col="mthcaldt", start="1999-01-01", to_end=True, yearly=True,
              keys=("permno", "mthcaldt")),
    TableSpec("crsp_stkdelists", "crspq", "stkdelists", None),
    TableSpec("crsp_dsp500_v2", "crspq", "dsp500_v2",
              ("caldt", "sprtrn", "vwretd", "totval", "totcnt"),
              date_col="caldt", start="2004-01-01", to_end=True),
    TableSpec("ff_factors_daily", "ff", "factors_daily",
              ("date", "mktrf", "smb", "hml", "umd", "rf"),
              date_col="date", start="1999-01-01", to_end=True),
    TableSpec("ff_factors_monthly", "ff", "factors_monthly",
              ("date", "mktrf", "smb", "hml", "umd", "rf"),
              date_col="date", start="1999-01-01", to_end=True),
    # All firms: eg needs cross-sectional regressions, chpmia industry averages.
    TableSpec("comp_funda", "comp", "funda",
              ("gvkey", "datadate", "fyear", "fyr", "curcd", "sich", "at", "act", "che", "lct", "dlc",
               "dltt", "txp", "dp", "lt", "ceq", "pstk", "mib", "revt", "sale", "cogs", "ib", "ni",
               "pi", "ppegt", "invt", "capx", "csho", "ajex", "emp"),
              where=(COMP_FILTER,), date_col="datadate", start="1995-01-01",
              keys=("gvkey", "datadate")),
    TableSpec("comp_fundq", "comp", "fundq",
              ("gvkey", "datadate", "fyearq", "fqtr", "rdq", "ibq", "niq", "atq", "ltq", "cheq", "ceqq",
               "saleq", "cshoq", "prccq"),
              where=(COMP_FILTER, f"gvkey IN ({UNIVERSE_GVKEYS})"),
              date_col="datadate", start="2004-01-01", keys=("gvkey", "datadate")),
    TableSpec("comp_co_hgic", "comp", "co_hgic",
              ("gvkey", "indfrom", "indthru", "gsector", "ggroup", "gind", "gsubind")),
)
BY_NAME = {t.name: t for t in TABLES}


# ---------------------------------------------------------------- SQL
def _type_names(schema):
    """{column: upper-case type string} from a describe_table DataFrame."""
    return {str(n): str(t).upper() for n, t in zip(schema["name"], schema["type"])}


def select_list(spec, schema):
    """SELECT expressions for ``spec`` against the live ``schema``; NUMERIC cast to float8.

    Raises ValueError listing every requested column missing from the table.
    """
    types = _type_names(schema)
    cols = list(spec.columns) if spec.columns is not None else list(types)
    missing = [c for c in cols if c not in types]
    if missing:
        raise ValueError(f"{spec.lib}.{spec.table}: columns not found {missing}; "
                         f"available: {sorted(types)}")
    return [f"{c}::float8 AS {c}" if types[c].startswith(_FLOAT_TYPES) else c for c in cols]


def date_columns(spec, schema):
    """Selected columns with a DATE/TIMESTAMP type (converted to datetime64 after the query)."""
    types = _type_names(schema)
    cols = spec.columns if spec.columns is not None else list(types)
    return [c for c in cols if types.get(c, "").startswith(_DATE_TYPES)]


def _iso(d):
    return pd.Timestamp(d).date().isoformat()


def build_query(spec, schema, lo=None, hi=None):
    """Full SQL for ``spec`` restricted to ``lo <= date_col <= hi`` (either bound optional)."""
    where = list(spec.where)
    if spec.date_col and lo is not None:
        where.append(f"{spec.date_col} >= '{_iso(lo)}'")
    if spec.date_col and hi is not None:
        where.append(f"{spec.date_col} <= '{_iso(hi)}'")
    q = f"SELECT {', '.join(select_list(spec, schema))} FROM {spec.lib}.{spec.table}"
    if where:
        q += " WHERE " + " AND ".join(f"({w})" for w in where)
    if spec.keys:
        q += " ORDER BY " + ", ".join(spec.keys)
    return q


def chunks(spec, end):
    """[(file_stem, lo, hi)] for ``spec``; ``end`` caps tables flagged ``to_end``."""
    lo = pd.Timestamp(spec.start) if spec.start else None
    hi = pd.Timestamp(end) if (spec.to_end and end is not None) else None
    if not spec.yearly:
        return [(spec.name, lo, hi)]
    if lo is None or hi is None:
        raise ValueError(f"{spec.name}: yearly chunks need a start and an end date")
    return [(f"{spec.name}_{y}", max(lo, pd.Timestamp(year=y, month=1, day=1)),
             min(hi, pd.Timestamp(year=y, month=12, day=31)))
            for y in range(lo.year, hi.year + 1)]


# ---------------------------------------------------------------- run
def latest_crsp_date(session):
    """Last trading day in crspq.dsf_v2 (the run's end date)."""
    d = session.sql("SELECT max(dlycaldt) AS d FROM crspq.dsf_v2")["d"].iloc[0]
    return pd.Timestamp(d)


def meta_dir(data_dir):
    """``data/meta`` (created if needed)."""
    d = Path(data_dir) / META_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_schemas(session, data_dir, tables=TABLES):
    """describe_table for every table -> ``meta/_schemas.txt``; returns {name: schema DataFrame}."""
    schemas, parts = {}, []
    for spec in tables:
        key = f"{spec.lib}.{spec.table}"
        try:
            schemas[spec.name] = sch = session.describe(spec.lib, spec.table)
            parts.append(f"## {key}\n{sch.to_string(index=False)}\n")
        except Exception as e:
            parts.append(f"## {key}\nERROR: {e}\n")
    (meta_dir(data_dir) / "_schemas.txt").write_text("\n".join(parts))
    return schemas


MANIFEST_COLS = ["file", "rows", "seconds", "pulled_utc", "sql"]


def _read_manifest(path):
    """{file: manifest row} from an existing manifest.csv (empty if none)."""
    if not path.exists():
        return {}
    return {r["file"]: r for r in pd.read_csv(path).to_dict("records")}


def pull(session, data_dir, end, tables=TABLES, schemas=None, log=print):
    """Pull every chunk of every table sequentially into ``data_dir/<source>/``; skip existing files.

    Returns the manifest DataFrame (also written to ``meta/manifest.csv`` after each file); its
    ``file`` column is the path relative to ``data_dir``, e.g. ``crsp/crsp_dsf_v2_2004.parquet``.
    """
    data_dir = Path(data_dir)
    schemas = schemas or {}
    man_path = meta_dir(data_dir) / "manifest.csv"
    manifest = _read_manifest(man_path)

    def _frame():
        return pd.DataFrame(list(manifest.values()), columns=MANIFEST_COLS)
    for spec in tables:
        schema = schemas.get(spec.name)
        if schema is None:
            schema = schemas[spec.name] = session.describe(spec.lib, spec.table)
        dcols = date_columns(spec, schema)
        folder = data_dir / spec.folder
        folder.mkdir(parents=True, exist_ok=True)
        for stem, lo, hi in chunks(spec, end):
            f = folder / f"{stem}.parquet"
            rel = f"{spec.folder}/{f.name}"
            if f.exists():
                continue
            q = build_query(spec, schema, lo, hi)
            t0 = time.time()
            df = session.sql(q)
            for c in dcols:    # out-of-range sentinels such as 9999-12-31 become NaT (= open)
                df[c] = pd.to_datetime(df[c], errors="coerce")
            df.to_parquet(f, compression="zstd", index=False)
            row = {"file": rel, "rows": len(df), "seconds": round(time.time() - t0, 1),
                   "pulled_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sql": q}
            manifest[rel] = row
            _frame().to_csv(man_path, index=False)
            log(f"{rel}: {len(df):,} rows in {row['seconds']}s")
    return _frame()


def read_table(data_dir, name, columns=None, folder=None):
    """Read a pulled table from ``data_dir/<folder>/``, concatenating yearly files if needed.

    ``folder`` defaults to the source folder of the table in :data:`TABLES` (e.g. crsp).
    """
    if folder is None:
        folder = BY_NAME[name].folder if name in BY_NAME else ""
    d = Path(data_dir) / folder
    single = d / f"{name}.parquet"
    files = [single] if single.exists() else sorted(d.glob(f"{name}_[0-9][0-9][0-9][0-9].parquet"))
    if not files:
        raise FileNotFoundError(f"no file for {name} in {d}")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


# ---------------------------------------------------------------- not on WRDS
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fetch_fred(series, path, timeout=30):
    """Download a public FRED series (e.g. GNPDEF, the GNP price deflator for oscore) as CSV."""
    import urllib.request
    with urllib.request.urlopen(FRED_CSV.format(series=series), timeout=timeout) as r:
        Path(path).write_bytes(r.read())


def write_readme(data_dir, end, manifest, notes=()):
    """meta/README.md describing the pull: tables, filters, period, files."""
    lines = ["# WRDS raw pull (feature engineering, no TAQ)", "",
             f"Pulled {datetime.now(timezone.utc):%Y-%m-%d} UTC with `data/scripts/pull_wrds.py`; "
             f"end date (latest CRSP day) {_iso(end)}.",
             "WRDS licence: do not share these files outside the team.", "",
             f"Universe permnos: every permno in crspq.msp500list_v2 with mbrenddt >= {UNIVERSE_FROM}.",
             "No name filter is applied to the S&P 500 universe; the common-stock filter on msf_v2 "
             "only serves industry / cross-sectional aggregates.", "",
             "| name | folder | WRDS table | filters | period | files | rows |",
             "|---|---|---|---|---|---|---|"]
    for spec in TABLES:
        files = manifest[manifest["file"].str.match(rf"^{spec.folder}/{spec.name}(_\d{{4}})?\.parquet$")]
        period = f"{spec.start or 'all'} -> {'latest' if spec.to_end else 'all'}" if spec.date_col else "all"
        filt = "; ".join(spec.where).replace(UNIVERSE_PERMNOS, "universe permnos") \
            .replace(UNIVERSE_GVKEYS, "universe gvkeys (via CCM)") or "-"
        lines.append(f"| `{spec.name}` | `{spec.folder}/` | {spec.lib}.{spec.table} | {filt} | {period} | "
                     f"{len(files)} | {int(files['rows'].sum()):,} |")
    lines += ["", "Exact SQL for every file is in `manifest.csv`; live schemas in `_schemas.txt`; "
              "checks in `CHECKS.md` (all in `data/meta/`)."]
    lines += [""] + [f"- {n}" for n in notes]
    (meta_dir(data_dir) / "README.md").write_text("\n".join(lines) + "\n")
