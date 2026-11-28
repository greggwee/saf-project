# data/

One folder per data source. Pulled data is **gitignored** (WRDS and Databento licences: do not share
outside the team); only `scripts/` and `universe/` are committed. Every script defaults to these
folders, so run them from the repo root without path arguments (after `pip install -e ".[dev]"`).

| Folder | Contents | Produced by | Committed |
|---|---|---|---|
| `scripts/` | thin CLIs that build every folder below | - | yes |
| `universe/` | `sp500_pit_identifiers_segments.csv`: PERMNO x period of S&P 500 membership with stable ticker/CUSIP | `scripts/build_sp500_pit_ids.py` | yes |
| `crsp/` | `crsp_*.parquet`: S&P 500 list, security info, CCM links, daily (`dsf_v2`, by year) and monthly (`msf_v2`, by year) stock files, delistings, S&P 500 index | `scripts/pull_wrds.py` | no |
| `compustat/` | `comp_*.parquet`: `funda`, `fundq`, GICS history (`co_hgic`) | `scripts/pull_wrds.py` | no |
| `ff/` | `ff_factors_daily.parquet`, `ff_factors_monthly.parquet` | `scripts/pull_wrds.py` | no |
| `fred/` | `gnp_deflator.csv` (GNPDEF, quarterly) for `oscore` | `scripts/pull_wrds.py` | no |
| `taq/` | `wrds_15min/`: TAQ NBBO midquotes on the 15-min grid, one parquet per month; `databento_mid15m/`: Databento alternative, one parquet per PERMNO | `scripts/data_getter.py` (`modal_app.py` on Modal), `scripts/fetch_sp500_pit_15m.py` | no |
| `meta/` | `manifest.csv` (exact SQL, rows, time per file), `_schemas.txt`, `README.md` (pull summary), `CHECKS.md` + `checks_monthly.csv` | `scripts/pull_wrds.py` | no |

## Rebuild

```bash
python data/scripts/pull_wrds.py --wrds-user USER                 # CRSP, Compustat, FF, FRED + checks (~2 min)
python data/scripts/pull_wrds.py --checks-only                    # re-run the checks, no WRDS needed
python data/scripts/data_getter.py --start 2008-01 --end 2026-06 --wrds-user USER   # TAQ 15-min grid
modal run data/scripts/modal_app.py --start 2008-01 --end 2026-06                    # same, on Modal
```

Credentials come from `WRDS_PASSWORD` or `~/.pgpass` (never a prompt); Databento from
`DATABENTO_API_KEY` or `data/scripts/.env` (gitignored).
