# Data contract (inputs delivered by the data team)

The pipeline **does not clean data**. It reads already-cleaned files and checks them against
the schemas in `src/corrforecast/data/schemas.py`. Every violation found is reported in one
error message, so you can fix everything at once. Check a delivery with:

```bash
corrforecast validate configs/baseline_15min.yaml
```

File locations are set in the `data:` block of the run config. Parquet is preferred. A file
path, a directory, or a hive-partitioned directory (for example `year=2010/`) all work. CSV
also works for the small tables.

Conventions used throughout:

* `asset_id` is a stable security identifier, ideally the CRSP PERMNO. It must be the same
  integer or string in every file.
* Times are **exchange-local wall clock (US/Eastern)**. Timestamps can be tz-naive or
  tz-aware; tz-aware ones are converted to tz-naive.
* Returns are decimals, not percent.

---

## 1. Intraday prices — `data.intraday`

There is one row per (asset, timestamp) observation. The same schema serves tick-level
quotes, second bars and minute bars. **The sampling and denoising are chosen in the config,
not in the data.**

| column      | type          | required | notes |
|-------------|---------------|----------|-------|
| `asset_id`  | int / str     | yes | |
| `timestamp` | datetime      | yes | unique per asset; the last observation wins if a timestamp repeats |
| `price`     | float > 0     | yes | **NBBO midquote** recommended. BLT use midquotes to limit bid-ask bounce |
| `date`      | date          | no  | trading date; derived from `timestamp` if missing |

What the data team is expected to have done (following standard TAQ cleaning, e.g.
Barndorff-Nielsen, Hansen, Lunde & Shephard 2009):

* Keep regular trading hours only, 09:30–16:00. The reader also drops anything outside the
  configured session.
* Drop quotes with zero or negative bid/ask, crossed or locked markets, and abnormally wide
  spreads. Merge quotes with the same timestamp, for example by median.
* Make no split or dividend adjustment inside a day. Intraday returns are only taken within
  a day, so raw prices are fine. The overnight return comes from the daily file instead (see
  below).

Suggested layouts:

* `data/intraday/quotes/` — tick-level cleaned midquotes. Used by the 15-minute baseline
  (previous-tick sampling) and by the tick-level denoised configs.
* `data/intraday/bars_1min/` — 1-minute last-midquote bars, used by the `*_1min` configs.

Partition by year or month (`year=2010/month=03/part-0.parquet`) so the reader can stream one
month at a time.

## 2. Daily stock panel — `data.daily`

| column          | type   | required | notes |
|-----------------|--------|----------|-------|
| `date`          | date   | yes | trading days. The set of dates defines the trading calendar |
| `asset_id`      | id     | yes | |
| `ret`           | float  | yes | daily simple total return (CRSP `RET`). Used for pairs-trading betas |
| `ret_overnight` | float  | yes, unless `realized.include_overnight: false` | **log** return from the previous close to today's first intraday price, **adjusted for splits and dividends** (for example `log(1+RET) - log(P_close/P_open)`, or from CRSP `OPENPRC`/`cfacpr`) |
| `mktcap`        | float  | no | |
| `price`, `shrcd`, `exchcd`, `n_trades` | | no | only needed if you want the pipeline to re-check the universe filters |

The **universe** is the sorted set of `asset_id`s that defines the row and column order of
every matrix. It comes from one of two places:

1. `data.universe` — an optional file with an `asset_id` column. This is the recommended way
   to hand over BLT's filtered S&P 500 sample: share code 10/11, $1 ≤ price ≤ $1,000, at
   least 100 trades a day, and full history from 2000 to 2020.
2. Otherwise `data.universe_rule`:
   * `balanced` keeps assets present on every trading day (BLT's 417 stocks).
   * `all` keeps every asset (the unbalanced extension in BLT §6.3).

## 3. Monthly stock panel — `data.monthly`

| column         | type   | required | notes |
|----------------|--------|----------|-------|
| `month`        | date   | yes | any date in the month; normalised to the calendar month-end |
| `asset_id`     | id     | yes | |
| `mktcap`       | float>0| yes | market cap at month-end. Used for value weights |
| `sector`       | str    | yes | GICS 2-digit sector (10, 15, …, 60). Used for the ExpScRC features |
| `ret`          | float  | for applications | monthly simple total return |
| characteristics| float  | yes (those listed in `features.factor_characteristics`) | NaN allowed |

The default characteristic columns are the loading matrix `L` of BLT Appendix A.2:

`beta, size, bm, rev` (CAPM beta from 60-month rolling regressions, log market equity,
book-to-market, short-term reversal) plus the 11 Stambaugh-Yu-Yuan (2012) anomalies
`nsi, cei, acc, noa, agr, inta, oscore, mom, gpf, roa, dis`.

Each characteristic must be **known at the end of `month`**. Apply the usual reporting lags,
for example Green, Hand & Zhang (2017) / Chen & Zimmermann (2022). The pipeline uses
month-t values to build the forecasts made at the end of month t. The pipeline itself
rank-transforms each characteristic to [-1, 1], except `beta`, so deliver raw values.

## 4. Monthly market data — `data.market` (applications only)

| column       | required | notes |
|--------------|----------|-------|
| `month`      | yes | |
| `mkt_excess` | yes | CRSP value-weighted market return minus the risk-free rate, in month `month` |
| `rf`         | yes | one-month T-bill return |
| `dp, ep, bm, ntis, tbl, tms, dfy, svar` | for Table 6 controls | Welch–Goyal predictors from Amit Goyal's website |

---

## Synthetic stand-in

`corrforecast synthetic data/synthetic` writes a complete set of files that satisfy this
contract. It simulates a factor model with time-varying correlations, optional noise and
optional asynchronous ticks. Use it to check the format of a delivery or to run the pipeline
before real data exists.
