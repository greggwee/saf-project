# Methodology: from the paper to the code

This document maps the paper to the code. The paper is Bollerslev, Li & Tang, *Forecasting
and Managing Correlation Risks*, Management Science (2026), referred to below as **BLT**.
Where the paper leaves a detail open, the choice made here is listed in the last section and
can be changed in the config.

## Pipeline stages

| stage | module | output (`artifacts/<run>/…`) |
|---|---|---|
| 1. daily realized measures | `realized/compute.py` + `realized/estimators/*` | `daily/{cov,semicov}/<year>.npy` |
| 2. features + target       | `features/build.py`, `features/transforms.py` | `features/pairs/year=*/part.parquet`, `features/monthly/*.npy` |
| 3. rolling forecasts       | `models/rolling.py`, `models/estimators.py` | `forecasts/predictions/…`, `forecasts/coefficients.parquet` |
| 4. statistical evaluation  | `evaluation/metrics.py`, `evaluation/compare.py` | `results/r2_oos.csv`, `results/dm_{ew,vw}.csv`, `results/feature_contributions_LASSO.csv` |
| 5. applications            | `applications/*` | `results/pairs_*.csv`, `equity_premium.csv`, `risk_targeting.csv`, `gmv.csv` |

## Equation map

| BLT | content | code |
|---|---|---|
| eq. (1) | daily RV / RCov from n = 27 returns (26 × 15-min + overnight), ×252 | `realized_cov.RealizedCov` (sampling `15min`) + overnight term in `realized/compute.py` |
| eq. (A.1) | negative realized semicovariance `Σ r_i 1{r_i<0} r_j 1{r_j<0}` | `realized_cov.realized_cov_and_semicov` |
| §2.2 | weekly/monthly measures = averages of daily ones | `features/build.monthly_matrices` |
| eq. (2) | `RCov = √RV · RC · √RV` (correlation from covariance) | `realized/psd.cov_to_corr` |
| A.1 | HAR features `RC^{d,w,m}` and semicorrelations `RC^{d-,w-,m-}` | `features/build.monthly_matrices` |
| eq. (A.2) | factor-driven covariance `L(L'L)^{-1}L'·RCov·L(L'L)^{-1}L' + Diag(RCov^e)` | `features/transforms.factor_projection`, `loading_matrix` (rank → [-1,1], beta raw) |
| eq. (A.3) | exponentially weighted RCov over 500 days, `λ = log(1+1/CoM)`, CoM ∈ {1,5,21,63} | `features/transforms.exp_weights` |
| A.3 | sector-specific `ExpScRC` (within-sector average, 0 across sectors) | `features/transforms.sector_average` |
| endnote 8 | features bounded to [-1, 1] | `cov_to_corr(clip=True)` |
| §3.1 | rolling train (t-4…t-1) / validation (t) / test (t+1); OLS on t-4…t | `models/rolling.make_splits` |
| eq. (3) | pooled LASSO with features standardised on the training sample, λ chosen on validation | `models/estimators.Lasso` |
| Table 2 | HAR, SHAR (6), SHAR-F (9), SHAR-F-Exp (25), LASSO (25) | `features/names.FEATURE_SETS`, `configs/baseline_15min.yaml` |
| endnote 10 | "insanity filter": clip predictions to [-1, 1] | `models.clip_predictions` |
| eqs. (4)–(5) | R²_OOS, EW and VW (product of caps winsorised at the 90th percentile) | `evaluation/metrics.r2_oos`, `value_weights` |
| eq. (6) | modified Diebold–Mariano across stocks | `evaluation/metrics.dm_test` |
| Fig. 4 | LASSO coefficient shares by year | `models/rolling.feature_contributions` |
| eq. (7), Tables 4–5 | RetDiff pairs trading, ΔRC conditioning, Fama–MacBeth | `applications/pairs_trading.py` |
| eq. (9), Table 6 | AvgCorr (EW/VW) equity premium regressions | `applications/equity_premium.py` |
| eqs. (10)–(11), Fig. 6 | HAR variances × predicted correlations; risk-targeting ratios with bootstrap CIs | `applications/portfolios.run_risk_targeting` |
| eqs. (12)–(15), Figs. 7–8 | GMV and beta-neutral GMV; utility gains Δγ | `applications/portfolios.run_gmv`, `common.utility_gain` |
| Table 8 | Ridge, ENet, PCR (≤ 20 PCs), FNN (4-2) | `models/estimators.py`, `configs/robustness_ml.yaml` |
| §6.3 | unbalanced panel | `data.universe_rule: all` |

## Avoiding look-ahead

* Every sample is indexed by its **target month**. A model refit at the end of year t uses
  only targets dated no later than December of year t. The test
  `tests/test_pipeline_e2e.py::test_no_lookahead_in_rolling` corrupts all test-year targets
  and checks that every fitted coefficient stays the same.
* Feature standardisation uses training-sample moments only.
* Characteristics are taken as of month-end t. The data team is responsible for reporting
  lags.
* HAR variance forecasts for the applications are re-estimated every test year on past data
  only.

## Details the paper leaves open: choices made here

| item | choice (config key) | note |
|---|---|---|
| semicorrelation normalisation | `RCov^-_ij / √(RCov^-_ii RCov^-_jj)` | matches Table 1, where `RC^{d-}` ∈ [0, 1] with mean 0.46. A.3 also normalises `ExpRC^-` by `ExpRV^-` |
| weekly / monthly window | last 5 trading days / calendar month (`features.weekly_days`, `features.monthly_window`) | the response is the calendar-month RC |
| intercept in L | none (`features.factor_add_intercept`) | the unranked CAPM beta acts like a level |
| ExpScRC | mean off-diagonal `ExpRC` within the sector | gives Table 1's mean of about 0.04 with 10 sectors |
| LASSO final model | the training-sample fit at the chosen λ (`models.refit_on_train_plus_val: false`) | "choosing the final preferred model … from this collection" |
| VW weights | normalised within each month; DM-VW multiplies `d_ij,t` by `ω_ij,t · n_pairs,t` | |
| HAR volatility model | pooled OLS on monthly RV^{d,w,m}, 5-year window, refit yearly | §5.3 only says "the popular HAR model" |
| PSD correction before GMV | eigenvalue clipping of the predicted correlation matrix | the paper's Online Appendix procedure is not reproduced here |
| risk-targeting portfolios | equal-weighted top-minus-bottom decile on each characteristic | BLT's exact anomaly portfolio construction is in their Online Appendix |
