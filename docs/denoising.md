# Using finer intraday data: noise-robust estimators

BLT deliberately use a **coarse 15-minute** grid. At higher frequencies, plain realized
covariance is biased by two problems:

* **Microstructure noise.** Bid-ask bounce and discreteness add a bias of roughly
  `2n·Var(noise)` to the variances.
* **Asynchronous trading.** This is the Epps effect: correlations shrink toward 0 as the
  sampling interval gets shorter.

The paper's conclusion (p. 21) names two "more sophisticated realized estimators" that may
let finer data improve the forecasts. Both are implemented here as drop-in daily estimators.
Every later stage (features, models, evaluation, applications) is unchanged.

| config | data | estimator |
|---|---|---|
| `baseline_15min.yaml` | quotes → 15-min previous tick | plain RCov (BLT) |
| `naive_1min.yaml` | quotes → 1-min previous tick | plain RCov (shows the noise/Epps bias) |
| `denoised_qmle_tick.yaml` | tick quotes | **Aït-Sahalia, Fan & Xiu (2010)**, pairwise refresh time |
| `denoised_qmle_1min.yaml` | 1-min bars | AFX QMLE on a common 1-min grid |
| `denoised_bmn_tick.yaml` | tick quotes | **Bollerslev, Meddahi & Nyawa (2019)**, refresh time across all stocks |
| `denoised_bmn_1min.yaml` | 1-min bars | BMN on a common 1-min grid |

Other estimators are available as building blocks or benchmarks: `mrc` (pre-averaging,
Christensen, Kinnebrock & Podolskij 2010) and `realized_kernel` (Barndorff-Nielsen et al.
2011).

## 1. Aït-Sahalia, Fan & Xiu (2010) — quasi-MLE with polarisation

*JASA 105(492), 1504–1517. Code: `realized/estimators/qmle.py` (`qmle_afx`).*

1. **Univariate QMLE (Xiu 2010).** The observed log price is `Y = X + U`, with `U` i.i.d.
   noise. The estimator treats `X` as if it had constant volatility. Returns then follow a
   Gaussian MA(1) with covariance `Ω = σ²Δ·I + a²·tridiag(-1, 2, -1)`. A type-I discrete
   sine transform diagonalises Ω exactly, with eigenvalues `σ²Δ + a²(2 − 2cos(kπ/(n+1)))`.
   The likelihood is therefore a 2-parameter problem. It is solved by vectorised Fisher
   scoring across thousands of series at once (`qmle_batch`). The integrated variance is
   `n·σ̂²Δ`. The estimator stays consistent under stochastic volatility (rate n^{1/4}) and
   needs no tuning parameters.
2. **Synchronisation.** For each pair, the *refresh times* of the two stocks are the
   "Generalized Sampling Time" (`sampling.refresh_times`).
3. **Polarisation.** On variance-normalised prices,
   `ρ_ij = [V(X̃_i + X̃_j) − V(X̃_i − X̃_j)] / 4`, where every V is a univariate QMLE.
   The result is clipped to [−1, 1]. The daily covariance is `ρ_ij √(IV_i IV_j)`, with
   `IV_i` from each stock's own ticks (`variance_source: own`) or from the synchronised data
   (`sync`).
4. The matrix is not guaranteed to be PSD. Set `psd: true` to project it. This is not needed
   for the pairwise features.

Sync modes:

* `pairwise_refresh` is the paper's approach. It uses the most data but is O(N²) per day:
  about 86k pairs for N = 417, roughly 1 ms per pair. Parallelise over days with
  `realized.n_jobs`.
* `refresh` uses one refresh-time grid for all stocks.
* A grid such as `1min` is cheap: the whole matrix is one batched solve. This works because
  the DST is linear, so the DST of a sum or difference is computed from each stock's DST.

## 2. Bollerslev, Meddahi & Nyawa (2019) — factor-based estimator for large dimensions

*J. Econometrics 212(1), 116–136. Code: `realized/estimators/bmn.py` (`bmn_factor`).*

This estimator allows the noise to have its own **factor structure**,
`u_it = c_i g_t + η_it`. Most noise-robust covariance estimators assume noise that is
cross-sectionally independent, and BMN argue that this assumption fails for quotes.
Prices are synchronised by refresh time, giving returns `r` of size n × p.

1. A pilot multivariate realized kernel (`kernel.realized_kernel`: Parzen, jittered end
   points, BNHLS bandwidth) gives the eigenvectors `W` of its K largest eigenvalues.
2. The rotated factors are `f_t = W' r_t / p`.
3. Factor variances are `PRV(f_k)` (pre-averaged RV). Loadings are
   `b_ik = MRC(r_i, f_k) / PRV(f_k)`, using the modulated realized covariance
   (`preaveraging.py`, g(x) = min(x, 1−x), k_n = ⌈θ√n⌉).
4. Residuals are `e = r − f b'`, with idiosyncratic variances `PRV(e_i)`.
5. `Σ = b·diag(PRV(f))·b' + diag(PRV(e))`. This is positive definite by construction.

K is either fixed or chosen each day by the Bai–Ng (2002) IC_p1 on 5-minute returns. BMN's
S&P 500 application found an average K of about 3.3. Tuning parameters: `theta` (pre-averaging
window, default 1/3), `k_max`, `pilot` (`mrker` | `mrc` | `rcov`) and `min_obs`. BMN drop
stocks with fewer than 195 trades a day.

One caveat applies to large universes. Refresh time across all stocks is limited by the
least active stock. Raise `min_obs` to exclude illiquid names for the day, or use a 1-minute
grid.

## Semicovariances

Neither paper defines a noise-robust *semi*covariance. The six `RC^-` features, and the
`ExpRC^-` and `ExpScRC^-` features, therefore keep the 15-minute definition through
`realized.semicov_estimator`. A pre-averaged semicovariance would be a natural next step:
it would be a new estimator class with `provides_semicov = True`.

## Comparing variants fairly

Switching the estimator changes **both** the features and the target: next month's RC is
measured with the same estimator. The R²_OOS of two runs scored against their own targets
are therefore not comparable. Two tools address this:

* `corrforecast compare <runs…> --target-run baseline_15min` re-scores every run's
  forecasts against one common realized target, on the intersection of (month, pair)
  observations, relative to that run's HAR.
* `models.target_run: baseline_15min` trains a variant's features *on* another run's target.
  This separates "better predictors" from "better measured target". A typical 2×2 study
  crosses {15-min features, denoised features} with {15-min target, denoised target}.

Suggested workflow once the data is in place:

```bash
corrforecast run configs/baseline_15min.yaml
corrforecast run configs/denoised_bmn_tick.yaml
corrforecast run configs/denoised_qmle_1min.yaml
corrforecast compare baseline_15min denoised_bmn_tick denoised_qmle_1min --target-run baseline_15min
corrforecast compare baseline_15min denoised_bmn_tick denoised_qmle_1min --target-run denoised_bmn_tick
```

## Validation on simulated data

`tests/test_estimators.py` simulates noisy, asynchronous (Poisson) ticks with factor noise.
It checks that:

* the QMLE recovers integrated variance within 15% when the naive RV is more than 50× too
  large;
* the MRC/PRV bias correction works;
* the QMLE matches a brute-force optimiser;
* AFX and BMN correlations beat both naive 10-second RCov and 15-minute RCov.

In one illustrative simulation (8 stocks, true mean correlation 0.35), the mean absolute
correlation error was: 15-min RCov 0.14, 5-sec RCov 0.33, QMLE 0.07, BMN 0.09, MRC 0.08.
