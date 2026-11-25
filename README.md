# corrforecast

This package replicates **Bollerslev, Li & Tang (2026), *Forecasting and Managing Correlation
Risks*, Management Science**. It also extends the paper from the 15-minute sampling grid to
finer intraday data, using the two noise-robust estimators the paper points to:

* **Aït-Sahalia, Fan & Xiu (2010)** — quasi-MLE with refresh-time synchronisation and
  polarisation.
* **Bollerslev, Meddahi & Nyawa (2019)** — a factor-based estimator built from pre-averaging
  and a realized kernel, which also handles noise with a factor structure.

The data is cleaned by a separate workstream. This repo **ingests** it through a validated
data contract and runs the full methodology.

```
intraday prices ─┐
daily panel ─────┼─► [1] daily RCov / RCov⁻  ─► [2] 25 features + target ─► [3] rolling OLS/LASSO
monthly panel ───┘     (pluggable estimator)       (pair × month panel)        (train/val/test)
market data ─────────────────────────────────────────────────────────────────► [4] R²_OOS, DM
                                                                               [5] pairs trading, equity
                                                                                   premium, risk targeting, GMV
```

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q                                            # ~40 s, includes an end-to-end run
```

Run the whole pipeline on synthetic data before the real data arrives:

```bash
corrforecast synthetic data/synthetic --n-assets 40 --end 2009-12-31
corrforecast run configs/synthetic_demo.yaml
ls artifacts/synthetic_demo/results/
```

With the real data, put the files where `configs/baseline_15min.yaml` expects them, or edit
the `data:` block, and run:

```bash
corrforecast validate configs/baseline_15min.yaml   # data contract check
corrforecast run configs/baseline_15min.yaml        # stages: realized features forecast evaluate apps
corrforecast run configs/baseline_15min.yaml --stages forecast evaluate   # re-run later stages only
```

## Configs

| config | purpose |
|---|---|
| `baseline_15min.yaml` | the paper: 15-min midquotes plus overnight, 25 features, HAR/SHAR/SHAR-F/SHAR-F-Exp/LASSO |
| `robustness_ml.yaml` | Table 8: Ridge, ENet, PCR, FNN |
| `naive_1min.yaml` | plain 1-min RCov (shows why naive fine sampling fails) |
| `denoised_qmle_tick.yaml`, `denoised_qmle_1min.yaml` | AFX (2010) QMLE on tick data or 1-min bars |
| `denoised_bmn_tick.yaml`, `denoised_bmn_1min.yaml` | BMN (2019) on tick data or 1-min bars |
| `synthetic_demo.yaml` | runs everything on simulated data |

A variant config uses `extends: baseline_15min.yaml` and states only what differs. Each run
writes its resolved config, git revision and stage timings to `artifacts/<name>/`.

Compare variants on a **common target**. The target RC changes with the estimator, so R²s
scored against each run's own target are not comparable:

```bash
corrforecast compare baseline_15min denoised_bmn_tick --target-run baseline_15min
```

## Layout

```
src/corrforecast/
  config.py               typed (pydantic) run configuration, YAML with `extends`
  data/                   schemas (data contract), loaders, synthetic data
  realized/               sampling (previous tick, refresh time), daily matrix store
    estimators/           realized_cov (BLT) | qmle_afx | bmn_factor | mrc | realized_kernel
  features/               25 BLT features, pair-month panel
  models/                 OLS / LASSO / Ridge / ENet / PCR / FNN; rolling scheme
  evaluation/             R²_OOS (EW/VW), modified Diebold–Mariano, cross-run comparison
  applications/           pairs trading, equity premium, risk targeting, GMV
  pipeline/               stage orchestration and CLI
docs/
  data_contract.md        what the data team delivers (read this first)
  methodology.md          paper equation → code map; choices where the paper is silent
  denoising.md            the two noise-robust estimators and how to compare variants
```

## Development

```bash
ruff check src tests && ruff format --check src tests && mypy src && pytest -q
```

CI (`.github/workflows/ci.yml`) runs the same checks. Data and artifacts are git-ignored, so
never commit them.
