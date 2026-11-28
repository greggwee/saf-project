# Repository guidelines

Team project for Statistical Aspects of Finance (Columbia). The project replicates and extends
Bollerslev, Li & Tang (Management Science 2026; `Bollerslev.pdf`): 15-minute NBBO midquote returns
plus an overnight return for the point-in-time S&P 500 universe.

The course grades the project on three engineering criteria. Every change must respect them:

1. **Clearly organized, modular code with substantial shared libraries that team members use.**
2. **Sound software development practices**: reasonably sized commits, feature branches, pull requests.
3. **Source-of-truth code files for the project whitepaper and presentation.**

## 1. Code organization and shared library

### Target layout

```
saf/                  # shared Python package: the ONLY place reusable logic lives
  universe.py         # S&P 500 PIT segments, monthly universe, symbol mapping (CRSP -> TAQ/Databento)
  wrds.py             # WRDS connection, ~/.pgpass handling, retrying SQL helper
  grid.py             # 15-min grid construction, midquote snapping, staleness checks
  returns.py          # intraday log returns, overnight returns, split/dividend adjustment
  io.py               # parquet read/write, paths, dataset conventions
  ...                 # analysis modules (realized measures, models, plots) as the project grows
data/                 # data-building entry points (thin CLIs) + small committed reference CSVs
  build_sp500_pit_ids.py
  utils/              # current ingestion scripts (to migrate into saf/ + thin CLIs)
analysis/             # thin scripts that produce every table/figure used in the paper and slides
notebooks/            # exploration only; import from saf/, never define reusable logic here
tests/                # pytest tests for saf/
paper/                # whitepaper sources (see section 3)
slides/               # presentation sources (see section 3)
```

### Rules

- **Shared logic goes in `saf/`.** If a function is (or could be) used by two scripts, notebooks or
  team members, it belongs in the package, not copied. Example: segment loading and CRSP symbol
  mapping live once in `saf/universe.py` and both ingestion scripts import them.
- **Setup:** `pip install -e ".[dev]"` from the repo root makes `saf` importable from any script
  or notebook; run tests with `pytest`. Modal jobs ship `saf/` in the image (`modal_app.py`).
- **Scripts are thin.** Entry points parse arguments, call `saf` functions, and write outputs.
  Use `argparse` and a `main()` guarded by `if __name__ == "__main__":`.
- **Notebooks never hold the source of truth.** Anything worth keeping moves into `saf/` or
  `analysis/`. Clear outputs before committing.
- **Small, single-purpose functions** with docstrings stating inputs, outputs, units and
  timezone (all intraday timestamps are `America/New_York`).
- Keep method constants (grid frequency, 27 points/day, 09:30-16:00 ET, half-day sessions,
  staleness threshold) defined once in `saf` and imported, never re-hardcoded.
- No secrets in code or commits. Credentials come from `.env` (gitignored), `~/.pgpass`,
  or Modal secrets. Never print or log them.
- Large data (`mid15m/`, parquet outputs, raw TAQ/Databento pulls) is not committed; only small
  reference files such as `data/sp500_pit_identifiers_segments.csv`. Document how to regenerate
  every dataset.
- Python 3, `numpy`/`pandas`/`pyarrow`. Pin dependencies in `requirements.txt` (or
  `pyproject.toml`) when adding one. Write new code, comments and docstrings in English.
- Add or update `pytest` tests in `tests/` for any `saf/` function with non-trivial logic
  (grid snapping, return computation, symbol mapping, overnight adjustment). Tests run on small
  synthetic data and must not require WRDS or Databento access.

## 2. Development workflow

- **Never commit directly to `main`.** Create a feature branch per task:
  `feature/<topic>`, `fix/<topic>`, `paper/<section>`, `slides/<topic>`.
- **Reasonably sized commits.** One logical change per commit (e.g. "Move load_segments into
  saf.universe" separate from "Add overnight return tests"). Don't mix refactors, new features
  and data changes in one commit. Don't commit generated data or notebook outputs.
- **Commit messages**: imperative subject line under ~70 characters, optional body explaining why.
- **Pull requests for every change to `main`.** A PR has a clear description (what, why, how
  tested), stays focused on one topic, and is reviewed by at least one teammate before merging.
  Keep PRs small enough to review in one sitting.
- Before opening a PR: code runs, tests pass (`pytest`), and no secrets or large files are staged.
- Rebase or merge `main` into the branch to resolve conflicts before review; never force-push to
  `main`.

## 3. Whitepaper and presentation as source code

- The whitepaper lives in `paper/` and the presentation in `slides/` as **text source files**
  (LaTeX or Quarto/Markdown), version-controlled like code. Binary exports (PDF, PPTX) are build
  outputs, not the source of truth.
- **Every number, table and figure in the paper or slides is produced by code** in `analysis/`
  (using `saf/`) and written to `paper/figures/` / `paper/tables/` (and reused by `slides/`).
  Never paste a number or a screenshot by hand.
- Each generated artifact should be traceable: the script name that produces it appears in a
  comment next to where it is included.
- Provide one command (e.g. a `Makefile` target such as `make paper`, `make slides`) that rebuilds
  figures/tables and compiles the documents.
- Changes to the paper and slides follow the same branch + PR workflow as code.

## Reference: current data pipeline

- `data/build_sp500_pit_ids.py`: builds `sp500_pit_identifiers_segments.csv` (PERMNO x period with
  stable ticker/CUSIP) from CRSP/CCM/IBES/CIQ parquet extracts.
- `data/utils/data_getter.py`: WRDS TAQ millisecond NBBO midquotes on the 15-min grid, plus
  overnight returns (`--check`, `--start/--end`, `--overnight-only`).
- `data/utils/modal_app.py`: runs `data_getter.py` on Modal, one container per month.
- `data/utils/fetch_sp500_pit_15m.py`: alternative source, Databento EQUS.MINI `bbo-1m`.
