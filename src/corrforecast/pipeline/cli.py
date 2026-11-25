"""Command-line entry point: ``corrforecast <command> ...``.

Examples::

    corrforecast validate configs/baseline_15min.yaml
    corrforecast run configs/baseline_15min.yaml                      # all stages
    corrforecast run configs/denoised_qmle_tick.yaml --stages realized features
    corrforecast compare baseline_15min denoised_bmn_tick --target-run baseline_15min
    corrforecast synthetic data/synthetic --end 2007-12-31            # demo inputs
"""

from __future__ import annotations

import argparse
import logging
import sys

from corrforecast.config import load_config
from corrforecast.pipeline.run import STAGES, run


def _cmd_validate(args: argparse.Namespace) -> int:
    from corrforecast.data.loaders import IntradayReader, load_inputs

    cfg = load_config(args.config)
    inputs = load_inputs(
        cfg.data, cfg.features.factor_characteristics, cfg.applications.equity_premium_controls
    )
    reader = IntradayReader(
        cfg.data.intraday, inputs.assets, cfg.data.session_open, cfg.data.session_close
    )
    first = inputs.trading_days[: args.days]
    n = sum(len(f) for _, f in reader.iter_days(first))
    print(
        f"OK: {inputs.n_assets} assets, {len(inputs.trading_days)} trading days, "
        f"{inputs.monthly['month'].nunique()} months; {n} intraday rows in the first {len(first)} days."
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    run(cfg, tuple(args.stages), overwrite=args.overwrite)
    print(f"done: {cfg.run_dir}")
    return 0


def _cmd_synthetic(args: argparse.Namespace) -> int:
    from corrforecast.data.synthetic import SyntheticSpec, write_synthetic

    spec = SyntheticSpec(
        n_assets=args.n_assets,
        start=args.start,
        end=args.end,
        obs=args.obs,
        steps_per_day=args.steps_per_day,
        noise_std=args.noise_std,
        noise_factor_std=args.noise_factor_std,
        seed=args.seed,
    )
    paths = write_synthetic(args.out, spec)
    print("\n".join(f"{k}: {v}" for k, v in paths.items()))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from pathlib import Path

    from corrforecast.evaluation.compare import compare_runs

    df = compare_runs(
        Path(args.output_dir), args.target_run, args.runs, args.benchmark, args.models
    )
    out = Path(args.output_dir) / f"compare_target-{args.target_run}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwritten: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="corrforecast",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser(
        "validate", help="check that the configured inputs satisfy the data contract"
    )
    v.add_argument("config")
    v.add_argument("--days", type=int, default=5, help="number of intraday days to spot-check")
    v.set_defaults(func=_cmd_validate)

    r = sub.add_parser("run", help="run pipeline stages")
    r.add_argument("config")
    r.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    r.add_argument(
        "--overwrite", action="store_true", help="recompute daily matrices that already exist"
    )
    r.set_defaults(func=_cmd_run)

    c = sub.add_parser(
        "compare", help="score several runs' forecasts against one run's realized target"
    )
    c.add_argument("runs", nargs="+", help="run names under --output-dir")
    c.add_argument(
        "--target-run", required=True, help="run whose realized correlations are the common target"
    )
    c.add_argument("--output-dir", default="artifacts")
    c.add_argument(
        "--benchmark", default="HAR", help="benchmark model of the target run (R2 denominator)"
    )
    c.add_argument("--models", nargs="*", default=None)
    c.set_defaults(func=_cmd_compare)

    s = sub.add_parser("synthetic", help="write synthetic inputs that satisfy the data contract")
    s.add_argument("out")
    s.add_argument("--n-assets", type=int, default=12)
    s.add_argument("--start", default="2001-01-01")
    s.add_argument("--end", default="2007-12-31")
    s.add_argument("--obs", choices=["grid", "poisson"], default="grid")
    s.add_argument("--steps-per-day", type=int, default=78)
    s.add_argument("--noise-std", type=float, default=0.0)
    s.add_argument("--noise-factor-std", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_cmd_synthetic)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
