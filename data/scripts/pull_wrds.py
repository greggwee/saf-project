#!/usr/bin/env python3
"""Pull every WRDS table needed for features and targets (everything except TAQ).

Table list, filters and periods: saf/wrds_pull.py (TABLES). Checks: saf/wrds_checks.py.
TAQ intraday data comes from data/scripts/data_getter.py / modal_app.py instead.

Output (gitignored, WRDS licence): data/crsp/, data/compustat/, data/ff/, data/fred/ and
data/meta/ (manifest.csv, _schemas.txt, README.md, CHECKS.md).

Runs sequentially on ONE connection (WRDS cuts heavy parallel use), resumes after an interruption
(existing files are skipped). Credentials: WRDS_PASSWORD or ~/.pgpass, never a prompt.

Usage
  python data/scripts/pull_wrds.py --wrds-user USER --schemas-only          # step 1: data/meta/_schemas.txt
  python data/scripts/pull_wrds.py --wrds-user USER --dry-run               # print the SQL, pull nothing
  python data/scripts/pull_wrds.py --wrds-user USER                         # full pull + README + checks
  python data/scripts/pull_wrds.py --wrds-user USER --tables comp_fundq,comp_co_hgic
  python data/scripts/pull_wrds.py --checks-only                            # no WRDS needed
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from saf import wrds_checks, wrds_pull
from saf.wrds import WrdsSession

log = logging.getLogger("pull_wrds")
DATA_DIR = Path(__file__).resolve().parents[1]      # the repo's data/ folder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wrds-user", default=os.environ.get("WRDS_USERNAME"))
    p.add_argument("--data-dir", default=str(DATA_DIR), help="root data folder (default: repo data/)")
    p.add_argument("--end", help="end date YYYY-MM-DD (default: latest day in crspq.dsf_v2)")
    p.add_argument("--tables", help="comma-separated subset of table names (default: all)")
    p.add_argument("--schemas-only", action="store_true", help="only write _schemas.txt")
    p.add_argument("--dry-run", action="store_true", help="print the SQL of every file, pull nothing")
    p.add_argument("--checks-only", action="store_true", help="only run the checks on --data-dir")
    p.add_argument("--no-checks", action="store_true")
    p.add_argument("--no-fred", action="store_true", help="skip the FRED GNPDEF download")
    return p.parse_args()


def main():
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = Path(a.data_dir) / wrds_pull.META_DIR

    if a.checks_only:
        wrds_checks.run_checks(a.data_dir, a.end)
        print(f"-> {meta}/CHECKS.md")
        return

    tables = wrds_pull.TABLES
    if a.tables:
        names = a.tables.split(",")
        unknown = [n for n in names if n not in wrds_pull.BY_NAME]
        if unknown:
            sys.exit(f"unknown tables {unknown}; choose from {list(wrds_pull.BY_NAME)}")
        tables = tuple(wrds_pull.BY_NAME[n] for n in names)

    session = WrdsSession(a.wrds_user, log=log)
    session.connect()
    schemas = wrds_pull.write_schemas(session, a.data_dir, tables)
    print(f"-> {meta}/_schemas.txt")
    failed = [t.name for t in tables if t.name not in schemas]
    for t in tables:            # check every requested column before pulling anything
        if t.name in schemas:
            try:
                wrds_pull.select_list(t, schemas[t.name])
            except ValueError as e:
                failed.append(t.name)
                print(f"SCHEMA MISMATCH {e}")
    if a.schemas_only:
        return
    if failed:
        sys.exit(f"fix the table specs first (see {meta}/_schemas.txt): {sorted(set(failed))}")

    end = pd.Timestamp(a.end) if a.end else wrds_pull.latest_crsp_date(session)
    log.info("end date %s", end.date())

    if a.dry_run:
        for t in tables:
            for stem, lo, hi in wrds_pull.chunks(t, end):
                print(f"-- {stem}\n{wrds_pull.build_query(t, schemas[t.name], lo, hi)};\n")
        return

    manifest = wrds_pull.pull(session, a.data_dir, end, tables, schemas, log=log.info)
    session.close()

    notes = []
    if not a.no_fred:
        try:
            fred = Path(a.data_dir) / wrds_pull.FRED_DIR
            fred.mkdir(parents=True, exist_ok=True)
            wrds_pull.fetch_fred("GNPDEF", fred / "gnp_deflator.csv")
            notes.append("`fred/gnp_deflator.csv`: FRED GNPDEF (GNP implicit price deflator, quarterly; "
                         "annualize for oscore).")
        except Exception as e:
            notes.append(f"FRED GNPDEF not downloaded ({e}); fetch it manually for oscore.")
    wrds_pull.write_readme(a.data_dir, end, manifest, notes)

    if not a.no_checks and not a.tables:
        wrds_checks.run_checks(a.data_dir, end)
        print(f"-> {meta}/CHECKS.md")


if __name__ == "__main__":
    main()
