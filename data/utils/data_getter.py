#!/usr/bin/env python3
"""
Grille 15 min de midquotes NBBO (WRDS TAQ millisecondes) pour l'univers S&P 500 PIT.
Version script de data_getter.ipynb, pour tourner sans surveillance sur une VM.

Méthode (Bollerslev, Li & Tang, Management Science 2026) : midquote NBBO, 27 points par séance
(09:30 -> 16:00 ET ; 15 points les demi-séances), plus un rendement overnight.
Aucun filtre de nom n'est appliqué (share code 10/11, prix 1-1000 $, >= 100 transactions/jour).

Connexion WRDS sans invite : le script exige ~/.pgpass (voir --init-pgpass) et n'ouvre jamais
de prompt. Si la connexion tombe, il se reconnecte et retente UNE seule fois.

Usage
  # une fois, en interactif sur la VM (crée ~/.pgpass, chmod 600)
  python data_getter.py --init-pgpass --wrds-user ethan099
  # vérifier les schémas TAQ / lien CRSP-TAQM
  python data_getter.py --check --wrds-user ethan099
  # test sur un mois
  python data_getter.py --start 2016-06 --end 2016-06 --wrds-user ethan099
  # run complet, détaché de la session SSH
  nohup python data_getter.py --start 2008-01 --end 2026-06 --wrds-user ethan099 > run.log 2>&1 &
  # rendements overnight (après téléchargement ; ne nécessite pas WRDS)
  python data_getter.py --overnight-only --crsp-dsf-dir crsp_dsf_v2
"""
import argparse
import logging
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from saf.universe import crsp_to_taq_symbol, get_monthly_universe, load_segments

log = logging.getLogger("data_getter")
conn = None
WRDS_USER = None
PGPASS_HOST = "wrds-pgdata.wharton.upenn.edu"
PGPASS_PORT = 9737


# ---------------------------------------------------------------- connexion WRDS
def pgpass_ok(user):
    p = Path.home() / ".pgpass"
    if not p.exists():
        return False
    return any(line.startswith(f"{PGPASS_HOST}:{PGPASS_PORT}:") and f":{user}:" in line
               for line in p.read_text().splitlines())


def init_pgpass(user):
    """Crée ~/.pgpass via le client wrds (saisie interactive du mot de passe, une seule fois)."""
    import wrds
    c = wrds.Connection(wrds_username=user)   # demande le mot de passe (et Duo si exigé)
    c.create_pgpass_file()
    c.close()
    p = Path.home() / ".pgpass"
    p.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"{p} créé (chmod 600).")


def connect():
    """Connexion WRDS sans aucune invite.

    Le client wrds n'ouvre son prompt (username/password) que si la 1re tentative échoue, et
    masque alors l'erreur réelle. On affiche l'erreur (verbose=True) et on remplace le prompt
    par une exception explicite.
    Identifiants : WRDS_PASSWORD (variable d'environnement / secret Modal) sinon ~/.pgpass.
    """
    global conn
    import wrds
    import wrds.sql

    def _no_prompt(self):
        raise RuntimeError(
            "Connexion WRDS refusée (cause dans la ligne err=... ci-dessus). Causes usuelles : "
            "mot de passe/username faux, notification Duo non validée à temps, IP refusée.")

    wrds.sql.Connection._Connection__get_user_credentials = _no_prompt

    kw = {"wrds_username": WRDS_USER, "verbose": True}
    pwd = os.environ.get("WRDS_PASSWORD")
    if pwd:
        kw["wrds_password"] = pwd
    elif not pgpass_ok(WRDS_USER):
        sys.exit(f"Ni WRDS_PASSWORD ni ~/.pgpass pour {WRDS_USER}. Lancez une fois : "
                 f"python data_getter.py --init-pgpass --wrds-user {WRDS_USER}")
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    log.info("connexion WRDS en tant que %s (%s)", WRDS_USER, "mot de passe env" if pwd else ".pgpass")
    conn = wrds.Connection(**kw)
    return conn


def sql(query, params=None, date_cols=None, retries=1):
    """raw_sql avec UNE reconnexion + nouvelle tentative si la connexion est coupée.
    Si la seconde tentative échoue, l'erreur remonte (le jour est noté dans failed.csv)."""
    attempts = retries + 1
    for k in range(attempts):
        try:
            return conn.raw_sql(query, params=params, date_cols=date_cols)
        except Exception as e:
            # Nouvelle tentative uniquement pour une vraie coupure (erreurs de pilote/connexion),
            # jamais pour un bug de code ou une table absente.
            transient = (type(e).__name__ in ("OperationalError", "InterfaceError", "DisconnectionError")
                         or isinstance(e, (ConnectionError, TimeoutError)))
            missing = "does not exist" in str(e).lower()
            if missing or not transient or k == attempts - 1:
                raise
            wait = 30
            log.warning("connexion perdue (%s) -> reconnexion dans %ss", str(e)[:120], wait)
            time.sleep(wait)
            connect()


# ---------------------------------------------------------------- univers
# 1. Univers PIT (sortie de build_sp500_pit_ids.py). Aucun filtre de nom appliqué ici.
#    Filtres de l'article (documentés, NON appliqués) : NYSE/NASDAQ/AMEX, share code 10/11,
#    prix 1–1000 $, >= 100 transactions/jour, historique complet 2000–2020.

# ---------------------------------------------------------------- symboles
# 4. Correspondance PERMNO -> (sym_root, sym_suffix), jour par jour.
#    Source principale : table de lien WRDS CRSP-TAQM (PIT). Repli : tickers CRSP du segment.
LINK_TABLE = "wrdsapps_link_crsp_taqm.taqmclink"
LINK_DATE, LINK_PERMNO, LINK_ROOT, LINK_SUFFIX = "date", "permno", "sym_root", "sym_suffix"
LINK_MATCH = "match_lvl"          # None si la colonne n'existe pas
USE_LINK = True


def get_symbol_map(month, universe, trading_days):
    """DataFrame [date, permno, sym_root, sym_suffix, map_source] pour les jours du mois."""
    permnos = tuple(int(p) for p in universe["permno"].unique())
    start, end = min(trading_days), max(trading_days)
    link = pd.DataFrame()
    if USE_LINK:
        cols = f"{LINK_DATE} AS date, {LINK_PERMNO} AS permno, {LINK_ROOT} AS sym_root, {LINK_SUFFIX} AS sym_suffix"
        if LINK_MATCH:
            cols += f", {LINK_MATCH} AS match_lvl"
        try:
            link = sql(
                f"SELECT {cols} FROM {LINK_TABLE} "
                f"WHERE {LINK_DATE} BETWEEN %(s)s AND %(e)s AND {LINK_PERMNO} IN %(p)s",
                params={"s": start.date(), "e": end.date(), "p": permnos}, date_cols=["date"])
        except Exception as e:
            log.warning("lien CRSP-TAQ indisponible (%s) -> repli tickers CRSP", e)
            link = pd.DataFrame()
    if not link.empty:
        link["sym_root"] = link["sym_root"].str.strip()
        link["sym_suffix"] = link["sym_suffix"].fillna("").str.strip()
        if "match_lvl" in link:
            link = link.sort_values("match_lvl")
        link = link.drop_duplicates(["date", "permno"]).assign(map_source="link")

    # repli pour les (jour, PERMNO) absents du lien
    rows = []
    for r in universe.itertuples():
        root, suffix = crsp_to_taq_symbol(r.crsp_ticker, r.crsp_tradingsymbol, r.shareclass)
        for d in trading_days:
            if r.valid_from <= d <= r.valid_to:
                rows.append((d, int(r.permno), root, suffix))
    fb = pd.DataFrame(rows, columns=["date", "permno", "sym_root", "sym_suffix"]).assign(map_source="crsp")
    out = pd.concat([link[fb.columns] if not link.empty else fb.iloc[:0], fb], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out[out["date"].isin(trading_days)]
    return out.drop_duplicates(["date", "permno"], keep="first").reset_index(drop=True)


# ---------------------------------------------------------------- NBBO
# 5. Grille 15 min de midquotes NBBO pour un jour.
#    Points : 09:30, 09:45, ..., clôture (16:00 ou 13:00 les demi-séances) -> 27 points / jour normal.
#    09:30 = première cotation NBBO valide à partir de l'ouverture (fenêtre OPEN_WINDOW).
#    t > 09:30 = dernière NBBO valide strictement avant t, propagée si la NBBO n'a pas changé.
FREQ = "15min"
OPEN_WINDOW = "00:05:00"      # cherche la cotation d'ouverture dans [09:30, 09:35)
BATCH_SIZE = 1000             # symboles par requête (1000 = tout l'univers en une requête/jour)


def _nbbo_query(table, roots, seconds, market_close):
    """UNE requête par jour (une seule lecture de la table) :
    - dernière NBBO valide de chaque tranche de 15 min, étiquetée à la fin de la tranche ;
    - première NBBO valide de [09:30, 09:30 + OPEN_WINDOW), étiquetée bucket_end = -1."""
    sql_ = f"""
        WITH q AS (
            SELECT sym_root, sym_suffix, time_m,
                   best_bid::float8 AS bid, best_ask::float8 AS ask,
                   (FLOOR(EXTRACT(EPOCH FROM time_m) / %(seconds)s) + 1) * %(seconds)s AS bucket_end
            FROM {table}
            WHERE sym_root IN %(roots)s
              AND time_m >= '09:30:00' AND time_m < %(close)s
              AND best_bid > 0 AND best_ask >= best_bid
        )
        (SELECT DISTINCT ON (sym_root, sym_suffix, bucket_end)
                sym_root, sym_suffix, bucket_end, time_m AS quote_time, bid, ask
         FROM q
         ORDER BY sym_root, sym_suffix, bucket_end, time_m DESC)
        UNION ALL
        (SELECT DISTINCT ON (sym_root, sym_suffix)
                sym_root, sym_suffix, -1 AS bucket_end, time_m AS quote_time, bid, ask
         FROM q
         WHERE time_m < %(open_end)s
         ORDER BY sym_root, sym_suffix, time_m ASC)
    """
    params = {"roots": roots, "seconds": seconds, "close": market_close,
              "open_end": str(pd.Timedelta("09:30:00") + pd.Timedelta(OPEN_WINDOW))[-8:]}
    return sql_, params


def build_day_grid(date, symmap_day, last, opn, freq, market_close):
    """Assemble la grille d'un jour à partir des deux résultats SQL (testable hors WRDS)."""
    date = pd.Timestamp(date).normalize()
    close_ts = date + pd.Timedelta(market_close)
    grid_times = pd.date_range(date + pd.Timedelta("09:30:00"), close_ts, freq=freq)
    for d in (last, opn):
        d["sym_root"] = d["sym_root"].str.strip()
        d["sym_suffix"] = d["sym_suffix"].fillna("").str.strip()
    last = last.assign(grid_time=date + pd.to_timedelta(last["bucket_end"].astype(float), unit="s"))
    opn = opn.assign(grid_time=grid_times[0])
    q = pd.concat([opn, last.drop(columns="bucket_end")], ignore_index=True)
    q["quote_time"] = date + pd.to_timedelta(q["quote_time"].astype(str))
    q[["bid", "ask"]] = q[["bid", "ask"]].astype(float)      # NUMERIC TAQ -> Decimal sinon

    sm = symmap_day[["permno", "sym_root", "sym_suffix", "map_source"]]
    q = q.merge(sm, on=["sym_root", "sym_suffix"], how="inner")
    full = (sm.assign(_k=1).merge(pd.DataFrame({"grid_time": grid_times, "_k": 1}), on="_k").drop(columns="_k"))
    full = full.merge(q[["permno", "grid_time", "quote_time", "bid", "ask"]], on=["permno", "grid_time"], how="left")
    full = full.sort_values(["permno", "grid_time"])
    # la NBBO reste valable tant qu'elle ne change pas : propagation dans la journée
    full[["quote_time", "bid", "ask"]] = full.groupby("permno")[["quote_time", "bid", "ask"]].ffill()
    full["date"] = date
    full["stale_sec"] = (full["grid_time"] - full["quote_time"]).dt.total_seconds()
    full["mid"] = (full["bid"] + full["ask"]) / 2
    full["spread"] = full["ask"] - full["bid"]
    full["ret_15m"] = np.log(full["mid"]).groupby(full["permno"]).diff()
    full["grid_time"] = full["grid_time"].dt.tz_localize("America/New_York")
    full["quote_time"] = full["quote_time"].dt.tz_localize("America/New_York")
    return full.reset_index(drop=True)


def get_nbbo_day(date, symmap_day, freq=FREQ, market_close="16:00:00"):
    date = pd.Timestamp(date)
    table = f"taqm_{date.year}.complete_nbbo_{date:%Y%m%d}"
    seconds = int(pd.Timedelta(freq).total_seconds())
    roots_all = sorted(symmap_day["sym_root"].unique())
    parts = []
    for i in range(0, len(roots_all), BATCH_SIZE):
        query, params = _nbbo_query(table, tuple(roots_all[i:i + BATCH_SIZE]), seconds, market_close)
        parts.append(sql(query, params=params))
    res = pd.concat(parts, ignore_index=True)
    res["bucket_end"] = res["bucket_end"].astype(float)
    is_open = res["bucket_end"] < 0
    return build_day_grid(date, symmap_day, res[~is_open].copy(),
                          res[is_open].drop(columns="bucket_end").copy(), freq, market_close)


# ---------------------------------------------------------------- overnight
# 7. Rendement overnight (clôture 16:00/13:00 de la veille -> 09:30), mois par mois en
#    reportant la dernière clôture d'un mois sur l'autre. Brut : NON ajusté des splits/dividendes.
#    Recommandé : laisser brut et régulariser après coup avec adjust_overnight.py.
#    Si CRSP_DSF_DIR est renseigné, les jours de split (dlyfacprc != 1) sont mis à NaN.
CRSP_DSF_DIR = None        # fixé par --crsp-dsf-dir
CRSP_FAC_COL = "dlyfacprc" # facteur d'ajustement de prix CRSP CIZ (vérifier le nom de colonne)


def load_split_days():
    if not CRSP_DSF_DIR:
        return None
    parts = []
    for f in sorted(Path(CRSP_DSF_DIR).glob("crsp_dsf_v2_*.parquet")):
        d = pd.read_parquet(f, columns=["permno", "dlycaldt", CRSP_FAC_COL])
        f_ = d[CRSP_FAC_COL].astype(float)
        parts.append(d[f_.notna() & (f_ != 1)])          # CIZ : 1 = pas d'événement (10 = split 10:1)
    s = pd.concat(parts)
    return set(zip(s["permno"].astype(int), pd.to_datetime(s["dlycaldt"]).dt.normalize()))


def add_overnight(month_df, prev_close, split_days=None):
    """prev_close : {permno: log mid de clôture de la dernière séance du mois précédent}.
    Un PERMNO absent la veille -> NaN (pas de saut sur plusieurs séances)."""
    m = month_df.sort_values(["permno", "grid_time"]).copy()
    m["logmid"] = np.log(m["mid"])
    days = sorted(m["date"].unique())
    first = m.groupby(["permno", "date"]).head(1).set_index(["permno", "date"])["logmid"]
    last = m.groupby(["permno", "date"]).tail(1).set_index(["permno", "date"])["logmid"]
    on = {}
    for i, d in enumerate(days):
        for p, lo in first.xs(d, level="date").items():
            pc = prev_close.get(p) if i == 0 else last.get((p, days[i - 1]))
            on[(p, d)] = lo - pc if pc is not None else np.nan
    is_open = m.groupby(["permno", "date"]).cumcount() == 0
    key = list(zip(m["permno"], m["date"]))
    m["ret_overnight_raw"] = np.where(is_open, [on.get(k, np.nan) for k in key], np.nan)
    if split_days:
        hit = np.array([k in split_days for k in key])
        m.loc[is_open & hit, "ret_overnight_raw"] = np.nan
    new_close = last.xs(days[-1], level="date").to_dict()
    return m.drop(columns="logmid"), new_close


# ---------------------------------------------------------------- boucle mensuelle
def download(df, start, end, out_dir):
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    failed = {}
    for month in pd.period_range(start, end, freq="M"):
        month_str = str(month)
        filepath = out_dir / f"{month_str}.parquet"
        if filepath.exists():
            continue
        t0 = time.time()
        try:
            universe = get_monthly_universe(df, month_str)
            schedule = nyse.schedule(start_date=month.start_time.date(), end_date=month.end_time.date())
            days = pd.DatetimeIndex(schedule.index).normalize()
            symmap = get_symbol_map(month_str, universe, days)
            log.info("%s : %d PERMNO, %d séances, lien CRSP-TAQ %d/%d (jour, PERMNO)", month_str,
                     universe["permno"].nunique(), len(days), int((symmap["map_source"] == "link").sum()), len(symmap))
            monthly, coverage = [], []
            for date, session in schedule.iterrows():
                d = pd.Timestamp(date).normalize()
                market_close = session["market_close"].tz_convert("America/New_York").strftime("%H:%M:%S")
                sm = symmap[symmap["date"] == d]
                t_day = time.time()
                try:
                    day = get_nbbo_day(d, sm, FREQ, market_close)
                except Exception as e:
                    failed[str(d.date())] = str(e)[:200]
                    log.error("jour %s : %s", d.date(), e)
                    continue
                got = day.dropna(subset=["mid"])["permno"].nunique()
                log.info("  %s : %d/%d PERMNO avec mid, %.0fs", d.date(), got, len(sm), time.time() - t_day)
                coverage.append((d.date(), len(sm), got, int((sm["map_source"] == "crsp").sum())))
                monthly.append(day)
            if not monthly:
                raise ValueError("aucun jour récupéré")
            out = pd.concat(monthly, ignore_index=True)
            tmp = filepath.with_suffix(".tmp.parquet")
            out.to_parquet(tmp, index=False, compression="snappy")
            tmp.replace(filepath)
            cov = pd.DataFrame(coverage, columns=["date", "permno_expected", "permno_with_mid", "n_crsp_fallback"])
            cov.to_csv(out_dir / f"{month_str}_coverage.csv", index=False)
            log.info("%s : %d PERMNO, %d jours, %s lignes, couverture min %.1f%%, %.0fs",
                     month_str, universe["permno"].nunique(), len(monthly), f"{len(out):,}",
                     100 * (cov["permno_with_mid"] / cov["permno_expected"]).min(), time.time() - t0)
        except Exception as e:
            failed[month_str] = str(e)[:200]
            log.error("mois %s : %s", month_str, e)
    if failed:
        ff = out_dir / f"failed_{start}_{end}.csv"     # un fichier par plage : pas de collision en parallèle
        pd.Series(failed, name="error").to_csv(ff, index_label="period")
        log.warning("%d échecs -> %s", len(failed), ff)
    return failed


def overnight(out_dir):
    split_days = load_split_days()
    prev_close = {}
    final_dir = out_dir / "with_overnight"
    final_dir.mkdir(exist_ok=True)
    for f in sorted(out_dir.glob("????-??.parquet")):
        m = pd.read_parquet(f)
        m["date"] = pd.to_datetime(m["date"]).dt.normalize()
        m, prev_close = add_overnight(m, prev_close, split_days)
        m.to_parquet(final_dir / f.name, index=False)
        log.info("%s : %s rendements overnight", f.name, f"{m['ret_overnight_raw'].notna().sum():,}")


def check():
    print(sql("SELECT table_name FROM information_schema.tables "
              "WHERE table_schema = 'wrdsapps_link_crsp_taqm'"))
    print(conn.describe_table("wrdsapps_link_crsp_taqm", "taqmclink"))
    print(conn.describe_table("taqm_2008", "complete_nbbo_20080102"))
    # Vue ou table ? (une vue cache les index de la table sous-jacente dans pg_indexes)
    print(sql("SELECT c.relname, c.relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
              "WHERE n.nspname = 'taqm_2018' AND c.relname = 'complete_nbbo_20180102'"))
    # Test décisif : plan d'exécution + durée pour UN symbole sur un jour.
    q = ("SELECT time_m, best_bid, best_ask FROM taqm_2018.complete_nbbo_20180102 "
         "WHERE sym_root = 'AAPL' AND time_m >= '09:30:00' AND time_m < '16:00:00'")
    plan = sql("EXPLAIN " + q)
    print("\n".join(plan.iloc[:, 0].astype(str).tolist()))
    t0 = time.time()
    n = len(sql(q))
    print(f"AAPL 2018-01-02 : {n:,} lignes en {time.time() - t0:.1f}s "
          "(quelques secondes = index utilisé ; ~100 s = lecture complète de la table)")


def main():
    global WRDS_USER, CRSP_DSF_DIR, FREQ, BATCH_SIZE
    a = argparse.ArgumentParser()
    a.add_argument("--segments", default="sp500_pit_identifiers_segments.csv")
    a.add_argument("--start", default="2008-01")
    a.add_argument("--end", default="2026-06")
    a.add_argument("--out", default=None, help="défaut data_wrds_<freq>")
    a.add_argument("--freq", default=FREQ)
    a.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    a.add_argument("--wrds-user", default=os.environ.get("WRDS_USERNAME"))
    a.add_argument("--crsp-dsf-dir", default=None)
    a.add_argument("--init-pgpass", action="store_true")
    a.add_argument("--check", action="store_true")
    a.add_argument("--overnight-only", action="store_true")
    a.add_argument("--no-overnight", action="store_true")
    args = a.parse_args()

    FREQ, BATCH_SIZE, CRSP_DSF_DIR = args.freq, args.batch_size, args.crsp_dsf_dir
    out_dir = Path(args.out or f"data_wrds_{FREQ}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "data_getter.log")])

    if args.overnight_only:
        overnight(out_dir)
        return
    WRDS_USER = args.wrds_user
    if not WRDS_USER:
        sys.exit("--wrds-user ou WRDS_USERNAME requis")
    if args.init_pgpass:
        init_pgpass(WRDS_USER)
        return
    connect()
    if args.check:
        check()
        return
    df = load_segments(args.segments)
    log.info("%d segments, %d PERMNO ; %s -> %s", len(df), df["permno"].nunique(), args.start, args.end)
    download(df, args.start, args.end, out_dir)
    if not args.no_overnight:
        overnight(out_dir)


if __name__ == "__main__":
    main()