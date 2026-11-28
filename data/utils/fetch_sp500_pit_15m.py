#!/usr/bin/env python3
"""
Grille 15 min de midquotes pour l'univers S&P 500 point-in-time (Databento EQUS.MINI, bbo-1m).

Approche Bollerslev, Li & Tang (Management Science 2026) : rendements 15 min calculés sur
le MIDQUOTE (pas sur les prix de transaction), 27 points par jour 09:30 -> 16:00 ET
(26 rendements intrajournaliers) + un rendement overnight (16:00 veille -> 09:30).

Entrée
  sp500_pit_identifiers_segments.csv : une ligne = PERMNO x période où le titre est membre
  et où ticker/CUSIP sont stables, donc le symbole est pris AS-OF. Aucun filtre de nom
  (share code, prix, nombre de transactions) n'est appliqué ici.

Sortie (un parquet par PERMNO dans --out)
  Colonnes brutes Databento : ts_recv, ts_event, instrument_id, symbol (= symbol_used),
  bid_px_00, ask_px_00, bid_sz_00, ask_sz_00, price, size, side
  + permno, symbol_requested, date, grid_time (ET), mid, spread, crossed, stale_sec,
    ret_15m (log), ret_overnight_raw (log, NON ajusté), gvkey, cusip8
  --keep-raw écrit aussi le bbo-1m complet dans --out/raw/.

Usage
  export DATABENTO_API_KEY=db-xxxxxxxx
  pip install databento pandas pyarrow
  python fetch_sp500_pit_15m.py --segments sp500_pit_identifiers_segments.csv --dry-run
  python fetch_sp500_pit_15m.py --segments sp500_pit_identifiers_segments.csv --out mid15m --limit 5
  python fetch_sp500_pit_15m.py --segments sp500_pit_identifiers_segments.csv --out mid15m

Points à vérifier avant le run complet : voir la liste fournie avec le script
(sémantique du bbo-1m, date de début, BBO incluant les odd lots, demi-séances, etc.).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from saf.universe import clip_segments, databento_candidates, load_segments

NY = "America/New_York"
SCHEMA = "bbo-1m"
RAW_COLS = ["ts_recv", "ts_event", "instrument_id", "symbol",
            "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "price", "size", "side"]


def load_dotenv(path=Path(__file__).with_name(".env")):
    """Lit KEY=VALUE depuis .env (à côté du script) sans écraser l'environnement."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--segments", required=True)
    p.add_argument("--dataset", default="EQUS.MINI")
    p.add_argument("--start", default=None, help="YYYY-MM-DD, défaut = début du dataset")
    p.add_argument("--end", default=None, help="YYYY-MM-DD inclus, défaut = fin du dataset")
    p.add_argument("--out", default="mid15m")
    p.add_argument("--freq", default="15min")
    p.add_argument("--max-stale", type=float, default=300,
                   help="secondes : au-delà, la cotation est jugée périmée -> mid = NaN")
    p.add_argument("--keep-raw", action="store_true", help="sauver aussi le bbo-1m complet")
    p.add_argument("--dry-run", action="store_true", help="estime le coût sans télécharger")
    p.add_argument("--limit", type=int, default=None, help="limiter à N PERMNO (test)")
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()


def year_chunks(start, end):
    """Découpe [start, end) en tranches annuelles (mémoire + reprise sur erreur)."""
    cur = start
    while cur < end:
        nxt = min(pd.Timestamp(year=cur.year + 1, month=1, day=1), end)
        yield cur, nxt
        cur = nxt


# ---------------------------------------------------------------- téléchargement
def fetch_bbo(client, dataset, symbol, start, end, retries, sleep):
    for k in range(retries):
        try:
            data = client.timeseries.get_range(
                dataset=dataset, symbols=[symbol], stype_in="raw_symbol", schema=SCHEMA,
                # bornes à minuit ET (pas UTC) : sinon la fenêtre commence à 19:00/20:00 ET
                # la veille et crée une fausse séance vide qui écrase la vraie au raccord.
                start=pd.Timestamp(start.date(), tz=NY).isoformat(),
                end=pd.Timestamp(end.date(), tz=NY).isoformat())
            try:
                df = data.to_df(price_type="float", map_symbols=True)
            except TypeError:                     # anciennes versions du client
                df = data.to_df()
            return df.reset_index()
        except Exception as e:
            msg = str(e)
            if "symbol" in msg.lower() and ("not found" in msg.lower() or "resolve" in msg.lower()):
                return pd.DataFrame()
            if k == retries - 1:
                raise
            time.sleep(sleep * 10 * (k + 1))
    return pd.DataFrame()


# ---------------------------------------------------------------- grille 15 min
def snap_grid(raw, freq, max_stale):
    """bbo-1m brut -> une ligne par point de grille 09:30..16:00 ET.

    09:30 : première cotation à partir de l'ouverture (évite la cotation de pré-ouverture).
    Autres points : dernière cotation <= point de grille.
    """
    if raw.empty:
        return raw
    raw = raw.copy()
    for c in ("ts_recv", "ts_event"):
        raw[c] = pd.to_datetime(raw[c], utc=True)
    raw = raw.sort_values("ts_recv")
    raw["_t"] = raw["ts_recv"].dt.tz_convert(NY)
    # séances = jours avec au moins une cotation pendant 09:30-16:00 ET
    # (exclut les jours ne contenant que du pré/post-marché, ex. bord de fenêtre)
    hm = raw["_t"].dt.hour * 60 + raw["_t"].dt.minute
    days = raw.loc[(hm >= 570) & (hm <= 960), "_t"].dt.normalize().unique()

    grid = []
    for d in days:
        g = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), d + pd.Timedelta(hours=16), freq=freq)
        grid.append(pd.DataFrame({"grid_time": g, "date": d.date()}))
    grid = pd.concat(grid, ignore_index=True).sort_values("grid_time")

    keep = [c for c in RAW_COLS if c in raw.columns] + ["_t"]
    back = pd.merge_asof(grid, raw[keep], left_on="grid_time", right_on="_t", direction="backward")
    fwd = pd.merge_asof(grid, raw[keep], left_on="grid_time", right_on="_t", direction="forward")
    is_open = back["grid_time"].dt.strftime("%H:%M") == "09:30"
    out = back.where(~is_open, fwd)
    out["stale_sec"] = (out["grid_time"] - out["_t"]).dt.total_seconds().abs()
    out = out.drop(columns="_t")

    bid, ask = out["bid_px_00"].astype(float), out["ask_px_00"].astype(float)
    ok = (bid > 0) & (ask > 0) & (out["stale_sec"] <= max_stale)
    out["crossed"] = ok & (bid > ask)
    out["mid"] = np.where(ok & ~out["crossed"], (bid + ask) / 2, np.nan)
    out["spread"] = np.where(ok, ask - bid, np.nan)
    return out


def add_returns(df):
    """ret_15m dans la journée ; ret_overnight_raw sur la ligne 09:30 (non ajusté des OST)."""
    df = df.sort_values("grid_time").reset_index(drop=True)
    lm = np.log(df["mid"])
    df["ret_15m"] = lm.groupby(df["date"]).diff()
    is_open = df["grid_time"].dt.strftime("%H:%M") == "09:30"
    is_close = df["grid_time"].dt.strftime("%H:%M") == "16:00"
    # clôture (16:00) de la séance précédente, rattachée à la séance suivante
    closes = df.loc[is_close, ["date"]].assign(c=lm[is_close]).drop_duplicates("date", keep="last")
    closes["next_date"] = closes["date"].shift(-1)
    m = dict(zip(closes["next_date"], closes["c"]))
    df["ret_overnight_raw"] = np.where(is_open, lm - df["date"].map(m).astype(float), np.nan)
    return df


# ---------------------------------------------------------------- main
def main():
    a = parse_args()
    import databento as db
    load_dotenv()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("DATABENTO_API_KEY manquant")
    client = db.Historical(key)

    rng = client.metadata.get_dataset_range(dataset=a.dataset)
    norm = lambda x: pd.Timestamp(x).tz_localize(None).normalize() if pd.Timestamp(x).tzinfo else pd.Timestamp(x).normalize()
    ds_lo, ds_hi = norm(rng["start"]), norm(rng["end"])
    lo = max(ds_lo, pd.Timestamp(a.start)) if a.start else ds_lo
    hi = min(ds_hi, pd.Timestamp(a.end) + pd.Timedelta(days=1)) if a.end else ds_hi
    print(f"{a.dataset} {SCHEMA}: disponible {ds_lo.date()} -> {ds_hi.date()} ; fenêtre {lo.date()} -> {hi.date()}")

    seg = clip_segments(load_segments(a.segments), lo, hi)
    permnos = seg["permno"].unique()
    if a.limit:
        permnos = permnos[: a.limit]
        seg = seg[seg["permno"].isin(permnos)]
    print(f"{len(permnos)} PERMNO, {len(seg)} segments")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.keep_raw:
        (out_dir / "raw").mkdir(exist_ok=True)
    issues = []

    if a.dry_run:
        total = 0.0
        for r in seg.itertuples():
            sym = databento_candidates(r.crsp_ticker, r.crsp_tradingsymbol, r.shareclass)[0]
            try:
                total += client.metadata.get_cost(dataset=a.dataset, symbols=[sym], stype_in="raw_symbol",
                                                  schema=SCHEMA, start=r.req_start.date().isoformat(),
                                                  end=r.req_end.date().isoformat())
            except Exception as e:
                issues.append((r.permno, sym, r.req_start.date(), r.req_end.date(), str(e)[:150]))
        print(f"Coût estimé {SCHEMA}: ${total:,.2f}")
        if issues:
            pd.DataFrame(issues, columns=["permno", "symbol", "start", "end", "error"]).to_csv(out_dir / "unresolved.csv", index=False)
            print(f"{len(issues)} segments non estimables -> {out_dir/'unresolved.csv'}")
        return

    for n, (permno, g) in enumerate(seg.groupby("permno"), 1):
        f = out_dir / f"{permno}.parquet"
        if f.exists():
            continue
        grids, raws = [], []
        for r in g.itertuples():
            cands = databento_candidates(r.crsp_ticker, r.crsp_tradingsymbol, r.shareclass)
            for s0, s1 in year_chunks(r.req_start, r.req_end):
                raw, used = pd.DataFrame(), None
                for sym in cands:
                    try:
                        raw = fetch_bbo(client, a.dataset, sym, s0, s1, a.retries, a.sleep)
                    except Exception as e:
                        issues.append((permno, sym, s0.date(), s1.date(), str(e)[:150]))
                        raw = pd.DataFrame()
                    if not raw.empty:
                        used = sym
                        break
                    time.sleep(a.sleep)
                if raw.empty:
                    issues.append((permno, "|".join(cands), s0.date(), s1.date(), "no data"))
                    continue
                if "symbol" not in raw.columns:
                    raw["symbol"] = used
                if a.keep_raw:
                    raws.append(raw.assign(permno=permno))
                gr = snap_grid(raw, a.freq, a.max_stale)
                gr["symbol_requested"] = r.crsp_ticker
                gr["symbol"] = used
                gr["gvkey"], gr["cusip8"] = r.gvkey, r.cusip8
                grids.append(gr)
                time.sleep(a.sleep)
        if grids:
            res = pd.concat(grids, ignore_index=True).drop_duplicates("grid_time", keep="last")
            res = add_returns(res)
            res.insert(0, "permno", permno)
            res.to_parquet(f, index=False)
            if a.keep_raw:
                pd.concat(raws, ignore_index=True).to_parquet(out_dir / "raw" / f"{permno}.parquet", index=False)
            print(f"[{n}/{len(permnos)}] {permno}: {res['date'].nunique()} jours, "
                  f"mid manquant {res['mid'].isna().mean():.1%}")
        else:
            print(f"[{n}/{len(permnos)}] {permno}: aucune donnée")

    if issues:
        pd.DataFrame(issues, columns=["permno", "symbol", "start", "end", "error"]).to_csv(out_dir / "unresolved.csv", index=False)
        print(f"{len(issues)} problèmes -> {out_dir/'unresolved.csv'}")


if __name__ == "__main__":
    main()
