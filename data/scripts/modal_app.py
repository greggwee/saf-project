"""
Lance data_getter.py sur Modal : un conteneur par mois, résultats sur un Volume Modal.

Mise en place (une fois, en local)
  pip install modal && modal setup
  modal secret create wrds WRDS_USERNAME=ethan099 WRDS_PASSWORD='...'
  modal volume create taq-15m
  # Splits/dividendes : régularisés a posteriori en local avec adjust_overnight.py
  # (ne pas mettre CRSP sur le volume, sinon les jours de split passent à NaN)

Fichiers locaux embarqués : data/scripts/data_getter.py, saf/, data/universe/sp500_pit_identifiers_segments.csv

Lancement
  modal run modal_app.py --check                                # schémas WRDS
  modal run modal_app.py --start 2016-06 --end 2016-06          # test 1 mois
  modal run --detach modal_app.py --spawn --start 2008-01 --end 2026-06   # run complet, Mac libre
  modal run modal_app.py --status --start 2008-01 --end 2026-06           # avancement
  modal run modal_app.py --overnight-only                                 # à la fin

Récupérer les résultats puis régulariser les splits/dividendes en local
  modal volume get taq-15m /data_wrds_15min ./
  python adjust_overnight.py --data-dir data_wrds_15min/with_overnight --crsp-dir crsp_dsf_v2
"""
import sys
from pathlib import Path

import modal

APP_NAME = "sp500-taq-15m"
VOLUME_NAME = "taq-15m"
MOUNT = "/data"
OUT_SUBDIR = "data_wrds_15min"
CRSP_SUBDIR = "crsp_dsf_v2"
SEGMENTS = "sp500_pit_identifiers_segments.csv"

# Conteneurs simultanés = connexions WRDS simultanées. WRDS limite le nombre de connexions
# par utilisateur et chaque nouveau conteneur peut déclencher une validation Duo : rester bas.
MAX_CONTAINERS = 4
MONTH_TIMEOUT = 3 * 3600          # mesuré : ~130 s/jour -> ~45 min/mois ; marge x4

here = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Versions figées : wrds 3.3.0 exige pandas >=2.2,<2.3 (pandas plus récent -> erreur
    # "'Connection' object has no attribute 'cursor'"). Combinaison vérifiée avec pip.
    .pip_install("wrds==3.3.0", "pandas==2.2.3", "pandas_market_calendars==5.5.0",
                 "sqlalchemy>=2,<2.1", "numpy<2.3", "pyarrow")
    .add_local_file(here / "data_getter.py", "/root/data_getter.py")
    .add_local_dir(here.parents[1] / "saf", "/root/saf")   # shared library (saf.universe, ...)
    .add_local_file(here.parent / "universe" / SEGMENTS, f"/root/{SEGMENTS}")
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
wrds_secret = modal.Secret.from_name("wrds", required_keys=["WRDS_USERNAME", "WRDS_PASSWORD"])

# IP de sortie fixe (si WRDS/Duo bloque les IP changeantes) : créer un Proxy dans le tableau de
# bord Modal puis décommenter `proxy=PROXY` dans les décorateurs ci-dessous.
# PROXY = modal.Proxy.from_name("wrds-proxy")


def _setup():
    """Écrit ~/.pgpass depuis le secret (aucune invite) et configure data_getter."""
    import logging
    import os
    import stat

    sys.path.insert(0, "/root")
    import data_getter as dg

    user, pwd = os.environ["WRDS_USERNAME"], os.environ["WRDS_PASSWORD"]
    esc = lambda s: s.replace("\\", "\\\\").replace(":", "\\:")
    pg = Path.home() / ".pgpass"
    pg.write_text(f"{dg.PGPASS_HOST}:{dg.PGPASS_PORT}:wrds:{esc(user)}:{esc(pwd)}\n")
    pg.chmod(stat.S_IRUSR | stat.S_IWUSR)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout, force=True)
    dg.WRDS_USER = user
    crsp = Path(MOUNT) / CRSP_SUBDIR
    dg.CRSP_DSF_DIR = str(crsp) if crsp.exists() else None
    return dg


@app.function(secrets=[wrds_secret], volumes={MOUNT: vol}, timeout=MONTH_TIMEOUT,
              max_containers=MAX_CONTAINERS, cpu=1, memory=2048)   # , proxy=PROXY
def run_month(month: str) -> dict:
    dg = _setup()
    out_dir = Path(MOUNT) / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    vol.reload()
    if (out_dir / f"{month}.parquet").exists():
        return {"month": month, "status": "skip"}
    dg.connect()
    df = dg.load_segments(f"/root/{SEGMENTS}")
    failed = dg.download(df, month, month, out_dir)
    vol.commit()
    ok = (out_dir / f"{month}.parquet").exists()
    return {"month": month, "status": "ok" if ok else "failed", "n_failed": len(failed), "failed": failed}


@app.function(secrets=[wrds_secret], volumes={MOUNT: vol}, timeout=6 * 3600, cpu=2, memory=16384)
def run_overnight():
    """Séquentiel : la clôture de chaque mois sert à l'ouverture du suivant."""
    dg = _setup()
    vol.reload()
    dg.overnight(Path(MOUNT) / OUT_SUBDIR)
    vol.commit()


@app.function(secrets=[wrds_secret], timeout=600)   # , proxy=PROXY
def run_check():
    dg = _setup()
    dg.connect()
    dg.check()


@app.function(volumes={MOUNT: vol}, timeout=300)
def run_status(start: str, end: str) -> dict:
    import pandas as pd
    vol.reload()
    out_dir = Path(MOUNT) / OUT_SUBDIR
    months = [str(m) for m in pd.period_range(start, end, freq="M")]
    done = [m for m in months if (out_dir / f"{m}.parquet").exists()]
    failed = {}
    for f in sorted(out_dir.glob("failed_*.csv")):
        for _, r in pd.read_csv(f).iterrows():
            failed[str(r["period"])] = str(r["error"])[:150]
    return {"done": done, "todo": [m for m in months if m not in done], "failed": failed}


@app.local_entrypoint()
def main(start: str = "2016-06", end: str = "2016-06", check: bool = False,
         overnight_only: bool = False, no_overnight: bool = False,
         spawn: bool = False, status: bool = False):
    import pandas as pd

    if check:
        run_check.remote()
        return
    if status:
        s = run_status.remote(start, end)
        print(f"terminés {len(s['done'])}, restants {len(s['todo'])}, échecs {len(s['failed'])}")
        if s["todo"]:
            print("restants :", " ".join(s["todo"][:30]), "..." if len(s["todo"]) > 30 else "")
        for k, v in s["failed"].items():
            print(" ", k, v)
        return
    if spawn:
        # Envoie tous les mois puis rend la main : le Mac peut s'éteindre (avec --detach).
        months = [str(m) for m in pd.period_range(start, end, freq="M")]
        run_month.spawn_map(months)
        print(f"{len(months)} mois envoyés ({MAX_CONTAINERS} conteneurs max). Suivi : "
              f"modal run modal_app.py --status --start {start} --end {end} ; "
              "puis, une fois tout terminé : modal run modal_app.py --overnight-only")
        return
    if not overnight_only:
        months = [str(m) for m in pd.period_range(start, end, freq="M")]
        print(f"{len(months)} mois, {MAX_CONTAINERS} conteneurs max")
        results = []
        for r in run_month.map(months, order_outputs=False, return_exceptions=True):
            if isinstance(r, Exception):
                print("ERREUR conteneur :", repr(r)[:300])
                results.append({"month": None, "status": "error", "error": repr(r)[:300]})
            else:
                print(r["month"], r["status"], f"{r.get('n_failed', 0)} échecs")
                results.append(r)
        pd.DataFrame(results).to_csv(f"modal_run_{start}_{end}.csv", index=False)
        bad = [r for r in results if r["status"] in ("failed", "error") or r.get("n_failed")]
        print(f"Terminé : {len(bad)} mois avec erreurs -> modal_run_{start}_{end}.csv")
    if not no_overnight:
        run_overnight.remote()
        print("Overnight calculés -> volume", VOLUME_NAME, f"/{OUT_SUBDIR}/with_overnight")