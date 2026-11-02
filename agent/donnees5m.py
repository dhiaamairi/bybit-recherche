"""Historique 5 minutes des cryptos + financement réel des contrats à terme (fichiers publics Binance).

    python -m agent.donnees5m            -> data/5m/<ACTIF>.npz (prix) et data/5m/<ACTIF>-fund.npz (financement)

Binance publie un fichier par mois. Les prix sont en float32 pour tenir en mémoire (930 000 bougies par actif).
Depuis 2025, les dates des fichiers spot sont en microsecondes : elles sont ramenées en millisecondes.
"""
from __future__ import annotations

import csv
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ACTIFS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"]
BASE = "https://data.binance.vision/data"
DOSSIER = Path("data/5m")
PREMIER = (2018, 1)
DERNIER = (2026, 10)


def mois() -> list[tuple[int, int]]:
    out, (y, m) = [], PREMIER
    while (y, m) <= DERNIER:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _lire(url: str, essais: int = 4) -> list[list[str]] | None:
    """Renvoie les lignes du CSV du mois, ou None si ce mois n'existe pas (actif pas encore coté)."""
    for k in range(essais):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                texte = z.read(z.namelist()[0]).decode("utf-8")
            return [ligne for ligne in csv.reader(io.StringIO(texte)) if ligne]
        except (requests.RequestException, zipfile.BadZipFile):
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"téléchargement impossible : {url}")


def _bougies(actif: str, y: int, m: int) -> np.ndarray | None:
    lignes = _lire(f"{BASE}/spot/monthly/klines/{actif}/5m/{actif}-5m-{y}-{m:02d}.zip")
    if not lignes:
        return None
    if not lignes[0][0].isdigit():               # ligne de titres, présente dans certains fichiers
        lignes = lignes[1:]
    a = np.array([[float(x) for x in (li[0], li[1], li[2], li[3], li[4], li[7])] for li in lignes], dtype=np.float64)
    a[:, 0] = np.where(a[:, 0] > 1e14, a[:, 0] / 1000, a[:, 0])
    return a


def _financement(actif: str, y: int, m: int) -> np.ndarray | None:
    lignes = _lire(f"{BASE}/futures/um/monthly/fundingRate/{actif}/{actif}-fundingRate-{y}-{m:02d}.zip")
    if not lignes:
        return None
    if not lignes[0][0].isdigit():
        lignes = lignes[1:]
    return np.array([[float(li[0]), float(li[2])] for li in lignes], dtype=np.float64)


def telecharger(actif: str, pool: ThreadPoolExecutor) -> str:
    cible = DOSSIER / f"{actif}.npz"
    if not cible.exists():
        morceaux = [p for p in pool.map(lambda ym, a=actif: _bougies(a, *ym), mois()) if p is not None]
        x = np.concatenate(morceaux)
        x = x[np.argsort(x[:, 0], kind="stable")]
        x = x[np.concatenate([[True], np.diff(x[:, 0]) > 0])]
        np.savez_compressed(cible, ts=x[:, 0].astype(np.int64), o=x[:, 1].astype(np.float32), h=x[:, 2].astype(np.float32),
                            l=x[:, 3].astype(np.float32), c=x[:, 4].astype(np.float32), qv=x[:, 5].astype(np.float32))
    fond = DOSSIER / f"{actif}-fund.npz"
    if not fond.exists():
        morceaux = [p for p in pool.map(lambda ym, a=actif: _financement(a, *ym), mois()) if p is not None]
        f = np.concatenate(morceaux) if morceaux else np.zeros((0, 2))
        np.savez_compressed(fond, ts=f[:, 0].astype(np.int64), rate=f[:, 1].astype(np.float32))
    d = np.load(cible)
    n = len(d["ts"])
    return f"{actif} : {n} bougies, {cible.stat().st_size / 1e6:.1f} Mo, financement {len(np.load(fond)['ts'])} points"


if __name__ == "__main__":
    DOSSIER.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        for a in sys.argv[1:] or ACTIFS:
            print(telecharger(a, ex), flush=True)
    print("TERMINE", flush=True)
