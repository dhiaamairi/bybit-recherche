"""L'idée du reel « FLASH Ai BOT » (03/11/2026) testée à la seconde : sur l'or, plusieurs trades ouverts dès que le prix
accélère, gain pris en quelques secondes ou stop. Simulation seulement.

Données : ticks publics de Dukascopy (prix d'achat ET prix de vente réels, plusieurs fois par seconde).
Coûts : on achète au prix de vente du courtier (ask), on revend au prix d'achat (bid), plus une commission par lot
(compte à écart serré, ~3,5 $ par lot et par sens). 1 lot = 100 onces.

    python -m agent.secondes telecharger 2024-04 2024-08
    python -m agent.secondes tester
"""
from __future__ import annotations

import io
import lzma
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numba as nb
import numpy as np
import requests

DOSSIER = Path("data/xau")
MANQUANTES: list[str] = []
URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
ECHELLE = 1000.0                    # les prix de l'or sont stockés en millièmes de dollar


def _heure(jour: date, h: int) -> np.ndarray | None:
    url = URL.format(y=jour.year, m=jour.month - 1, d=jour.day, h=h)      # mois comptés à partir de 0 chez Dukascopy
    for k in range(6):
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 404 or not r.content:
                return None
            r.raise_for_status()
            brut = lzma.decompress(r.content)
            n = len(brut) // 20
            if not n:
                return None
            x = np.frombuffer(brut[: n * 20], dtype=np.dtype([("t", ">i4"), ("ask", ">i4"), ("bid", ">i4"), ("va", ">f4"), ("vb", ">f4")]))
            t0 = int(datetime(jour.year, jour.month, jour.day, h, tzinfo=timezone.utc).timestamp() * 1000)
            return np.stack([t0 + x["t"].astype(np.int64), x["ask"].astype(np.int64), x["bid"].astype(np.int64)], axis=1)
        except (requests.RequestException, lzma.LZMAError):
            time.sleep(3 * (k + 1))
    MANQUANTES.append(url)                  # heure sautée et comptée, plutôt que tout arrêter
    return None


def telecharger(debut: str, fin: str) -> None:
    DOSSIER.mkdir(parents=True, exist_ok=True)
    d = date.fromisoformat(debut + "-01")
    f = date.fromisoformat(fin + "-01")
    with ThreadPoolExecutor(12) as ex:
        while d <= f:
            cible = DOSSIER / f"{d:%Y-%m}.npz"
            if not cible.exists():
                jours = [d + timedelta(days=i) for i in range(31) if (d + timedelta(days=i)).month == d.month]
                taches = [(j, h) for j in jours for h in range(24)]
                morceaux = [x for x in ex.map(lambda jh: _heure(*jh), taches) if x is not None]
                x = np.concatenate(morceaux)
                np.savez_compressed(cible, t=x[:, 0], ask=x[:, 1].astype(np.int32), bid=x[:, 2].astype(np.int32))
                print(f"{d:%Y-%m} : {len(x):,} ticks, {cible.stat().st_size / 1e6:.1f} Mo, heures manquantes {len(MANQUANTES)}".replace(",", " "), flush=True)
            d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def mois_entre(debut: str, fin: str) -> list[str]:
    d, f, out = date.fromisoformat(debut + "-01"), date.fromisoformat(fin + "-01"), []
    while d <= f:
        out.append(f"{d:%Y-%m}")
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def charger(mois: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, a, b = [], [], []
    for m in mois:
        z = np.load(DOSSIER / f"{m}.npz")
        t.append(z["t"])
        a.append(z["ask"])
        b.append(z["bid"])
    return np.concatenate(t), np.concatenate(a).astype(np.float64) / ECHELLE, np.concatenate(b).astype(np.float64) / ECHELLE


# ---------------------------------------------------------------- le moteur, tick par tick
@nb.njit(nogil=True)
def _un(T, ASK, BID, fen_ms, seuil, tp, sl, duree_ms, maxpos, espace, lots, commission, levier, sens_mode):
    """Une règle : si le prix a bougé d'au moins `seuil` $ en `fen_ms` ms, ouvrir une position dans ce sens (`sens_mode` 1)
    ou contre ce sens (-1), jusqu'à `maxpos` positions espacées d'au moins `espace` $. Chaque position se ferme à +tp $,
    à -sl $ ou après `duree_ms`. Capital 1 000 $, `lots` par position. Renvoie le capital final, le nombre de trades,
    le pire capital vu, et 1 si le compte a sauté (niveau de marge sous 50 %)."""
    n = len(T)
    eq = 1000.0
    ent = np.zeros(maxpos)
    sns = np.zeros(maxpos, np.int64)
    tdeb = np.zeros(maxpos, np.int64)
    actif = np.zeros(maxpos, np.bool_)
    nbt = 0
    pire = eq
    j = 0
    dernier = -1e18
    once = lots * 100.0
    for i in range(n):
        t = T[i]
        ask, bid = ASK[i], BID[i]
        mid = 0.5 * (ask + bid)
        while T[j] < t - fen_ms:
            j += 1
        latent = 0.0
        ouvertes = 0
        for k in range(maxpos):
            if not actif[k]:
                continue
            s = sns[k]
            px = bid if s > 0 else ask
            g = s * (px - ent[k])
            if g >= tp or g <= -sl or t - tdeb[k] >= duree_ms:
                eq += g * once - commission * lots
                actif[k] = False
                nbt += 1
            else:
                latent += g * once
                ouvertes += 1
        marge = ouvertes * once * mid / levier
        if ouvertes > 0 and (eq + latent) < 0.5 * marge:
            return eq + latent, nbt, min(pire, eq + latent), 1
        pire = min(pire, eq + latent)
        if eq + latent <= 0:
            return 0.0, nbt, 0.0, 1
        mouv = mid - 0.5 * (ASK[j] + BID[j])
        if abs(mouv) >= seuil and ouvertes < maxpos and abs(mid - dernier) >= espace:
            s = sens_mode * (1 if mouv > 0 else -1)
            if (ouvertes + 1) * once * mid / levier < 0.9 * (eq + latent):
                for k in range(maxpos):
                    if not actif[k]:
                        actif[k] = True
                        sns[k] = s
                        ent[k] = ask if s > 0 else bid
                        tdeb[k] = t
                        dernier = mid
                        break
    for k in range(maxpos):
        if actif[k]:
            s = sns[k]
            eq += s * ((BID[n - 1] if s > 0 else ASK[n - 1]) - ent[k]) * once - commission * lots
            nbt += 1
    return eq, nbt, pire, 0


@nb.njit(nogil=True, parallel=True)
def _grille(T, ASK, BID, P, commission, levier, out):
    for r in nb.prange(P.shape[0]):
        e, nbt, pire, saut = _un(T, ASK, BID, int(P[r, 0]), P[r, 1], P[r, 2], P[r, 3], int(P[r, 4]), int(P[r, 5]), P[r, 6], P[r, 7],
                                 commission, levier, int(P[r, 8]))
        out[r, 0], out[r, 1], out[r, 2], out[r, 3] = e, nbt, pire, saut


def grille() -> np.ndarray:
    lignes = []
    for fen in (1_000, 5_000, 15_000, 60_000):
        for seuil in (0.2, 0.5, 1.0, 2.0):
            for tp in (0.1, 0.3, 1.0, 3.0):
                for sl in (0.3, 1.0, 3.0, 10.0):
                    for duree in (10_000, 60_000, 600_000):
                        for maxpos in (1, 5, 20):
                            for sens in (1, -1):
                                lignes.append((fen, seuil, tp, sl, duree, maxpos, 0.05, 0.10, sens))
    return np.array(lignes, dtype=np.float64)


def tester(app: list[str], test: list[str], commission: float = 7.0, levier: float = 500.0) -> str:
    P = grille()
    res = {}
    for nom, mois in (("apprendre", app), ("tester", test)):
        T, A, B = charger(mois)
        out = np.zeros((len(P), 4))
        t = time.time()
        _grille(T, A, B, P, commission, levier, out)
        res[nom] = out
        print(f"{nom} : {len(T):,} ticks, {len(P)} règles en {time.time() - t:.0f} s".replace(",", " "), flush=True)
    a, b = res["apprendre"], res["tester"]
    jours_a, jours_b = len(app) * 22, len(test) * 22
    top = np.argsort(-a[:, 0])[:20]
    L = ["=" * 120, f"  L'IDÉE DU REEL, À LA SECONDE, SUR L'OR — {len(P)} règles, 0,10 lot par position, 1 000 $, levier 1:{levier:.0f}, "
         f"commission {commission:.1f} $ par lot aller-retour + vrai écart achat/vente", "=" * 120, "",
         f"  Apprentissage ({app[0]} à {app[-1]}) : au-dessus de 1 000 $ : {(a[:, 0] > 1000).sum()} règles sur {len(P)} | comptes sautés : {int(a[:, 3].sum())} | "
         f"médiane {np.median(a[:, 0]):,.0f} $ | meilleure {a[:, 0].max():,.0f} $".replace(",", " "),
         f"  Test ({test[0]} à {test[-1]}, jamais vu) : au-dessus de 1 000 $ : {(b[:, 0] > 1000).sum()} règles | comptes sautés : {int(b[:, 3].sum())} | "
         f"médiane {np.median(b[:, 0]):,.0f} $ | meilleure {b[:, 0].max():,.0f} $".replace(",", " "), "",
         "  Les 20 meilleures règles de l'apprentissage, puis sur les mois jamais vus :"]
    for i in top:
        p = P[i]
        L.append(f"    accélération de {p[1]:.1f} $ en {p[0] / 1000:.0f} s, {'dans le sens' if p[8] > 0 else 'contre le sens'}, gain {p[2]:.1f} $, stop {p[3]:.1f} $, "
                 f"{p[5]:.0f} position(s), durée max {p[4] / 1000:.0f} s | apprentissage {a[i, 0]:,.0f} $ ({a[i, 1] / jours_a:.0f} trades/jour) | "
                 f"test {b[i, 0]:,.0f} $ ({b[i, 1] / jours_b:.0f} trades/jour){' | COMPTE SAUTÉ' if b[i, 3] else ''}".replace(",", " "))
    texte = "\n".join(L)
    (DOSSIER / "secondes-report.txt").write_text(texte, encoding="utf-8")
    return texte


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if a and a[0] == "telecharger":
        telecharger(a[1], a[2])
    elif a and a[0] == "tester":
        if len(a) >= 3:                                 # ex. : tester 2024-01:2024-06 2024-07:2025-06
            print(tester(mois_entre(*a[1].split(":")), mois_entre(*a[2].split(":"))))
        else:
            print(tester([f"2024-{m:02d}" for m in range(4, 7)], [f"2024-{m:02d}" for m in range(7, 9)]))
    print("FIN", flush=True)
    _ = (io, struct)
