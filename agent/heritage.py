"""Reconstruit les meilleurs cerveaux de toutes les expériences à partir de leur description écrite.

Les anciennes expériences ont gardé les champions sous forme de phrases (chiffres arrondis), pas de génomes complets.
Ce module relit ces phrases. ATTENTION : ces cerveaux ont évolué sur des années allant jusqu'à 2026. Les rejouer sur le
passé est donc flatteur : ils connaissent déjà ces marchés. Seuls les mondes « nés de zéro » sont honnêtes.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

from .evolution import FEATURE_NAMES, K
from .vrai_monde import F_D, NOMS, NOMS_1H, TFS, Cerveaux

NOM_CRYPTO = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT", "BNB": "BNBUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT",
              "LINK": "LINKUSDT", "AVAX": "AVAXUSDT"}
_HEURES = re.compile(r"^(momentum|écart à la moyenne|position dans le canal) (\d+) (h|j)$")


def _indice(nom: str, source_bpd: int) -> int | None:
    """Numéro du signal dans le nouveau cerveau (un signal de 4 h prend la durée la plus proche sur les bougies de 1 h)."""
    nom = nom.strip()
    if source_bpd == 1:
        return FEATURE_NAMES.index(nom) if nom in FEATURE_NAMES else None
    if nom + " (1 h)" in NOMS_1H:
        return F_D + NOMS_1H.index(nom + " (1 h)")
    if nom in NOMS_1H:
        return F_D + NOMS_1H.index(nom)
    m = _HEURES.match(nom)
    if not m:
        return None
    voulu = int(m[2]) * (24 if m[3] == "j" else 1)
    mieux, ecart = None, 1e9
    for i, n in enumerate(NOMS_1H):
        c = _HEURES.match(n)
        if c and c[1] == m[1]:
            e = abs(int(c[2]) * (24 if c[3] == "j" else 1) - voulu)
            if e < ecart:
                mieux, ecart = F_D + i, e
    return mieux


def _signaux(texte: str, bpd: int) -> tuple[list[int], list[float]]:
    fs, ws = [], []
    for morceau in texte.split(" ; "):
        m = re.match(r"\s*([+-]\d+\.\d+) x (.+)$", morceau)
        if m:
            i = _indice(m[2], bpd)
            if i is not None:
                fs.append(i)
                ws.append(float(m[1]))
    fs, ws = fs[:K], ws[:K]
    return fs + [0] * (K - len(fs)), ws + [0.0] * (K - len(ws))


def relire(phrase: str, assets: list[str], bpd: int = 1) -> dict | None:
    """Une phrase de `arene.describe` -> les gènes d'un cerveau, ou None si elle n'est pas lisible."""
    regime = phrase.startswith("cerveau hausse")
    t = re.search(r"entrer si \|score\| > (-?\d+\.\d+), sortir si < (-?\d+\.\d+)", phrase)
    r = re.search(r"LEVIER max x(\d+\.\d+), position max (\d+) % du capital, volatilité visée (\d+) %, (\d+) position", phrase)
    st = re.search(r"stop (\d+\.\d+) x ATR", phrase)
    ac = re.search(r"actifs : ([^|]+)", phrase)
    if not (t and r and st and ac):
        return None
    if regime:
        h = re.search(r"cerveau hausse : (.*?) \| cerveau baisse : (.*?) \| entrer", phrase)
        if not h:
            return None
        fb, wb = _signaux(h[1], bpd)
        fr, wr = _signaux(h[2], bpd)
    else:
        h = re.search(r"signaux : (.*?) \| entrer", phrase)
        if not h:
            return None
        fb, wb = _signaux(h[1], bpd)
        fr, wr = fb, wb
    if not any(wb):
        return None
    masque = np.zeros(len(assets), bool)
    for nom in ac[1].split(","):
        nom = nom.strip().removesuffix("-USD")
        nom = NOM_CRYPTO.get(nom, nom)
        if nom in assets:
            masque[assets.index(nom)] = True
    if not masque.any():
        return None
    su = re.search(r"suiveur (\d+\.\d+)", phrase)
    du = re.search(r"durée max (\d+)", phrase)
    re_ = re.search(r"réserve (\d+) %", phrase)
    pyr = re.search(r"renfort \+(\d+) % après (\d+\.\d+) ATR", phrase)
    ad = re.search(r"taille selon ses résultats \(x(\d+\.\d+)\)", phrase)
    rythme = {1: 4, 6: 3, 24: 2}.get(bpd, 4)
    return {"fb": fb, "wb": wb, "fr": fr, "wr": wr, "regime": regime, "th_in": float(t[1]), "th_out": float(t[2]),
            "short": "vend à découvert" in phrase, "stop": float(st[1]), "trail": float(su[1]) if su else 0.0,
            "tvol": float(r[3]) / 100, "maxpos": float(r[2]) / 100, "gross": float(r[1]), "npos": int(r[4]),
            "hold": int(du[1]) if du else 0, "assets": masque, "reserve": float(re_[1]) / 100 if re_ else 0.0,
            "pyr_k": float(pyr[2]) if pyr else 0.0, "pyr_frac": float(pyr[1]) / 100 if pyr else 0.0,
            "ada": float(ad[1]) if ad else 0.0, "tf": rythme, "maker": False}


def en_cerveaux(genes: list[dict]) -> Cerveaux:
    c = {f: np.array([g[f] for g in genes]) for f in Cerveaux.__dataclass_fields__}
    return Cerveaux(**c)


def recolter(assets: list[str], dossier: str = "data") -> tuple[Cerveaux, list[dict]]:
    """Tous les meilleurs cerveaux écrits dans les résultats, chaque cerveau une seule fois."""
    fichiers = sorted(glob.glob(f"{dossier}/ecosysteme-*.json") + glob.glob(f"{dossier}/tout-*.json")
                      + glob.glob(f"{dossier}/github/exp*/tout-*.json"))
    genes, infos, vus, rejets = [], [], set(), 0
    for f in fichiers:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for r in d if isinstance(d, list) else [d]:
            if not isinstance(r, dict):
                continue
            bpd = (r.get("settings") or {}).get("bpd") or 1
            phrases = list(r.get("final_leaders") or [])
            if r.get("best_first_year_brain"):
                phrases.append(r["best_first_year_brain"])
            for rang, p in enumerate(phrases):
                if p in vus:
                    continue
                vus.add(p)
                g = relire(p, assets, bpd)
                if g is None:
                    rejets += 1
                    continue
                genes.append(g)
                infos.append({"source": Path(f).name, "rang": rang, "bpd": bpd, "phrase": p})
    return en_cerveaux(genes), infos


__all__ = ["recolter", "relire", "en_cerveaux", "TFS", "NOMS"]
