"""La suite du laboratoire (03/11/2026) : tous les outils autour de la seule piste qui a battu le bot sur des années inconnues
(tendance de fond + double épreuve), deux façons de penser nouvelles, puis l'ouverture unique du coffre 2025-2026.

    python -m agent.idee_suite programme5 <cerveaux> <générations>
    python -m agent.idee_suite portage          -> l'arbitrage du financement (spot acheté + contrat vendu)
    python -m agent.idee_suite coffre           -> ouvre le coffre une seule fois, sur les finalistes fixés à l'avance

Règle fixée AVANT d'ouvrir le coffre : les finalistes sont choisis sur 2023-2024 seulement (médiane des 20 meilleurs),
un par famille : version lente, version d'au moins 10 trades par jour, ta version à plus de 100 trades par jour, cible fixe.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import idee as e
from . import vrai_monde as v

PROGRAMME_5 = (
    # nom, étape, levier max, note, graines (essai précédent)
    ("P1-tendance-flux-croisee", "3-flux", 1.0, "robuste", None),
    ("P2-tendance-argent-croisee", "4-argent", 1.0, "robuste", None),
    ("P3-levier-x2-croisee", "4-argent", 2.0, "robuste", None),
    ("P4-levier-x3-croisee", "4-argent", 3.0, "robuste", None),
    ("P5-tendance-prolongee", "1-tendance", 1.0, "robuste", "E1-tendance-croisee"),
)


def lire_pop(nom: str) -> e.Pop:
    z = np.load(e.DOSSIER / f"{nom}.npz")
    p = e.Pop(z["G"], z["I"], z["Fid"], z["W"])
    return p


def marche_avant_croise(D: dict, cle: str, nom: str, n: int, gens: int, levier: float = 1.0) -> str:
    """Chaque année : évolution sur tout le passé vécu, avec la double épreuve, puis l'année suivante, inconnue."""
    pv = e.pouvoirs(cle, levier)
    bot = e.bot_crypto(D)
    val, val_b, lignes, graines = 1000.0, 1000.0, [], None
    debut = e.bougie(D, "2018-02-01")
    for an in (2021, 2022, 2023, 2024):
        t0, t1 = e.bougie(D, f"{an}-01-01"), e.bougie(D, f"{an + 1}-01-01")
        print(f"  {nom} : apprend jusqu'à fin {an - 1}, trade {an}", flush=True)
        pop, _ = e.evoluer(D, pv, n, gens, debut, t0, 7 + an, graines, croise=True)
        graines = pop.take(np.arange(200))
        top = pop.take(np.arange(20))
        top.I[:, e.I_ACTIFS] = 511
        S, R = e.simuler(D, top, t0, t1)
        b, _ = e.simuler(D, bot, t0, t1)
        m, mb = float(S[:, -1].mean()) / 1000, float(b[0, -1]) / 1000
        val *= m
        val_b *= mb
        lignes.append(f"{an} : x{m:.2f} (bot x{mb:.2f}, ruinés {int((R[:, 5] == 0).sum())}/20)")
    return f"{nom:28} | apprend chaque année avec double épreuve, 2021-2024 : 1 000 -> {e.fmt(val)} (bot {e.fmt(val_b)}) | " + " | ".join(lignes)


def comite(D: dict, f: dict) -> str:
    """Une autre façon de penser : ne pas choisir, mais partager le capital entre le bot et les 20 meilleurs cerveaux lents."""
    top = lire_pop("E1-tendance-croisee").take(np.arange(20))
    top.I[:, e.I_ACTIFS] = 511
    S, _ = e.simuler(D, top, *f["val"])
    b, _ = e.simuler(D, e.bot_crypto(D), *f["val"])
    mc, mb = float(S[:, -1].mean()) / 1000, float(b[0, -1]) / 1000
    return (f"{'Q1-comite-bot-et-cerveaux':28} | 2023-2024 jamais vu : les 20 cerveaux ensemble x{mc:.2f} | bot x{mb:.2f} | moitié-moitié x{(mc + mb) / 2:.2f}"
            f" | pour 1 000 : cerveaux {e.fmt(1000 * mc)}, bot {e.fmt(1000 * mb)}, moitié-moitié {e.fmt(500 * (mc + mb))}")


def programme5(n: int, gens: int, seed: int = 6) -> None:
    D = e.charger()
    f = e.fenetres(D)
    rapport = e.DOSSIER / "idee-report-5.txt"
    lignes = ["=" * 150, f"  TOUS LES OUTILS AUTOUR DE LA TENDANCE DE FOND — {n} cerveaux x {gens} générations, double épreuve", "=" * 150, ""]
    for nom, cle, lev, obj, gr in PROGRAMME_5:
        graines = lire_pop(gr) if gr else None
        if graines is not None:
            graines.I[:, e.I_ACTIFS] = 511
        _, ligne = e.essai(D, cle, nom, n, gens, lev, 0.0, obj, graines, f, seed, e.DOSSIER, croise=True)
        lignes.append(ligne)
        rapport.write_text("\n".join(lignes), encoding="utf-8")
    ligne = comite(D, f)
    print(ligne, flush=True)
    lignes.append(ligne)
    rapport.write_text("\n".join(lignes), encoding="utf-8")
    ligne = marche_avant_croise(D, "1-tendance", "Q2-continu-croise", n * 3 // 4, gens * 4 // 5)
    print(ligne, flush=True)
    lignes.append(ligne)
    rapport.write_text("\n".join(lignes), encoding="utf-8")


# ---------------------------------------------------------------- l'arbitrage du financement
def portage() -> str:
    """Acheter la crypto au comptant et vendre le même montant en contrat : le prix ne compte plus, on touche le financement
    quand il est positif. Frais : 0,10 % spot + 0,055 % contrat, à l'entrée et à la sortie. Chaque crypto a 1/9 du capital,
    dont la moitié en spot et la moitié en garantie du contrat. Hypothèses : pas d'écart entre les deux prix, pas de liquidation."""
    D = e.charger()
    r = D["RATES"].astype(np.float64)
    pos = np.flatnonzero(D["FI"] >= 0)
    jours = D["TS"][pos] // v.D
    A = r.shape[1]
    moy = np.vstack([np.full((8, A), np.nan), np.array([r[i - 8: i + 1].mean(0) for i in range(8, len(r))])])
    cout = 2 * (v.FRAIS_SPOT + v.FRAIS_PERP)

    def jouer(seuil: float, sortie: float, a0: int, a1: int) -> tuple[float, dict]:
        val = np.full(A, 1000.0 / A)
        on = np.zeros(A, bool)
        par_an, an_prec, v_prec = {}, None, 1000.0
        for i in range(a0, a1):
            m = moy[i - 1] if i > 0 else np.full(A, np.nan)        # décision sur le passé seulement
            entre = ~on & (m > seuil)
            sort = on & ~(m > sortie)
            val -= val * 0.5 * cout * (entre | sort)
            on = (on | entre) & ~sort
            val += val * 0.5 * r[i] * on
            an = int(np.datetime64(int(jours[i]), "D").astype(object).year)
            if an_prec is not None and an != an_prec:
                par_an[an_prec] = float(val.sum()) / v_prec
                v_prec = float(val.sum())
            an_prec = an
        par_an[an_prec] = float(val.sum()) / v_prec
        return float(val.sum()), par_an

    i_app0 = int(np.searchsorted(jours, np.datetime64("2019-09-01", "D").astype(int)))
    i_val0 = int(np.searchsorted(jours, np.datetime64("2023-01-01", "D").astype(int)))
    i_val1 = int(np.searchsorted(jours, np.datetime64("2025-01-01", "D").astype(int)))
    choix = []
    for s_ in (0.00005, 0.0001, 0.0002, 0.0003):
        fin, _ = jouer(s_, s_ / 2, i_app0, i_val0)
        choix.append((fin, s_))
    meilleur = max(choix)[1]
    app, an_app = jouer(meilleur, meilleur / 2, i_app0, i_val0)
    val_, an_val = jouer(meilleur, meilleur / 2, i_val0, i_val1)
    texte = (f"{'R1-arbitrage-financement':28} | seuil choisi sur sept. 2019-2022 : {meilleur * 100:.3f} % par 8 h | sept. 2019-2022 : 1 000 -> {e.fmt(app)} "
             f"({' '.join(f'{y}:x{m:.2f}' for y, m in an_app.items())}) | 2023-2024 jamais vu : 1 000 -> {e.fmt(val_)} ({' '.join(f'{y}:x{m:.2f}' for y, m in an_val.items())})")
    json.dump({"seuil": meilleur}, open(e.DOSSIER / "portage.json", "w"))
    print(texte, flush=True)
    return texte


# ---------------------------------------------------------------- le coffre, ouvert une seule fois
def _ligne_mediane(l: str) -> float:
    m = re.search(r"2023-2024 jamais vu : médiane\s+([\d ]+),", l)
    return float(m[1].replace(" ", "")) if m else -1.0


def _ar(l: str) -> float:
    m = re.search(r"\| ([\d.]+) a/r par jour", l)
    return float(m[1]) if m else 0.0


def _pouvoirs_de(nom: str) -> e.Pouvoirs:
    lev = float(re.search(r"x(\d+)", nom)[1]) if re.search(r"(levier-|G-|cible\d+-)x\d+", nom) else 1.0
    if nom.startswith("G-"):
        return e.version_dhia(lev)
    if nom.startswith("H-cible"):
        return e.version_cible(lev, int(re.search(r"cible(\d+)", nom)[1]) / 100)
    return e.Pouvoirs(nom=nom, feats=e.PRIX_5M, libres=(), levier=max(lev, 1.0))


def coffre() -> str:
    marque = e.DOSSIER / "coffre-ouvert.txt"
    if marque.exists():
        return "Le coffre a déjà été ouvert : " + marque.read_text(encoding="utf-8")
    lignes = []
    for r in sorted(e.DOSSIER.glob("idee-report*.txt")):
        lignes += [l for l in r.read_text(encoding="utf-8").splitlines() if re.match(r"^[A-Z]\d*-", l) and "2023-2024 jamais vu : médiane" in l]
    familles = {"lente (moins de 10 trades par jour)": lambda l: _ar(l) < 10 and not l.startswith(("G-", "H-")),
                "au moins 10 trades par jour": lambda l: _ar(l) >= 10 and not l.startswith(("G-", "H-")),
                "ta version, plus de 100 trades par jour": lambda l: l.startswith("G-"),
                "cible fixe de gain": lambda l: l.startswith("H-")}
    D = e.charger()
    f = e.fenetres(D)
    b, _ = e.simuler(D, e.bot_crypto(D), *f["coffre"])
    jours = (f["coffre"][1] - f["coffre"][0]) / v.BPD
    T = ["=" * 150, f"  LE COFFRE — janvier 2025 à octobre 2026 ({jours:.0f} jours), jamais vu par personne, ouvert une seule fois", "=" * 150,
         f"  Bot actuel (BTC, ETH, SOL) : 1 000 -> {e.fmt(b[0, -1])}", ""]
    for fam, test in familles.items():
        cand = [l for l in lignes if test(l)]
        if not cand:
            continue
        l = max(cand, key=_ligne_mediane)
        nom = l.split("|")[0].strip()
        p = lire_pop(nom).take(np.arange(20))
        p.I[:, e.I_ACTIFS] = np.where(p.I[:, e.I_ACTIFS] == 0, 511, p.I[:, e.I_ACTIFS])
        if "croise" in nom:
            p.I[:, e.I_ACTIFS] = 511
        pv = _pouvoirs_de(nom)
        S, R = e.simuler(D, p, *f["coffre"], stop_fixe=pv.stop_fixe, cible_fixe=pv.cible_fixe)
        ar = R[:, 0] / np.maximum(R[:, 4] / v.BPD, 1)
        T.append(f"  {fam:42} : {nom:28} (2023-2024 : médiane {e.fmt(_ligne_mediane(l))}) | coffre : médiane {e.fmt(np.median(S[:, -1]))}, moyenne {e.fmt(S[:, -1].mean())}, "
                 f"meilleur {e.fmt(S[:, -1].max())} | battent le bot {int((S[:, -1] > b[0, -1]).sum())}/20 | ruinés {int((R[:, 5] == 0).sum())}/20 | {np.median(ar):.1f} a/r par jour")
    texte = "\n".join(T)
    (e.DOSSIER / "coffre-report.txt").write_text(texte, encoding="utf-8")
    marque.write_text(time.strftime("%d/%m/%Y %H:%M"), encoding="utf-8")
    print(texte, flush=True)
    return texte


# ---------------------------------------------------------------- un essai par machine (GitHub Actions)
NOMS_GITHUB = ([f"H-cible{c}-x{lev}" for lev in (1, 3) for c in (10, 20, 30)] + [p[0] for p in PROGRAMME_5]
               + ["Q1-comite-bot-et-cerveaux", "Q2-continu-croise", "R1-arbitrage-financement"])


def un(nom: str, n: int, gens: int, sortie: str = "resultats") -> str:
    """Lance un seul essai par son nom et écrit sa ligne de résultat (et ses meilleurs cerveaux) dans `sortie`."""
    out = Path(sortie)
    out.mkdir(parents=True, exist_ok=True)
    if nom == "R1-arbitrage-financement":
        ligne = portage()
    else:
        D = e.charger()
        f = e.fenetres(D)
        if nom.startswith("H-cible"):
            c, lev = nom[len("H-cible"):].split("-x")
            _, ligne = e.essai(D, f"H-cible-{c}", nom, n, gens, float(lev), 0.0, "gain", None, f, 5, out)
        elif nom == "Q1-comite-bot-et-cerveaux":
            ligne = comite(D, f)
        elif nom == "Q2-continu-croise":
            ligne = marche_avant_croise(D, "1-tendance", nom, n * 3 // 4, gens * 4 // 5)
        else:
            nom_, cle, lev, obj, gr = next(p for p in PROGRAMME_5 if p[0] == nom)
            graines = lire_pop(gr) if gr else None
            if graines is not None:
                graines.I[:, e.I_ACTIFS] = 511
            _, ligne = e.essai(D, cle, nom, n, gens, lev, 0.0, obj, graines, f, 6, out, croise=True)
    (out / f"{nom}.txt").write_text(ligne, encoding="utf-8")
    print(ligne, flush=True)
    return ligne


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if a and a[0] == "un":
        un(a[1], int(a[2]), int(a[3]), a[4] if len(a) > 4 else "resultats")
    elif a and a[0] == "programme5":
        programme5(int(a[1]), int(a[2]))
    elif a and a[0] == "portage":
        portage()
    elif a and a[0] == "coffre":
        coffre()
    print("FIN", flush=True)
