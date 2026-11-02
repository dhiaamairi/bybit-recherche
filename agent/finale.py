"""Lancement de la grande finale.

    python -m agent.finale graines          -> data/finale/graines.npz (les meilleurs cerveaux reconstruits)
    python -m agent.finale rejeu            -> data/finale/rejeu.json + rejeu-report.txt : anciens champions et bot sur le vrai broker
    python -m agent.finale monde <n> [dir]  -> un monde de la grille (une machine GitHub par monde)
    python -m agent.finale rapport [dir]    -> finale-report.txt
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import vrai_monde as v

DOSSIER = Path("data/finale")
GRILLE = (
    [v.Reglage(heritage=True, levier=25.0, objectif="x1000", seed=s) for s in (1, 2, 3, 4)]
    + [v.Reglage(heritage=True, levier=3.0, objectif="x1000", seed=s) for s in (5, 6)]
    + [v.Reglage(heritage=True, levier=3.0, objectif="robuste", seed=s) for s in (7, 8)]
    + [v.Reglage(heritage=True, levier=1.0, objectif="robuste", seed=s) for s in (9, 10)]
    + [v.Reglage(heritage=False, levier=25.0, objectif="x1000", seed=s) for s in (11, 12)]
    + [v.Reglage(heritage=False, levier=3.0, objectif="robuste", seed=s) for s in (13, 14)]
    + [v.Reglage(heritage=False, levier=1.0, objectif="robuste", seed=s) for s in (15, 16)]
)
FMT = lambda x: f"{x:,.0f}".replace(",", " ")   # noqa: E731


def annee(ts: int) -> int:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year


def sauver_graines(M: v.Marche5) -> v.Cerveaux:
    from .heritage import recolter
    DOSSIER.mkdir(parents=True, exist_ok=True)
    b, infos = recolter(M.assets)
    np.savez_compressed(DOSSIER / "graines.npz", **{f: getattr(b, f) for f in v.Cerveaux.__dataclass_fields__})
    (DOSSIER / "graines.json").write_text(json.dumps({"assets": M.assets, "infos": infos}, ensure_ascii=False), encoding="utf-8")
    return b


def lire_graines(M: v.Marche5, chemin: Path = DOSSIER / "graines.npz") -> v.Cerveaux:
    d = np.load(chemin)
    return v.Cerveaux(**{f: d[f] for f in v.Cerveaux.__dataclass_fields__})


def rejeu(M: v.Marche5, b: v.Cerveaux) -> dict:
    """Les cerveaux fixes, sans évolution, sur tout l'historique avec les vrais coûts. Le bot actuel est en tête de liste."""
    bot5 = v.bot_actuel(M.assets)
    bot3 = v.bot_actuel(M.assets)
    bot3.assets[:] = [[a in ("BTCUSDT", "ETHUSDT", "SOLUSDT") for a in M.assets]]
    G = v.Cerveaux.concat([bot5, bot3, b])
    w = v.VraiMonde(M, G)
    fins = {}
    t0 = time.time()
    for k in range(len(M.ts)):
        w.bougie(k)
        if k % v.BPD == 0:
            y = annee(int(M.ts[k]))
            fins.setdefault(y, w.eq + 0)
    y_fin = annee(int(M.ts[-1]))
    annees = sorted(fins)
    seuil = np.array([fins[y] for y in annees] + [w.eq])
    mult = seuil[1:] / np.maximum(seuil[:-1], 1e-9)
    jours = len(M.ts) / v.BPD
    return {"annees": annees, "fin": y_fin, "eq": w.eq.tolist(), "vivant": w.alive.tolist(), "cause": w.cause.tolist(), "mult": mult.tolist(),
            "trades_jour": (w.trades / jours).tolist(), "frais": w.frais.tolist(), "gliss": w.gliss.tolist(), "fin_paye": w.fin.tolist(),
            "tf": G.tf.tolist(), "levier": G.gross.tolist(), "duree_s": time.time() - t0}


def rapport_rejeu(r: dict) -> str:
    eq, mult = np.array(r["eq"]), np.array(r["mult"])
    viv = np.array(r["vivant"])
    an = r["annees"]
    bot = eq[0]
    ch = eq[2:]
    ordre = np.argsort(-ch)[:10] + 2
    L = ["=" * 110, f"  LES {len(ch)} MEILLEURS CERVEAUX SUR LE VRAI BROKER (Bybit + IBKR, sans impôts, 2018-2026, sans évolution)", "=" * 110,
         "Ces cerveaux ont évolué sur ces mêmes années : le résultat est flatté. Le vrai broker ajoute écart, impact, financement réel et minimums.", ""]
    L.append(f"  Bot actuel (BTC, ETH, SOL, NVDA, TSLA) : 1 000 -> {FMT(eq[0])} | années : " + " ".join(f"{y}:x{m:.2f}" for y, m in zip(an, mult[:, 0])))
    L.append(f"  Bot actuel (BTC, ETH, SOL seulement)  : 1 000 -> {FMT(eq[1])} (ancien simulateur : 53 091) | trades/jour {r['trades_jour'][1]:.2f}")
    L.append(f"  Vivants à la fin : {int(viv[2:].sum())} sur {len(ch)} | au-dessus de 1 000 : {int((ch > 1000).sum())} | au-dessus du bot : {int((ch > bot).sum())}")
    tous = mult[:, 2:]
    L.append(f"  Meilleure année d'un cerveau : x{tous.max():.1f} | années à x1000 ou plus : {int((tous >= 1000).sum())}")
    L += ["", "  Les 10 meilleurs :"]
    for i in ordre:
        L.append(f"    1 000 -> {FMT(eq[i]):>12} | {'vivant' if viv[i] else 'mort'} | levier max x{r['levier'][i]:.1f} | {r['trades_jour'][i]:.2f} trades/jour | "
                 f"frais {FMT(r['frais'][i])} + écart {FMT(r['gliss'][i])} + financement {FMT(r['fin_paye'][i])} | meilleure année x{mult[:, i].max():.1f} | pire x{mult[:, i].min():.2f}")
    return "\n".join(L)


def lancer_monde(n: int, sortie: Path) -> None:
    s = GRILLE[n]
    sortie.mkdir(parents=True, exist_ok=True)
    M = v.construire()
    g = lire_graines(M) if s.heritage else None
    print(f"Monde {n} : {s.nom()}", flush=True)
    r = v.vivre(M, s, g, str(sortie / f"finale-{n:02d}.json"), log=lambda t: print(t, flush=True))
    print(f"Terminé : fonds 1 000 -> {FMT(r['courbe'][-1][1])}", flush=True)


def rapport(dossier: Path, rejeu_txt: str | None = None) -> str:
    res = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(dossier.glob("finale-*.json"))]
    L = ["=" * 118, f"  LA GRANDE FINALE — {len(res)} mondes, bougies de 5 minutes, Bybit + IBKR, frais réels, SANS IMPÔTS", "=" * 118,
         "Héritiers : partent des meilleurs cerveaux du passé, qui ont déjà vu ces années : résultat flatté.",
         "Nés de zéro : ne connaissent rien au départ : c'est la mesure honnête.", ""]
    for r in res:
        s, c, a = r["reglage"], r["courbe"], r["annees"]
        fin = c[-1][1]
        L.append(f"  {'héritiers' if s['heritage'] else 'nés de zéro':11} | levier x{s['levier']:g} | {s['objectif']:7} | tirage {s['seed']:2} | fonds 1 000 -> {FMT(fin):>14} | "
                 f"pire année x{min(a.values(), default=0):.2f} | meilleure x{max(a.values(), default=0):.1f} | {r['trades_jour']:.1f} trades/jour | "
                 f"coûts {r['frais_part'] * 100:.3f} % par jour | cerveaux x1000 : {r['x1000']}{' | INTERROMPU' if r['interrompu'] else ''}")
        L.append("      années : " + " ".join(f"{y}:x{v_:.2f}" for y, v_ in a.items()))
    if res:
        record = max(max(r["annees"].values(), default=0) for r in res)
        L += ["", f"  Meilleure année d'un fonds : x{record:.1f} (il manquait x{1000 / max(record, 1e-9):.0f})",
              f"  Cerveaux x1000 sur leur 1re année : {sum(r['x1000'] for r in res)} sur {FMT(sum(r['nes'] for r in res))} nés"]
        best = max(res, key=lambda r: r["meilleur_1an"])
        L += [f"  Record d'un cerveau seul : x{best['meilleur_1an']:,.1f}", f"    {best['meilleur_1an_cerveau']}"]
    if rejeu_txt:
        L += ["", rejeu_txt]
    return "\n".join(L)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if not a or a[0] == "rejeu":
        M = v.construire()
        print(f"marché : {len(M.ts)} bougies", flush=True)
        b = sauver_graines(M)
        print(f"{len(b)} cerveaux", flush=True)
        r = rejeu(M, b)
        (DOSSIER / "rejeu.json").write_text(json.dumps(r), encoding="utf-8")
        t = rapport_rejeu(r)
        (DOSSIER / "rejeu-report.txt").write_text(t, encoding="utf-8")
        print(t)
    elif a[0] == "monde":
        lancer_monde(int(a[1]), Path(a[2]) if len(a) > 2 else Path("resultats"))
    elif a[0] == "rapport":
        d = Path(a[1]) if len(a) > 1 else Path("resultats")
        j = DOSSIER / "rejeu-report.txt"
        t = rapport(d, j.read_text(encoding="utf-8") if j.exists() else None)
        (d / "finale-report.txt").write_text(t, encoding="utf-8")
        print(t)
    print("FIN", flush=True)
    _ = math
