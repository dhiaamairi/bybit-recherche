"""L'arène : des millions de cerveaux SANS les règles du projet, en simulation seulement (aucun argent).

Demande de Dhia (01/11/2026) : oublier les règles, l'échec = la mort, les survivants transmettent leur cerveau,
la génération finale doit savoir faire x1000 en un an « dans tous les cas », avec n'importe quelle méthode.

Ce qui est permis aux cerveaux (contrairement aux bots réels) :
  - ACHETER ou VENDRE À DÉCOUVERT (gagner quand le prix baisse), comme sur les contrats à terme
  - LEVIER jusqu'à x25 sur le compte, tout miser sur un seul actif, aucun arrêt d'urgence
  - DEUX CERVEAUX EN UN : un jeu de signaux pour le marché haussier, un autre pour le marché baissier
    (selon la tendance du Bitcoin), s'il le veut
La MORT : ruine (90 % du capital perdu) ou LIQUIDATION (avec le levier, une mauvaise journée efface le compte).
Les COÛTS : contrats à terme Bybit 0,055 % + 0,05 % de glissement par ordre ; financement du levier
0,03 % par jour sur les positions acheteuses (≈ 11 % par an), les vendeuses reçoivent 0,01 % par jour.
LA NOTE : chaque année repart de 1 000 USDT ; note = 0,6 x pire année + 0,4 x moyenne (en log) ; une ruine vaut
log(0,01). L'objectif est donc « réussir CHAQUE année », pas une seule.
L'HONNÊTETÉ : apprentissage, examen sur une année non apprise (choix du champion), test sur l'année suivante.

Les protections de l'argent réel (clés, mode réel) ne sont pas concernées : ceci n'est que du calcul.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evolution import F, FEATURE_NAMES, K, LOOKBACKS, Market, build_market, year_windows

COST = 0.00105              # par ordre : 0,055 % (contrats à terme, preneur) + 0,05 % de glissement
COST_SPOT = 0.0015          # cerveau sans levier ni vente à découvert : spot, 0,1 % + 0,05 %, pas de financement
FUND_LONG = 0.0003          # par jour, sur les positions acheteuses
FUND_SHORT = -0.0001        # par jour : les vendeuses reçoivent un peu
RUIN = 0.10                 # capital restant à partir duquel on est ruiné (mort)
MAINT = 0.005               # marge de maintenance : liquidation si capital <= 0,5 % de l'exposition
LIMITS = {"gross": 25.0}    # levier maximum autorisé dans cette arène (x25 = libre, x3 = modéré)


@dataclass
class Brains:
    fb: np.ndarray          # (P, K) signaux du cerveau « marché haussier »
    wb: np.ndarray
    fr: np.ndarray          # (P, K) signaux du cerveau « marché baissier »
    wr: np.ndarray
    regime: np.ndarray      # utilise le 2e cerveau quand le Bitcoin baisse
    th_in: np.ndarray
    th_out: np.ndarray
    short: np.ndarray       # a le droit de vendre à découvert
    stop: np.ndarray        # multiple d'ATR (8 = pratiquement pas de stop)
    trail: np.ndarray
    tvol: np.ndarray        # volatilité visée par position (peut dépasser 100 % = levier)
    maxpos: np.ndarray      # exposition max par position (jusqu'à 10 x le capital)
    gross: np.ndarray       # exposition totale max (levier du compte, 1 à 25)
    npos: np.ndarray
    hold: np.ndarray
    assets: np.ndarray      # (P, A)
    # gestion de l'argent (utilisée par l'écosystème ; l'arène les ignore)
    reserve: np.ndarray     # part de chaque nouveau gain mise de côté pour toujours (0 à 0,9)
    pyr_k: np.ndarray       # renforcer une position gagnante après pyr_k x ATR de gain (0 = jamais)
    pyr_frac: np.ndarray    # taille du renfort (fraction de la position)
    ada: np.ndarray         # taille selon ses propres résultats récents (0 = taille fixe)

    def __len__(self):
        return len(self.th_in)

    def take(self, i) -> "Brains":
        return Brains(*(getattr(self, f)[i] for f in self.__dataclass_fields__))

    @staticmethod
    def concat(bs: list["Brains"]) -> "Brains":
        return Brains(*(np.concatenate([getattr(b, f) for b in bs]) for f in Brains.__dataclass_fields__))


def random_brains(n: int, A: int, rng: np.random.Generator) -> Brains:
    th = rng.uniform(0.0, 0.8, n)
    lev = np.exp(rng.uniform(0, math.log(max(LIMITS["gross"], 1.0001)), n))
    return clean(Brains(
        fb=rng.integers(0, F, (n, K)), wb=rng.uniform(-1, 1, (n, K)) * (rng.random((n, K)) < 0.8),
        fr=rng.integers(0, F, (n, K)), wr=rng.uniform(-1, 1, (n, K)) * (rng.random((n, K)) < 0.8),
        regime=rng.random(n) < 0.5, th_in=th, th_out=th - rng.uniform(0.05, 1.0, n), short=rng.random(n) < 0.5,
        stop=rng.uniform(0.5, 8, n), trail=np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.5, 8, n)),
        tvol=np.exp(rng.uniform(math.log(0.05), math.log(3.0), n)), maxpos=np.exp(rng.uniform(math.log(0.1), math.log(10), n)),
        gross=lev, npos=rng.integers(1, 10, n), hold=np.where(rng.random(n) < 0.6, 0, rng.integers(2, 120, n)),
        assets=rng.random((n, A)) < 0.6,
        reserve=np.where(rng.random(n) < 0.4, 0.0, rng.uniform(0, 0.8, n)),
        pyr_k=np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.5, 5, n)), pyr_frac=rng.uniform(0, 1, n),
        ada=np.where(rng.random(n) < 0.4, 0.0, rng.uniform(0, 2, n))))


def reference(A: int, assets: list[str]) -> Brains:
    """Le bot actuel dans l'arène (sans levier, sans vente à découvert)."""
    i = [LOOKBACKS.index(k) for k in (14, 28, 56)]
    one = lambda v: np.array([v])   # noqa: E731
    return Brains(fb=np.array([i + [0]]), wb=np.array([[1.0, 1, 1, 0]]), fr=np.array([i + [0]]), wr=np.array([[1.0, 1, 1, 0]]),
                  regime=one(False), th_in=one(0.0), th_out=one(-1e-6), short=one(False), stop=one(3.0), trail=one(0.0),
                  tvol=one(0.15), maxpos=one(0.3), gross=one(1.0), npos=one(3), hold=one(0),
                  assets=np.array([[a in ("BTCUSDT", "ETHUSDT", "SOLUSDT") for a in assets]]),
                  reserve=one(0.0), pyr_k=one(0.0), pyr_frac=one(0.0), ada=one(0.0))


def clean(b: Brains) -> Brains:
    b.th_out = np.minimum(b.th_out, b.th_in - 0.01)
    b.stop = np.clip(b.stop, 0.3, 8)
    b.trail = np.clip(b.trail, 0, 8)
    b.tvol = np.clip(b.tvol, 0.02, 5.0)
    b.maxpos = np.clip(b.maxpos, 0.05, 10.0)
    b.gross = np.clip(b.gross, 1.0, LIMITS["gross"])
    b.npos = np.clip(b.npos, 1, 9)
    b.hold = np.clip(b.hold, 0, 200)
    b.wb = np.clip(b.wb, -1, 1)
    b.wr = np.clip(b.wr, -1, 1)
    b.reserve = np.clip(b.reserve, 0, 0.9)
    b.pyr_k = np.clip(b.pyr_k, 0, 8)
    b.pyr_frac = np.clip(b.pyr_frac, 0, 1)
    b.ada = np.clip(b.ada, 0, 3)
    none = ~b.assets.any(axis=1)
    b.assets[none, 0] = True
    return b


def simulate(m: Market, b: Brains, t0: int, t1: int, start: float = 1000.0) -> dict:
    P, A = len(b), len(m.assets)
    side = np.zeros((P, A), np.int8)        # +1 acheteur, -1 vendeur à découvert, 0 rien
    mag = np.zeros((P, A))                  # exposition en fraction du capital (peut dépasser 1 : levier)
    stop_px = np.zeros((P, A))
    ext_px = np.zeros((P, A))               # meilleur prix depuis l'entrée (stop suiveur)
    age = np.zeros((P, A), int)
    eq = np.full(P, start)
    peak = eq.copy()
    alive = np.ones(P, bool)
    cause = np.zeros(P, np.int8)            # 1 ruine, 2 liquidation
    trades = np.zeros(P, int)
    perp = (b.gross > 1.0001) | b.short                   # levier ou vente à découvert = contrats à terme
    cost = np.where(perp, COST, COST_SPOT)
    for t in range(t0, t1):
        if t > t0:
            o = np.nan_to_num(m.open[t])[None, :]
            lo = np.nan_to_num(m.low[t], nan=np.inf)[None, :]
            hi = np.nan_to_num(m.high[t], nan=-np.inf)[None, :]
            c = np.nan_to_num(m.close[t])[None, :]
            pc = np.nan_to_num(m.close[t - 1], nan=1.0)[None, :]
            lng, sht = side > 0, side < 0
            hit_l = lng & (lo <= stop_px)
            hit_s = sht & (hi >= stop_px)
            hit = hit_l | hit_s
            exit_l = np.minimum(o, stop_px)
            exit_s = np.maximum(o, stop_px)
            r_close = c / pc - 1
            r_pos = np.where(lng, np.where(hit_l, exit_l / pc - 1, r_close), 0.0)
            r_pos = np.where(sht, -np.where(hit_s, exit_s / pc - 1, r_close), r_pos)
            # pire moment de la journée (liquidation) : plus bas pour les acheteurs, plus haut pour les vendeurs
            r_worst = np.where(lng, np.where(hit_l, r_pos, lo / pc - 1), 0.0)
            r_worst = np.where(sht, np.where(hit_s, r_pos, -(hi / pc - 1)), r_worst)
            worst_eq = 1 + (mag * np.nan_to_num(r_worst)).sum(1)
            gross_now = mag.sum(1)
            liquid = alive & (gross_now > 0) & (worst_eq <= MAINT * gross_now)
            fund = ((mag * lng).sum(1) * FUND_LONG + (mag * sht).sum(1) * FUND_SHORT) * perp
            pnl = (mag * np.nan_to_num(r_pos)).sum(1)
            growth = 1 + pnl - (mag * hit).sum(1) * cost - fund
            eq = np.where(alive, eq * np.maximum(growth, 0), eq)
            eq = np.where(liquid, 0.0, eq)
            cause = np.where(liquid, 2, cause)
            g = np.maximum(growth, 1e-9)[:, None]
            keep = (side != 0) & ~hit
            mag = np.where(keep, mag * (1 + np.nan_to_num(r_pos)) / g, 0)
            side = np.where(keep, side, 0).astype(np.int8)
            trades += hit.sum(1)
            ext_px = np.where(side > 0, np.maximum(ext_px, c), np.where(side < 0, np.minimum(ext_px, c), 0))
            age = np.where(side != 0, age + 1, 0)
            peak = np.maximum(peak, eq)
            ruined = alive & ~liquid & (eq <= start * RUIN)
            cause = np.where(ruined, 1, cause)
            alive &= ~(liquid | ruined)
            side = np.where(alive[:, None], side, 0).astype(np.int8)
            mag *= alive[:, None]
        # décision à la clôture t
        x = m.X[t]
        s_bull = (x[:, b.fb] * b.wb[None, :, :]).sum(2).T
        s_bear = (x[:, b.fr] * b.wr[None, :, :]).sum(2).T
        btc_down = not bool(m.btc_up[t])
        score = np.where((b.regime & btc_down)[:, None], s_bear, s_bull)
        tradable = m.ok[t][None, :] & b.assets & alive[:, None]
        old = (b.hold[:, None] > 0) & (age >= b.hold[:, None])
        out = ((side > 0) & ((score < b.th_out[:, None]) | old)) | ((side < 0) & ((score > -b.th_out[:, None]) | old))
        fees = (mag * out).sum(1) * cost
        trades += out.sum(1)
        side = np.where(out, 0, side).astype(np.int8)
        mag = np.where(out, 0, mag)
        atr = m.atr[t][None, :]
        tl = (side > 0) & (b.trail[:, None] > 0)
        ts = (side < 0) & (b.trail[:, None] > 0)
        stop_px = np.where(tl, np.maximum(stop_px, ext_px * (1 - b.trail[:, None] * atr)), stop_px)
        stop_px = np.where(ts, np.minimum(stop_px, ext_px * (1 + b.trail[:, None] * atr)), stop_px)
        want_l = (side == 0) & tradable & (score > b.th_in[:, None])
        want_s = (side == 0) & tradable & b.short[:, None] & (score < -b.th_in[:, None])
        cand = want_l | want_s
        free = b.npos - (side != 0).sum(1)
        strength = np.where(cand, np.abs(score), -np.inf)
        rank = np.argsort(np.argsort(-strength, axis=1), axis=1)
        new = cand & (rank < free[:, None])
        size = np.minimum(b.tvol[:, None] / np.maximum(m.vol[t][None, :], 0.05), b.maxpos[:, None]) * new
        room = np.maximum(b.gross - mag.sum(1), 0)
        size *= np.minimum(1, room / np.maximum(size.sum(1), 1e-12))[:, None]
        new &= size > 0.005
        fees += (size * new).sum(1) * cost
        eq = np.where(alive, eq * np.maximum(1 - fees, 0), eq)
        side = np.where(new, np.where(want_l, 1, -1), side).astype(np.int8)
        mag = np.where(new, size, mag)
        cnow = np.nan_to_num(m.close[t])[None, :]
        stop_px = np.where(new & want_l, cnow * (1 - b.stop[:, None] * atr), stop_px)
        stop_px = np.where(new & ~want_l, cnow * (1 + b.stop[:, None] * atr), stop_px)
        stop_px = np.where(side != 0, stop_px, 0)
        ext_px = np.where(new, cnow, ext_px)
        trades += new.sum(1)
    eq = np.where(alive, eq * np.maximum(1 - mag.sum(1) * cost, 0), eq)
    return {"final": eq, "alive": alive, "cause": cause, "trades": trades}


def fitness(m: Market, b: Brains, windows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logs, finals, dead = [], [], np.zeros(len(b), int)
    for _, a, z in windows:
        r = simulate(m, b, a, z)
        logs.append(np.log(np.maximum(r["final"], 10.0) / 1000))     # ruine = log(0,01)
        finals.append(r["final"])
        dead += ~r["alive"]
    logs = np.array(logs)
    return 0.6 * logs.min(0) + 0.4 * logs.mean(0), np.array(finals), dead


def evolve(m: Market, windows, pop: int, gens: int, seed: int, log=print) -> tuple[Brains, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = len(m.assets)
    b = Brains.concat([reference(A, m.assets), random_brains(pop - 1, A, rng)])
    for g in range(gens):
        fit, finals, dead = fitness(m, b, windows)
        order = np.argsort(-fit)
        if log and (g % 10 == 0 or g == gens - 1):
            best = order[0]
            yrs = ", ".join(f"{int(v):,}".replace(",", " ") for v in finals[:, best])
            log(f"    génération {g:3d} : champion {yrs} | morts {(dead > 0).mean() * 100:.0f} % | "
                f"x1000 atteint (une année) : {int((finals >= 1e6).any(0).sum())} | chaque année : {int((finals >= 1e6).all(0).sum())}")
        if g == gens - 1:
            return b.take(order), fit[order]
        n_el, n_im = int(pop * 0.05), int(pop * 0.10)
        n_ch = pop - n_el - n_im

        def tour(n):
            cand = rng.integers(0, pop, (n, 4))
            return cand[np.arange(n), np.argmax(fit[cand], axis=1)]
        pa, pb = b.take(tour(n_ch)), b.take(tour(n_ch))
        ch = b.take(np.arange(n_ch))
        for f in Brains.__dataclass_fields__:
            va, vb = getattr(pa, f), getattr(pb, f)
            setattr(ch, f, np.where(rng.random(va.shape) < 0.5, va, vb))
        mut = lambda shape, s: rng.normal(0, s, shape) * (rng.random(shape) < 0.25)  # noqa: E731
        for f in ("wb", "wr"):
            setattr(ch, f, getattr(ch, f) + mut(getattr(ch, f).shape, 0.3))
        for f in ("fb", "fr"):
            v = getattr(ch, f)
            setattr(ch, f, np.where(rng.random(v.shape) < 0.08, rng.integers(0, F, v.shape), v))
        ch.th_in = ch.th_in + mut(n_ch, 0.1)
        ch.th_out = ch.th_out + mut(n_ch, 0.1)
        ch.stop = ch.stop + mut(n_ch, 0.5)
        ch.trail = np.where(rng.random(n_ch) < 0.05, 0, ch.trail + mut(n_ch, 0.5))
        for f in ("tvol", "maxpos", "gross"):
            setattr(ch, f, getattr(ch, f) * np.exp(mut(n_ch, 0.25)))
        ch.npos = ch.npos + np.round(mut(n_ch, 0.8)).astype(int)
        ch.hold = np.where(rng.random(n_ch) < 0.05, 0, ch.hold + np.round(mut(n_ch, 10)).astype(int))
        for f in ("regime", "short"):
            setattr(ch, f, np.where(rng.random(n_ch) < 0.05, ~getattr(ch, f), getattr(ch, f)))
        ch.assets = np.where(rng.random(ch.assets.shape) < 0.05, ~ch.assets, ch.assets)
        b = clean(Brains.concat([b.take(order[:n_el]), ch, random_brains(n_im, A, rng)]))


def describe(b: Brains, i: int, assets: list[str], names: list | None = None) -> str:
    names = names or FEATURE_NAMES

    def sigs(f, w):
        return " ; ".join(f"{w[i, k]:+.2f} x {names[f[i, k]]}" for k in range(K) if abs(w[i, k]) > 0.02) or "aucun"
    acts = ", ".join(a.removesuffix("USDT") for a, ok in zip(assets, b.assets[i]) if ok)
    brain = (f"cerveau hausse : {sigs(b.fb, b.wb)} | cerveau baisse : {sigs(b.fr, b.wr)}" if b.regime[i]
             else f"signaux : {sigs(b.fb, b.wb)}")
    return (f"{brain} | entrer si |score| > {b.th_in[i]:.2f}, sortir si < {b.th_out[i]:.2f} | "
            f"{'achète ET vend à découvert' if b.short[i] else 'achète seulement'} | LEVIER max x{b.gross[i]:.1f}, "
            f"position max {b.maxpos[i] * 100:.0f} % du capital, volatilité visée {b.tvol[i] * 100:.0f} %, {b.npos[i]} position(s) | "
            f"stop {b.stop[i]:.1f} x ATR{f', suiveur {b.trail[i]:.1f}' if b.trail[i] > 0 else ''}"
            f"{f', durée max {b.hold[i]} bougies' if b.hold[i] else ''} | actifs : {acts}"
            f" | argent : réserve {b.reserve[i] * 100:.0f} % des gains"
            f"{f', renfort +{b.pyr_frac[i] * 100:.0f} % après {b.pyr_k[i]:.1f} ATR' if b.pyr_k[i] > 0 else ', pas de renfort'}"
            f"{f', taille selon ses résultats (x{b.ada[i]:.1f})' if b.ada[i] > 0 else ''}")


def to_json(b: Brains, i: int) -> dict:
    return {f: getattr(b, f)[i].tolist() for f in Brains.__dataclass_fields__}


def run_fold(bars: dict, train: list[int], exam: int, test: int, pop: int, gens: int, seed: int, out_path: str,
             log=print, max_gross: float = 25.0) -> dict:
    """Un monde : apprentissage -> examen -> test. Écrit son résultat (JSON) pour le rapport final."""
    LIMITS["gross"] = max_gross
    m = build_market(bars)
    A = len(m.assets)
    log(f"MONDE {test} (levier max x{max_gross:g}) : apprentissage {train[0]}-{train[-1]}, examen {exam}, test {test} | "
        f"{pop} cerveaux x {gens} générations")
    b, fit = evolve(m, year_windows(m, train), pop, gens, seed, log)
    top = b.take(np.arange(200))
    ew, tw = year_windows(m, [exam])[0], year_windows(m, [test])[0]
    ex = simulate(m, top, ew[1], ew[2])
    pick = int(np.argmax(np.where(ex["alive"], np.log(np.maximum(ex["final"], 1) / 1000), -99)))
    te = simulate(m, top, tw[1], tw[2])
    ref = reference(A, m.assets)
    rr = simulate(m, ref, tw[1], tw[2])
    _, tr_finals, _ = fitness(m, top.take(np.array([pick])), year_windows(m, train))
    causes = {0: "vivant", 1: "MORT (ruine)", 2: "MORT (liquidation)"}
    res = {"train": train, "exam": exam, "test": test, "pop": pop, "gens": gens, "max_gross": max_gross,
           "train_years": [float(v) for v in tr_finals[:, 0]], "exam_final": float(ex["final"][pick]),
           "test_final": float(te["final"][pick]), "test_status": causes[int(te["cause"][pick])],
           "top200_test_median": float(np.median(te["final"])), "top200_test_dead": int((~te["alive"]).sum()),
           "top200_test_x1000": int((te["final"] >= 1e6).sum()), "reference_test": float(rr["final"][0]),
           "champion": describe(top, pick, m.assets), "genome": to_json(top, pick), "assets": m.assets}
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  TERMINÉ : test {test} champion 1 000 -> {res['test_final']:,.0f} ({res['test_status']}) | bot actuel "
        f"{res['reference_test']:,.0f}".replace(",", " "))
    return res


# ---------------------------------------------------------------- lancement : 3 mondes en parallèle
FOLDS = [([2022], 2023, 2024), ([2022, 2023], 2024, 2025), ([2022, 2023, 2024], 2025, 2026)]
REPORT = "data/arene-report.txt"


def _world(args) -> dict:
    train, exam, test, pop, gens, seed, data_path, out_dir, max_gross, tag = args
    import logging
    logging.disable(logging.CRITICAL)
    from .dataset import Dataset
    logf = Path(out_dir) / f"arene{tag}-x{max_gross:g}-{test}.log"

    def log(text):
        with open(logf, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return run_fold(Dataset.load(data_path).bars, train, exam, test, pop, gens, seed,
                    str(Path(out_dir) / f"arene{tag}-x{max_gross:g}-{test}.json"), log, max_gross)


def main(pop: int = 6000, gens: int = 120, data_path: str = "data/history.json.gz", out_dir: str = "data",
         folds=None, levers=(25.0,), tag: str = "", workers: int = 3) -> dict:
    import multiprocessing as mp
    folds = folds or FOLDS
    jobs = [(tr, ex, te, pop, gens, 100 + i, data_path, out_dir, lev, tag)
            for lev in levers for i, (tr, ex, te) in enumerate(folds)]
    jobs.sort(key=lambda j: -len(j[0]))                   # les mondes les plus longs d'abord
    for j in jobs:
        (Path(out_dir) / f"arene{tag}-x{j[8]:g}-{j[2]}.log").unlink(missing_ok=True)
    with mp.Pool(min(workers, len(jobs))) as p:
        res = p.map(_world, jobs)
    by_lev = {lev: sorted([r for r in res if r["max_gross"] == lev], key=lambda r: r["test"]) for lev in levers}
    text = "\n\n".join(write_report(rs, f"levier max x{lev:g}") for lev, rs in by_lev.items())
    if len(levers) > 1:
        text += "\n\n" + compare(by_lev)
    Path(f"data/arene{tag}-report.txt").write_text(text, encoding="utf-8")
    return by_lev


def compare(by_lev: dict) -> str:
    fmt = lambda v: f"{v:,.0f}".replace(",", " ")   # noqa: E731
    gm = lambda v: math.exp(sum(math.log(max(x, 10) / 1000) for x in v) / len(v)) * 1000   # noqa: E731
    first = next(iter(by_lev.values()))
    L = ["=" * 110, "  COMPARAISON SUR LES ANNÉES JAMAIS VUES (1 000 USDT au départ de chaque année)", "=" * 110,
         f"  {'année':8}" + "".join(f"{'levier x' + format(lev, 'g'):>18}" for lev in by_lev) + f"{'bot actuel':>14}"]
    for i, r in enumerate(first):
        L.append(f"  {r['test']:<8}" + "".join(f"{fmt(rs[i]['test_final']):>18}" for rs in by_lev.values())
                 + f"{fmt(r['reference_test']):>14}")
    L.append(f"  {'moyenne':8}" + "".join(f"{fmt(gm([x['test_final'] for x in rs])):>18}" for rs in by_lev.values())
             + f"{fmt(gm([x['reference_test'] for x in first])):>14}")
    L.append("=" * 110)
    return "\n".join(L)


def write_report(res: list, title: str = "levier max x25") -> str:
    fmt = lambda v: f"{v:,.0f}".replace(",", " ")   # noqa: E731
    total = sum(r["pop"] * r["gens"] for r in res)
    L = ["=" * 110, f"  L'ARÈNE ({title}) — {fmt(total)} cerveaux (vente à découvert, tout miser permis) ; mort = ruine ou liquidation",
         "=" * 110, ""]
    for r in res:
        L.append(f"MONDE {r['test']} : apprentissage {r['train'][0]}-{r['train'][-1]}, examen {r['exam']}, test {r['test']}")
        L.append("  Champion pendant l'apprentissage : " + ", ".join(
            f"{y} : 1 000 -> {fmt(v)}" for y, v in zip(r["train"], r["train_years"])))
        L.append(f"  Examen {r['exam']} : 1 000 -> {fmt(r['exam_final'])}")
        L.append(f"  TEST {r['test']} (jamais vu) : champion 1 000 -> {fmt(r['test_final'])} ({r['test_status']}) | "
                 f"200 meilleurs : médiane {fmt(r['top200_test_median'])}, morts {r['top200_test_dead']}/200, "
                 f"x1000 : {r['top200_test_x1000']} | bot actuel {fmt(r['reference_test'])}")
        L.append(f"  Champion : {r['champion']}")
        L.append("")
    gm = lambda v: math.exp(sum(math.log(max(x, 10) / 1000) for x in v) / len(v)) * 1000   # noqa: E731
    L.append("BILAN SUR LES ANNÉES JAMAIS VUES")
    L.append(f"  Champions : {', '.join(fmt(r['test_final']) for r in res)} -> moyenne {fmt(gm([r['test_final'] for r in res]))} par an")
    L.append(f"  Bot actuel : {', '.join(fmt(r['reference_test']) for r in res)} -> moyenne {fmt(gm([r['reference_test'] for r in res]))} par an")
    L.append(f"  Objectif 1 000 -> 1 000 000 atteint sur une année jamais vue : {sum(r['test_final'] >= 1e6 for r in res)} fois sur {len(res)}")
    L.append("=" * 110)
    return "\n".join(L)


FOLDS_LONG = [(list(range(2018, y - 1)), y - 1, y) for y in (2022, 2023, 2024, 2025, 2026)]


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if a and a[0] == "long":            # 9 ans de Binance, 5 mondes, levier x25 et x3 côte à côte
        main(int(a[1]), int(a[2]), data_path="data/history-binance.json.gz", folds=FOLDS_LONG, levers=(25.0, 3.0),
             tag="-long", workers=int(a[3]) if len(a) > 3 else 8)
    else:
        main(int(a[0]) if a else 6000, int(a[1]) if len(a) > 1 else 120)
