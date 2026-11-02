"""Écosystème vivant : des cerveaux qui tradent, apprennent et se reproduisent EN CONTINU, de 2018 à 2026.

Demande de Dhia (01/11/2026) : « des cerveaux, pas des bots, qui se développent avec le temps et apprennent
toujours ; ceux qui avancent restent en vie, les autres meurent ; trouver une génération qui atteint
1 000 -> 1 000 000 en un an (≈ +2 % par jour) ».

Différence avec l'arène : ici il n'y a PAS d'apprentissage sur le passé puis de test. La population vit le
marché jour après jour, sans jamais voir l'avenir :
  - chaque cerveau a son propre compte (1 000 USDT à sa naissance) et trade avec ses règles (arene.Brains :
    signaux, 2 cerveaux hausse/baisse, levier, vente à découvert, stops...)
  - toutes les `period` jours, on mesure ce que chacun a RÉELLEMENT gagné sur les `memory` derniers jours ;
    les ruinés (ou liquidés) meurent, ainsi que les `kill` % les plus mauvais ; les meilleurs survivants se
    reproduisent (croisement + mutations) ; 10 % de cerveaux totalement nouveaux naissent à chaque fois
  - LE FONDS : à chaque étape, il confie son argent (à parts égales) aux `k` meilleurs cerveaux du moment,
    choisis seulement sur leur passé. Son résultat est la vraie mesure de « cerveaux qui apprennent toujours »,
    car c'est ce qu'un investisseur aurait réellement obtenu, sans connaître l'avenir.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arene import COST, COST_SPOT, FUND_LONG, FUND_SHORT, LIMITS, MAINT, RUIN, Brains, clean, describe, random_brains
from .evolution import F, Market, build_market

D = 86_400_000
BIRTH = 1000.0


@dataclass
class Settings:
    pop: int = 20000
    lever: float = 3.0          # levier maximum autorisé
    period: int = 30            # jours entre deux sélections
    memory: int = 90            # jours de résultats réels regardés pour juger un cerveau
    kill: float = 0.3           # part des plus mauvais qui meurent à chaque sélection (en plus des ruinés)
    fund_k: int = 20            # nombre de cerveaux à qui le fonds confie son argent
    seed: int = 1
    scenarios: int = 0          # > 0 : chaque sélection juge les candidats sur N futurs inventés à partir du passé vécu
    scen_len: int = 90          # durée d'un scénario (jours)
    cand: int = 200             # nombre de candidats passés à l'épreuve des scénarios
    universe: str = "crypto"    # tout | actions | crypto (pour l'expérience « tout le marché »)
    objective: str = "robuste"  # robuste : résultat typique + pire cas | x1000 : part des futurs au rythme x1000 par an
    bpd: int = 1                # bougies par jour : 1 = 1 décision par jour, 6 = toutes les 4 h, 24 = toutes les heures
    mm: bool = True             # gènes de gestion de l'argent actifs (réserve, renfort, taille selon les résultats)
    inter: bool = True          # sens « intermarchés » (largeur, corrélations, or, obligations, dollar...)

    def name(self) -> str:
        if self.universe != "crypto" or self.scenarios:
            obj = " | note x1000" if self.objective == "x1000" else ""
            obj += {1: "", 6: " | toutes les 4 h", 24: " | toutes les heures"}.get(self.bpd, "")
            obj += "" if self.mm else " | sans gestion de l'argent"
            obj += "" if self.inter else " | SANS relations entre marchés"
            return (f"{self.universe:7} | levier x{self.lever:g} | "
                    f"{f'{self.scenarios} scénarios' if self.scenarios else 'sans scénarios'}{obj} | tirage {self.seed}")
        return f"levier x{self.lever:g} | sélection tous les {self.period} j | mémoire {self.memory} j | {self.kill * 100:.0f} % éliminés"


class World:
    """L'état de tous les comptes (une ligne par cerveau), simulé jour après jour."""

    def __init__(self, m: Market, b: Brains):
        self.m, self.b = m, b
        P, A = len(b), len(m.assets)
        self.bpd = getattr(m, "bpd", 1) or 1
        self.reserve_amt = np.zeros(P)          # argent mis de côté (jamais risqué)
        self.hwm = np.full(P, BIRTH)            # plus haut du capital (pour la réserve)
        self.add_px = np.zeros((P, A))          # prix du dernier renfort
        self.ema_r = np.zeros(P)                # rendement récent du cerveau (taille selon ses résultats)
        self.side = np.zeros((P, A), np.int8)
        self.mag = np.zeros((P, A))
        self.stop_px = np.zeros((P, A))
        self.ext_px = np.zeros((P, A))
        self.age = np.zeros((P, A), int)
        self.eq = np.full(P, BIRTH)
        self.alive = np.ones(P, bool)
        self.cause = np.zeros(P, np.int8)
        self.birth = np.zeros(P, int)
        self.trades = np.zeros(P, int)
        self.refresh_costs()

    def refresh_costs(self):
        self.perp = (self.b.gross > 1.0001) | self.b.short
        self.cost = np.where(self.perp, COST, COST_SPOT)

    def reborn(self, idx: np.ndarray, children: Brains, t: int):
        for f in Brains.__dataclass_fields__:
            getattr(self.b, f)[idx] = getattr(children, f)
        for arr in (self.side, self.mag, self.stop_px, self.ext_px, self.age):
            arr[idx] = 0
        self.eq[idx] = BIRTH
        self.reserve_amt[idx] = 0
        self.hwm[idx] = BIRTH
        self.add_px[idx] = 0
        self.ema_r[idx] = 0
        self.alive[idx] = True
        self.cause[idx] = 0
        self.birth[idx] = t
        self.trades[idx] = 0
        self.refresh_costs()

    def step(self, t: int, first: bool):
        m, b = self.m, self.b
        side, mag = self.side, self.mag
        if not first:
            o = np.nan_to_num(m.open[t])[None, :]
            lo = np.nan_to_num(m.low[t], nan=np.inf)[None, :]
            hi = np.nan_to_num(m.high[t], nan=-np.inf)[None, :]
            c = np.nan_to_num(m.close[t])[None, :]
            pc = np.nan_to_num(m.close[t - 1], nan=1.0)[None, :]
            lng, sht = side > 0, side < 0
            hit_l = lng & (lo <= self.stop_px)
            hit_s = sht & (hi >= self.stop_px)
            hit = hit_l | hit_s
            r_close = c / pc - 1
            r_pos = np.where(lng, np.where(hit_l, np.minimum(o, self.stop_px) / pc - 1, r_close), 0.0)
            r_pos = np.where(sht, -np.where(hit_s, np.maximum(o, self.stop_px) / pc - 1, r_close), r_pos)
            r_pos = np.nan_to_num(r_pos)
            r_worst = np.where(lng, np.where(hit_l, r_pos, lo / pc - 1), 0.0)
            r_worst = np.where(sht, np.where(hit_s, r_pos, -(hi / pc - 1)), r_worst)
            gross_now = mag.sum(1)
            liquid = self.alive & (gross_now > 0) & (1 + (mag * np.nan_to_num(r_worst)).sum(1) <= MAINT * gross_now)
            fund = ((mag * lng).sum(1) * FUND_LONG + (mag * sht).sum(1) * FUND_SHORT) * self.perp / self.bpd
            growth = 1 + (mag * r_pos).sum(1) - (mag * hit).sum(1) * self.cost - fund
            self.eq = np.where(self.alive, self.eq * np.maximum(growth, 0), self.eq)
            self.eq = np.where(liquid, 0.0, self.eq)
            self.ema_r = 0.97 * self.ema_r + 0.03 * np.log(np.maximum(growth, 1e-9))
            # réserve : une part de chaque nouveau gain est mise de côté
            gain = np.maximum(self.eq - self.hwm, 0)
            self.reserve_amt = np.minimum(self.reserve_amt + b.reserve * gain, self.eq)
            self.hwm = np.maximum(self.hwm, self.eq)
            self.cause = np.where(liquid, 2, self.cause)
            keep = (side != 0) & ~hit
            mag[:] = np.where(keep, mag * (1 + r_pos) / np.maximum(growth, 1e-9)[:, None], 0)
            side[:] = np.where(keep, side, 0)
            self.trades += hit.sum(1)
            self.ext_px = np.where(side > 0, np.maximum(self.ext_px, c), np.where(side < 0, np.minimum(self.ext_px, c), 0))
            self.age = np.where(side != 0, self.age + 1, 0)
            ruined = self.alive & ~liquid & (self.eq <= BIRTH * RUIN)
            self.cause = np.where(ruined, 1, self.cause)
            self.alive &= ~(liquid | ruined)
            side[:] = np.where(self.alive[:, None], side, 0)
            mag *= self.alive[:, None]
        x = m.X[t]
        s_bull = (x[:, b.fb] * b.wb[None, :, :]).sum(2).T
        s_bear = (x[:, b.fr] * b.wr[None, :, :]).sum(2).T
        btc_down = not bool(m.btc_up[t])
        score = np.where((b.regime & btc_down)[:, None], s_bear, s_bull)
        tradable = m.ok[t][None, :] & b.assets & self.alive[:, None]
        old = (b.hold[:, None] > 0) & (self.age >= b.hold[:, None])
        out = ((side > 0) & ((score < b.th_out[:, None]) | old)) | ((side < 0) & ((score > -b.th_out[:, None]) | old))
        fees = (mag * out).sum(1) * self.cost
        self.trades += out.sum(1)
        side[:] = np.where(out, 0, side)
        mag[:] = np.where(out, 0, mag)
        atr = m.atr[t][None, :]
        tl = (side > 0) & (b.trail[:, None] > 0)
        ts = (side < 0) & (b.trail[:, None] > 0)
        self.stop_px = np.where(tl, np.maximum(self.stop_px, self.ext_px * (1 - b.trail[:, None] * atr)), self.stop_px)
        self.stop_px = np.where(ts, np.minimum(self.stop_px, self.ext_px * (1 + b.trail[:, None] * atr)), self.stop_px)
        want_l = (side == 0) & tradable & (score > b.th_in[:, None])
        want_s = (side == 0) & tradable & b.short[:, None] & (score < -b.th_in[:, None])
        # renfort des positions gagnantes (pyramide)
        rs = self.reserve_amt / np.maximum(self.eq, 1e-9)
        room_all = np.maximum(b.gross * (1 - rs) - mag.sum(1), 0)
        cpx = np.nan_to_num(m.close[t])[None, :]
        if (b.pyr_k > 0).any():
            held = side != 0
            prog = np.where(held, side * (cpx / np.maximum(self.add_px, 1e-12) - 1), 0)
            go = held & (b.pyr_k[:, None] > 0) & (prog >= b.pyr_k[:, None] * atr)
            extra = np.where(go, np.minimum(b.pyr_frac[:, None] * mag, np.maximum(b.maxpos[:, None] - mag, 0)), 0)
            extra *= np.minimum(1, room_all / np.maximum(extra.sum(1), 1e-12))[:, None]
            fees += extra.sum(1) * self.cost
            mag[:] = mag + extra
            self.add_px = np.where(extra > 0, cpx, self.add_px)
            room_all = np.maximum(room_all - extra.sum(1), 0)
        cand = want_l | want_s
        free = b.npos - (side != 0).sum(1)
        strength = np.where(cand, np.abs(score), -np.inf)
        rank = np.argsort(np.argsort(-strength, axis=1), axis=1)
        new = cand & (rank < free[:, None])
        mult = np.clip(np.exp(b.ada * self.ema_r * 200 * self.bpd), 0.25, 3.0)     # taille selon ses résultats
        size = np.minimum(b.tvol[:, None] / np.maximum(m.vol[t][None, :], 0.05) * mult[:, None], b.maxpos[:, None]) * new
        room = room_all
        size *= np.minimum(1, room / np.maximum(size.sum(1), 1e-12))[:, None]
        new &= size > 0.005
        fees += (size * new).sum(1) * self.cost
        self.eq = np.where(self.alive, self.eq * np.maximum(1 - fees, 0), self.eq)
        side[:] = np.where(new, np.where(want_l, 1, -1), side)
        mag[:] = np.where(new, size, mag)
        cnow = np.nan_to_num(m.close[t])[None, :]
        self.stop_px = np.where(new & want_l, cnow * (1 - b.stop[:, None] * atr), self.stop_px)
        self.stop_px = np.where(new & ~want_l, cnow * (1 + b.stop[:, None] * atr), self.stop_px)
        self.stop_px = np.where(side != 0, self.stop_px, 0)
        self.ext_px = np.where(new, cnow, self.ext_px)
        self.add_px = np.where(new, cnow, np.where(side != 0, self.add_px, 0))
        self.trades += new.sum(1)


def make_scenario(m: Market, first: int, last: int, length: int, rng: np.random.Generator, block: int | None = None) -> Market:
    """Un futur possible : des blocs de `block` jours consécutifs tirés au hasard entre `first` et `last` (passé vécu
    seulement), recollés. Tous les actifs bougent ensemble dans un bloc (corrélations réelles). Les prix sont refaits
    en enchaînant les variations ; signaux, volatilité et ATR sont ceux des jours tirés."""
    block = block or 10 * (getattr(m, "bpd", 1) or 1)   # blocs de 10 jours
    n_blocks = -(-length // block)
    starts = rng.integers(first + 1, max(first + 2, last - block), n_blocks)
    idx = np.concatenate([np.arange(a, a + block) for a in starts])[:length]
    C, O, L, H = m.close, m.open, m.low, m.high
    prev = np.nan_to_num(C[idx - 1], nan=1.0)
    rel = lambda arr: np.nan_to_num(arr[idx] / prev, nan=1.0)   # noqa: E731
    path = np.cumprod(np.vstack([np.ones((1, C.shape[1])), rel(C)]), axis=0)   # prix « veille » de chaque jour
    base = path[:-1]
    return Market(days=np.arange(length), open=base * rel(O), low=base * rel(L), close=path[1:], ok=m.ok[idx],
                  X=m.X[idx], vol=m.vol[idx], atr=m.atr[idx], btc_up=m.btc_up[idx], assets=m.assets, high=base * rel(H),
                  names=m.names, bpd=m.bpd)


def run_world(m: Market, b: Brains, t0: int, t1: int) -> np.ndarray:
    """Capital final de chaque cerveau, simulé avec tous ses gènes (gestion de l'argent comprise)."""
    w = World(m, b)
    for t in range(t0, t1):
        w.step(t, t == t0)
    return np.where(w.alive, w.eq, np.minimum(w.eq, BIRTH * RUIN))


def scenario_fitness(m: Market, cand: Brains, first: int, last: int, s: "Settings", rng: np.random.Generator) -> np.ndarray:
    """Note d'un candidat sur `s.scenarios` futurs inventés : moitié résultat typique, moitié pire cas (5 %)."""
    bpd = getattr(m, "bpd", 1) or 1
    L = s.scen_len * bpd
    logs = []
    for _ in range(s.scenarios):
        ms = make_scenario(m, first, last, L + 1, rng)
        logs.append(np.log(np.maximum(run_world(ms, cand, 0, L + 1), 10.0) / 1000))
    logs = np.array(logs)
    if s.objective == "x1000":
        # fixé sur l'objectif : part des futurs où le cerveau suit le rythme x1000 par an (x5,5 en 90 jours)
        target = math.log(1000) * s.scen_len / 365
        return (logs >= target).mean(axis=0) + 0.1 * np.median(logs, axis=0)
    return 0.5 * np.median(logs, axis=0) + 0.5 * np.quantile(logs, 0.05, axis=0)


def breed(b: Brains, parents: np.ndarray, fit: np.ndarray, n: int, rng: np.random.Generator) -> Brains:
    """n enfants : deux parents tirés parmi les survivants (tournoi de 4), croisement gène par gène, mutations."""
    def tour():
        cand = parents[rng.integers(0, len(parents), (n, 4))]
        return cand[np.arange(n), np.argmax(fit[cand], axis=1)]
    pa, pb = b.take(tour()), b.take(tour())
    ch = b.take(tour())
    for f in Brains.__dataclass_fields__:
        va, vb = getattr(pa, f), getattr(pb, f)
        setattr(ch, f, np.where(rng.random(va.shape) < 0.5, va, vb))
    mut = lambda shape, s: rng.normal(0, s, shape) * (rng.random(shape) < 0.25)  # noqa: E731
    for f in ("wb", "wr"):
        setattr(ch, f, getattr(ch, f) + mut(getattr(ch, f).shape, 0.3))
    for f in ("fb", "fr"):
        v = getattr(ch, f)
        setattr(ch, f, np.where(rng.random(v.shape) < 0.08, rng.integers(0, F, v.shape), v))
    ch.th_in = ch.th_in + mut(n, 0.1)
    ch.th_out = ch.th_out + mut(n, 0.1)
    ch.stop = ch.stop + mut(n, 0.5)
    ch.trail = np.where(rng.random(n) < 0.05, 0, ch.trail + mut(n, 0.5))
    for f in ("tvol", "maxpos", "gross"):
        setattr(ch, f, getattr(ch, f) * np.exp(mut(n, 0.25)))
    ch.npos = ch.npos + np.round(mut(n, 0.8)).astype(int)
    ch.hold = np.where(rng.random(n) < 0.05, 0, ch.hold + np.round(mut(n, 10)).astype(int))
    for f in ("regime", "short"):
        setattr(ch, f, np.where(rng.random(n) < 0.05, ~getattr(ch, f), getattr(ch, f)))
    ch.assets = np.where(rng.random(ch.assets.shape) < 0.05, ~ch.assets, ch.assets)
    ch.reserve = ch.reserve + mut(n, 0.1)
    ch.pyr_k = np.where(rng.random(n) < 0.05, 0, ch.pyr_k + mut(n, 0.5))
    ch.pyr_frac = ch.pyr_frac + mut(n, 0.15)
    ch.ada = np.where(rng.random(n) < 0.05, 0, ch.ada + mut(n, 0.3))
    return clean(ch)


def strip_mm(b: Brains) -> Brains:
    """Monde témoin : pas de gestion de l'argent."""
    b.reserve[:] = 0
    b.pyr_k[:] = 0
    b.ada[:] = 0
    return b


def live(m: Market, s: Settings, t0: int, t1: int, log=None) -> dict:
    """L'écosystème vit de t0 à t1. Renvoie la courbe du fonds, les records, les champions."""
    LIMITS["gross"] = s.lever
    rng = np.random.default_rng(s.seed)
    A = len(m.assets)
    bpd = getattr(m, "bpd", 1) or 1                        # bougies par jour : les durées « en jours » deviennent des bougies
    period, memory, year = s.period * bpd, s.memory * bpd, 365 * bpd
    first = random_brains(s.pop, A, rng)
    w = World(m, first if s.mm else strip_mm(first))
    names = getattr(m, "names", None)
    decisions = 0
    snaps: list[np.ndarray] = []           # capital de chaque cerveau à chaque sélection (pour juger sur `memory` j)
    births_snap: list[np.ndarray] = []
    fund = 1000.0
    fund_curve = [(int(m.days[t0]), fund)]
    selected = np.array([], int)
    sel_eq = np.array([])
    look = max(1, memory // period)
    best_year = (0.0, None)                # meilleur gain réalisé par un cerveau sur ses 365 premiers jours
    born_at_year = {}                      # cerveaux suivis pour mesurer leurs 365 premiers jours
    n_born = s.pop
    reached = 0
    gen_stats: list[dict] = []             # santé de la population à chaque sélection (les générations s'améliorent-elles ?)
    brain_days = 0
    for t in range(t0, t1):
        w.step(t, t == t0)
        decisions += s.pop
        if (t - t0) % bpd == 0:
            brain_days += s.pop
        # suivi des 365 premiers jours de chaque cerveau (résultat réel, jamais vu à l'avance)
        if (t - t0) % (7 * bpd) == 0:
            age_days = t - w.birth
            done = (age_days >= year) & (age_days < year + 7 * bpd)
            if done.any():
                mult = np.where(w.alive[done], w.eq[done] / BIRTH, 0)
                reached += int((mult >= 1000).sum())
                j = int(np.argmax(mult))
                if mult[j] > best_year[0]:
                    best_year = (float(mult[j]), describe(w.b, int(np.where(done)[0][j]), m.assets, names))
        if (t - t0) % period != 0 or t == t0:
            continue
        # 1) le fonds encaisse le résultat de la période écoulée
        if len(selected):
            r = np.where(w.birth[selected] < t - period + 1, w.eq[selected] / np.maximum(sel_eq, 1e-9), 1.0)
            fund *= max(float(np.mean(r)) - 0.002, 0)    # 0,2 % de frais pour changer de cerveaux
            fund_curve.append((int(m.days[t]), fund))
        # 2) jugement : gain réel sur les `memory` derniers jours
        snaps.append(w.eq.copy())
        births_snap.append(w.birth.copy())
        del snaps[:-(look + 1)], births_snap[:-(look + 1)]            # on ne garde que la mémoire utile
        ref_i = max(0, len(snaps) - 1 - look)
        ref = np.where(births_snap[ref_i] == w.birth, snaps[ref_i], BIRTH)
        perf = np.log(np.maximum(w.eq, 1.0) / np.maximum(ref, 1.0))
        old_enough = (t - w.birth) >= min(memory, period)
        fit = np.where(w.alive, perf, -99.0)
        if w.alive.any():
            gen_stats.append({"day": int(m.days[t]), "ruined": float(1 - w.alive.mean()),
                              "median": float(np.median(perf[w.alive])), "lev": float(np.median(w.b.gross[w.alive])),
                              "short": float(w.b.short[w.alive].mean())})
        scen = None
        if s.scenarios:
            # mémoire : les meilleurs candidats affrontent des futurs inventés à partir de TOUT le passé vécu
            pool = np.where(w.alive & old_enough & (w.birth < t))[0]
            if len(pool):
                cand = pool[np.argsort(-fit[pool])[: s.cand]]
                sf = scenario_fitness(m, w.b.take(cand), t0, t, s, rng)
                scen = (cand, sf)
        # 3) mort des ruinés et des plus mauvais, naissance des enfants des meilleurs
        dead = ~w.alive
        if scen is not None:                              # les candidats validés par les scénarios deviennent les parents préférés
            fit = fit.copy()
            fit[scen[0]] = fit.max() + 10 + scen[1]
        ranked = np.argsort(fit)
        n_kill = int(s.kill * s.pop)
        dead[ranked[:n_kill]] = True
        survivors = np.where(~dead)[0]
        idx = np.where(dead)[0]
        if len(idx) and len(survivors):
            n_imm = int(len(idx) * 0.15)
            kids = Brains.concat([breed(w.b, survivors, fit, len(idx) - n_imm, rng), random_brains(n_imm, A, rng)])
            w.reborn(idx, kids if s.mm else strip_mm(kids), t)
            for sn in snaps:
                sn[idx] = BIRTH
            n_born += len(idx)
        # 4) le fonds choisit les meilleurs cerveaux du moment (sur leur seul passé)
        elig = np.where(w.alive & old_enough & (w.birth < t))[0]
        if scen is not None:
            elig = scen[0][w.alive[scen[0]] & (w.birth[scen[0]] < t)]
        if len(elig):
            selected = elig[np.argsort(-fit[elig])[: s.fund_k]]
            sel_eq = w.eq[selected].copy()
        else:
            selected = np.array([], int)
        if log and len(snaps) % 12 == 0:
            log(f"    {int(m.days[t])} : fonds {fund:,.0f} | vivants {int(w.alive.sum())} | nés {n_born:,}".replace(",", " "))
    top = np.argsort(-np.where(w.alive, w.eq, 0))[:3]
    return {"settings": s.__dict__, "fund_curve": fund_curve, "born": n_born, "brain_days": brain_days,
            "decisions": decisions, "best_first_year": best_year[0], "best_first_year_brain": best_year[1],
            "reached_x1000": reached, "final_leaders": [describe(w.b, int(i), m.assets, names) for i in top],
            "gen_stats": gen_stats}


def yearly(curve: list[tuple[int, float]]) -> dict[int, float]:
    """Multiple réalisé par le fonds pour chaque année civile."""
    from datetime import datetime, timezone
    out, start = {}, {}
    for d, v in curve:
        y = datetime.fromtimestamp(d * 86400, tz=timezone.utc).year
        start.setdefault(y, None)
        out[y] = v
    years = sorted(out)
    res, prev = {}, curve[0][1]
    for y in years:
        res[y] = out[y] / prev if prev > 0 else 0.0       # un fonds ruiné reste à 0
        prev = out[y]
    return res


# ---------------------------------------------------------------- lancement : des dizaines d'écosystèmes
GRID = [Settings(lever=lv, period=p, memory=mem, kill=k)
        for lv in (1.0, 3.0, 25.0) for p in (7, 30) for mem in (30, 90, 180) for k in (0.2, 0.5)]


def _run(args):
    i, s, data_path, out_dir = args
    import logging
    logging.disable(logging.CRITICAL)
    from .dataset import Dataset
    from .evolution import year_windows
    m = build_market(Dataset.load(data_path).bars)
    t0 = year_windows(m, [2018])[0][1]
    res = live(m, s, t0, len(m.days))
    res["yearly"] = {int(y): v for y, v in yearly(res["fund_curve"]).items()}
    Path(out_dir, f"ecosysteme-{i:02d}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def main(pop: int = 20000, workers: int = 6, data_path: str = "data/history-binance.json.gz", out_dir: str = "data") -> list:
    import multiprocessing as mp
    jobs = []
    for i, s in enumerate(GRID):
        s.pop = pop
        jobs.append((i, s, data_path, out_dir))
    with mp.Pool(workers) as p:
        res = p.map(_run, jobs)
    report(res)
    return res


def report(res: list, path: str = "data/ecosysteme-report.txt") -> str:
    fmt = lambda v: f"{v:,.0f}".replace(",", " ")   # noqa: E731
    total_days = sum(r["brain_days"] for r in res)
    total_born = sum(r["born"] for r in res)
    L = ["=" * 120, f"  ÉCOSYSTÈMES VIVANTS 2018-2026 — {len(res)} mondes, {fmt(total_born)} cerveaux nés, "
         f"{fmt(total_days)} journées de trading simulées (une décision par cerveau et par jour)", "=" * 120,
         "Le FONDS confie son argent aux 20 meilleurs cerveaux du moment, jugés uniquement sur leur passé : c'est ce qu'on",
         "aurait réellement obtenu. 1 000 USDT au 1er janvier 2018.", ""]
    years = sorted({y for r in res for y in r["yearly"]})
    L.append(f"  {'réglages':62} " + "".join(f"{y:>8}" for y in years) + f"{'2018->2026':>14}")
    for r in sorted(res, key=lambda r: -r["fund_curve"][-1][1]):
        s = Settings(**r["settings"])
        L.append(f"  {s.name():62} " + "".join(f"{'x' + format(r['yearly'].get(y, 1), '.2f'):>8}" for y in years)
                 + f"{fmt(r['fund_curve'][-1][1]):>14}")
    best = max(res, key=lambda r: r["best_first_year"])
    L += ["", f"  Meilleur cerveau sur ses 365 premiers jours de vie (résultat réel, sans voir l'avenir) : "
              f"x{best['best_first_year']:,.1f}".replace(",", " "),
          f"    {best['best_first_year_brain']}",
          f"  Cerveaux ayant fait x1000 sur leurs 365 premiers jours : {sum(r['reached_x1000'] for r in res)} "
          f"sur {fmt(total_born)} nés (on ne pouvait pas savoir à l'avance lesquels)",
          f"  Meilleur fonds : 1 000 -> {fmt(max(r['fund_curve'][-1][1] for r in res))} en près de 9 ans", "=" * 120]
    text = "\n".join(L)
    Path(path).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- tout le marché (actions, matières, devises, crypto)
GRID_TOUT = [Settings(universe="tout", lever=1.0), Settings(universe="tout", lever=1.0, scenarios=24),
             Settings(universe="tout", lever=3.0), Settings(universe="tout", lever=3.0, scenarios=24),
             Settings(universe="tout", lever=10.0, scenarios=24), Settings(universe="tout", lever=25.0, scenarios=24),
             Settings(universe="actions", lever=3.0, scenarios=24), Settings(universe="crypto", lever=3.0, scenarios=24)]


def universe_assets(tag: str) -> list[str]:
    from .univers import UNIVERSE
    if tag == "actions":
        return UNIVERSE["actions"] + UNIVERSE["indices"]
    if tag == "crypto":
        return UNIVERSE["crypto"]
    return [a for g in UNIVERSE.values() for a in g]


def references(m: Market, t0: int) -> dict:
    """Le bot actuel appliqué à cet univers, et « acheter et garder » le S&P 500 (ou le Bitcoin)."""
    from .arene import LIMITS as AL, reference, simulate as sim_fresh
    from .evolution import year_windows
    AL["gross"] = 25.0
    ref = reference(len(m.assets), m.assets)
    ref.assets[:] = True
    hold = reference(len(m.assets), m.assets)
    target = "SPY" if "SPY" in m.assets else m.assets[0]
    hold.assets[:] = [[a == target for a in m.assets]]
    hold.wb[:] = 0
    hold.th_in[:] = -9
    hold.th_out[:] = -10
    hold.stop[:] = 8
    hold.tvol[:] = 5
    hold.maxpos[:] = 1
    hold.npos[:] = 1
    first_year = int(1970 + m.days[t0] // 365.25)
    years = list(range(first_year, 2027))
    out = {"bot actuel": {}, f"garder {target}": {}}
    for y, a, z in year_windows(m, years):
        a = max(a, t0)
        out["bot actuel"][y] = float(sim_fresh(m, ref, a, z)["final"][0] / 1000)
        out[f"garder {target}"][y] = float(sim_fresh(m, hold, a, z)["final"][0] / 1000)
    return out


def _run_tout(args):
    i, s, data_path, out_dir = args
    import logging
    logging.disable(logging.CRITICAL)
    from .dataset import Dataset
    from .evolution import year_windows
    bars = Dataset.load(data_path).bars
    assets = universe_assets(s.universe)
    regime = "SPY" if "SPY" in assets else "BTC-USD"
    m = build_market(bars, assets, regime=regime, inter=s.inter)
    start = 2015 if s.universe == "crypto" else 2001
    t0 = year_windows(m, [start])[0][1]
    res = live(m, s, t0, len(m.days))
    res["yearly"] = {int(y): v for y, v in yearly(res["fund_curve"]).items()}
    res["references"] = references(m, t0)
    Path(out_dir, f"tout-{i:02d}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def main_tout(pop: int = 10000, workers: int = 8, data_path: str = "data/history-marche.json.gz", out_dir: str = "data"):
    import multiprocessing as mp
    jobs = []
    for i, s in enumerate(GRID_TOUT):
        s.pop = pop
        jobs.append((i, s, data_path, out_dir))
    with mp.Pool(min(workers, len(jobs))) as p:
        res = p.map(_run_tout, jobs)
    report_tout(res)
    return res


def report_tout(res: list, path: str = "data/tout-marche-report.txt") -> str:
    fmt = lambda v: f"{v:,.0f}".replace(",", " ")   # noqa: E731
    L = ["=" * 124, f"  TOUT LE MARCHÉ — actions, indices, obligations, or, argent, pétrole, devises, cryptos — "
         f"{len(res)} écosystèmes, {fmt(sum(r['born'] for r in res))} cerveaux nés, "
         f"{fmt(sum(r['brain_days'] for r in res))} journées simulées", "=" * 124,
         "Le FONDS suit les 20 meilleurs cerveaux du moment (avec scénarios : les 20 meilleurs sur 24 futurs inventés).",
         "1 000 USDT au départ (au début de chaque monde). Résultat = ce qu'on aurait réellement obtenu.", ""]
    for r in res:
        s = Settings(**r["settings"])
        ys = r["yearly"]
        refs = r["references"]
        cum = lambda d: np.prod([v for y, v in d.items() if y in ys]) * 1000   # noqa: E731
        worst = min(ys.values())
        L.append(f"  {s.name():40} fonds 1 000 -> {fmt(r['fund_curve'][-1][1]):>14} | pire année x{worst:.2f} | "
                 + " | ".join(f"{k} -> {fmt(cum(v))}" for k, v in refs.items()))
        L.append("      années : " + " ".join(f"{y}:x{v:.2f}" for y, v in ys.items()))
        # objectif x1000 par an : où en est-on ?
        days = max(1, r["fund_curve"][-1][0] - r["fund_curve"][0][0])
        daily = (max(r["fund_curve"][-1][1], 1e-9) / 1000) ** (1 / days) - 1
        best_y = max(ys.values())
        L.append(f"      OBJECTIF x1000 : rythme nécessaire +1,91 % par jour | rythme obtenu {daily * 100:+.3f} % par jour | "
                 f"meilleure année x{best_y:.1f} (il manquait x{1000 / max(best_y, 1e-9):,.0f})".replace(",", " "))
    best = max(res, key=lambda r: r["best_first_year"])
    L += ["", f"  Meilleur cerveau sur ses 365 premiers jours (réel) : x{best['best_first_year']:,.1f}".replace(",", " "),
          f"    {best['best_first_year_brain']}",
          f"  Cerveaux x1000 sur leur 1re année : {sum(r['reached_x1000'] for r in res)} sur {fmt(sum(r['born'] for r in res))} nés",
          "  Chefs de file à la fin (meilleur écosystème) :"]
    top = max(res, key=lambda r: r["fund_curve"][-1][1])
    L += [f"    - {d}" for d in top["final_leaders"]]
    L.append("=" * 124)
    text = "\n".join(L)
    Path(path).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- LE MONDE : le plus grand test, tout le marché ensemble
GRID_MONDE = [Settings(universe="tout", lever=1.0, scenarios=24, seed=11),
              Settings(universe="tout", lever=3.0, scenarios=24, seed=12),
              Settings(universe="tout", lever=10.0, scenarios=24, seed=13),
              Settings(universe="tout", lever=3.0, scenarios=0, seed=14)]


GRID_MONDE3 = [Settings(universe="tout", lever=1.0, scenarios=24, seed=s) for s in (21, 22, 23, 24)] + [
    Settings(universe="tout", lever=1.0, scenarios=24, seed=25, objective="x1000"),
    Settings(universe="tout", lever=1.0, scenarios=24, seed=26, objective="x1000"),
    Settings(universe="tout", lever=3.0, scenarios=24, seed=27, objective="x1000"),
    Settings(universe="tout", lever=10.0, scenarios=24, seed=28, objective="x1000")]

GRID_MONDE2 = [Settings(universe="tout", lever=25.0, scenarios=24, seed=15),
               Settings(universe="tout", lever=1.0, scenarios=0, seed=16),
               Settings(universe="tout", lever=5.0, scenarios=24, seed=17),
               Settings(universe="tout", lever=3.0, scenarios=24, seed=18)]


def main_monde(pop: int = 40000, workers: int = 4, data_path: str = "data/history-marche.json.gz", out_dir: str = "data",
               grid=None, offset: int = 0, report_path: str = "data/monde-report.txt"):
    import multiprocessing as mp
    jobs = []
    for i, s in enumerate(grid or GRID_MONDE):
        s.pop = pop
        jobs.append((offset + i, s, data_path, out_dir))
    with mp.Pool(min(workers, len(jobs))) as p:
        res = p.map(_run_tout, jobs)
    text = report_tout(res, report_path)
    Path(report_path).write_text(text + "\n\n" + generations_report(res), encoding="utf-8")
    return res


def generations_report(res: list) -> str:
    """Les générations deviennent-elles plus intelligentes et plus raisonnables ? Année par année."""
    from datetime import datetime, timezone
    L = ["=" * 124, "  LES GÉNÉRATIONS S'AMÉLIORENT-ELLES ? (moyenne des sélections de chaque année)",
         "  ruinés = part de la population morte par ruine ou liquidation (cumul de l'année) ;",
         "  gain médian = résultat réel d'un cerveau typique sur ses 90 derniers jours ; levier = levier médian des vivants", "=" * 124]
    for r in res:
        s = Settings(**r["settings"])
        L.append(f"  {s.name()}")
        by_year: dict = {}
        for g in r.get("gen_stats", []):
            y = datetime.fromtimestamp(g["day"] * 86400, tz=timezone.utc).year
            by_year.setdefault(y, []).append(g)
        for y, gs in sorted(by_year.items()):
            L.append(f"    {y} : ruinés {sum(g['ruined'] for g in gs) * 100:5.1f} % | gain médian sur 90 j "
                     f"{(math.exp(np.mean([g['median'] for g in gs])) - 1) * 100:+6.1f} % | levier x{np.mean([g['lev'] for g in gs]):.1f} "
                     f"| vendeurs à découvert {np.mean([g['short'] for g in gs]) * 100:.0f} %")
    L.append("=" * 124)
    return "\n".join(L)


# ---------------------------------------------------------------- MONDE 4 : décider plus souvent + gérer l'argent
GRID_MONDE4 = [
    Settings(universe="crypto", bpd=1, lever=1.0, scenarios=24, scen_len=90, cand=200, pop=40000, seed=41),
    Settings(universe="crypto", bpd=6, lever=1.0, scenarios=12, scen_len=30, cand=100, pop=20000, seed=42),
    Settings(universe="crypto", bpd=24, lever=1.0, scenarios=12, scen_len=30, cand=100, pop=10000, seed=43),
    Settings(universe="crypto", bpd=24, lever=3.0, scenarios=12, scen_len=30, cand=100, pop=10000, seed=44, objective="x1000"),
    Settings(universe="crypto", bpd=6, lever=3.0, scenarios=12, scen_len=30, cand=100, pop=20000, seed=45, objective="x1000"),
    Settings(universe="crypto", bpd=1, lever=3.0, scenarios=24, scen_len=90, cand=200, pop=40000, seed=46, objective="x1000"),
    Settings(universe="crypto", bpd=1, lever=1.0, scenarios=24, scen_len=90, cand=200, pop=40000, seed=47, mm=False),
    Settings(universe="crypto", bpd=24, lever=1.0, scenarios=12, scen_len=30, cand=100, pop=10000, seed=48, mm=False),
]


def _run_monde4(args):
    i, s, data_path, out_dir = args
    import logging
    logging.disable(logging.CRITICAL)
    from .broker import resample
    from .dataset import Dataset
    from .evolution import year_windows
    h1 = Dataset.load(data_path).bars
    bars = h1 if s.bpd == 24 else {k: resample(list(v), 1440 // s.bpd) for k, v in h1.items()}
    m = build_market(bars, bpd=s.bpd)
    t0 = year_windows(m, [2018])[0][1]
    res = live(m, s, t0, len(m.days))
    res["yearly"] = {int(y): v for y, v in yearly(res["fund_curve"]).items()}
    daily = build_market({k: resample(list(v), 1440) for k, v in h1.items()})        # références en journalier
    res["references"] = references(daily, year_windows(daily, [2018])[0][1])
    Path(out_dir, f"tout-{i:02d}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def main_monde4(workers: int = 8, data_path: str = "data/history-binance-1h.json.gz", out_dir: str = "data",
                wait_for: str | None = None) -> list:
    import multiprocessing as mp
    import time
    while wait_for and not Path(wait_for).exists():         # démarre juste après l'expérience précédente
        time.sleep(60)
    jobs = [(30 + i, s, data_path, out_dir) for i, s in enumerate(GRID_MONDE4)]
    with mp.Pool(min(workers, len(jobs))) as p:
        res = p.map(_run_monde4, jobs)
    text = report_tout(res, "data/monde4-report.txt")
    Path("data/monde4-report.txt").write_text(text + "\n\n" + generations_report(res), encoding="utf-8")
    best = max(res, key=lambda r: r["fund_curve"][-1][1])
    Path("data/cerveau-champion.json").write_text(json.dumps(
        {"settings": best["settings"], "leaders": best["final_leaders"], "fund_final": best["fund_curve"][-1][1],
         "yearly": best["yearly"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------- MONDE 5 : les relations entre marchés aident-elles ?
GRID_MONDE5 = []
for _seed, _lev, _obj in ((51, 1.0, "robuste"), (52, 3.0, "robuste"), (53, 3.0, "x1000"), (54, 1.0, "x1000")):
    for _inter in (True, False):                  # mêmes cerveaux, avec et sans les sens intermarchés
        GRID_MONDE5.append(Settings(universe="tout", lever=_lev, scenarios=24, seed=_seed, objective=_obj, inter=_inter))


def main_monde5(pop: int = 40000, workers: int = 8, wait_for: str | None = None) -> list:
    import time
    while wait_for and not Path(wait_for).exists():
        time.sleep(60)
    return main_monde(pop, workers, grid=GRID_MONDE5, offset=50, report_path="data/monde5-report.txt")


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if a and a[0] == "monde5":
        main_monde5(int(a[1]) if len(a) > 1 else 40000, int(a[2]) if len(a) > 2 else 8, wait_for=a[3] if len(a) > 3 else None)
    elif a and a[0] == "monde4":
        main_monde4(int(a[1]) if len(a) > 1 else 8, wait_for=a[2] if len(a) > 2 else None)
    elif a and a[0] == "monde3":
        main_monde(int(a[1]) if len(a) > 1 else 40000, int(a[2]) if len(a) > 2 else 8, grid=GRID_MONDE3, offset=20,
                   report_path="data/monde3-report.txt")
    elif a and a[0] == "monde2":
        main_monde(int(a[1]) if len(a) > 1 else 40000, int(a[2]) if len(a) > 2 else 4, grid=GRID_MONDE2, offset=10,
                   report_path="data/monde2-report.txt")
    elif a and a[0] == "monde":
        main_monde(int(a[1]) if len(a) > 1 else 40000, int(a[2]) if len(a) > 2 else 4)
    elif a and a[0] == "tout":
        main_tout(int(a[1]) if len(a) > 1 else 10000, int(a[2]) if len(a) > 2 else 8)
    else:
        main(int(a[0]) if a else 20000, int(a[1]) if len(a) > 1 else 6)
