"""Le laboratoire de l'idée de Dhia (02/11/2026), simulation seulement.

L'idée : beaucoup de trades par jour, plusieurs en même temps, une durée libre ; couper une partie quand on gagne,
laisser courir le reste tant que le gain continue, sortir quand il s'essouffle. La question : jusqu'où peut-elle aller,
une fois développée au maximum, avec les frais réels ?

Le moteur compilé (numba) fait vivre chaque cerveau bougie de 5 minutes par bougie de 5 minutes sur 9 cryptos. Mêmes
règles de broker que `vrai_monde.py` : ordre à l'ouverture suivante, écart, impact, plafond de 5 % du volume, financement
réel, liquidation sur le plus bas ou le plus haut de la bougie. Si le stop et la prise de gain sont touchés dans la même
bougie, c'est le stop qui passe d'abord (cas prudent).

Le protocole honnête, à ne jamais casser :
  - APPRENDRE : février 2018 -> 2022. L'évolution ne voit que ces années.
  - VALIDER   : 2023-2024. Sert à comparer les étapes, jamais à faire évoluer.
  - COFFRE    : 2025 -> octobre 2026. Ouvert une seule fois, à la toute fin, sur le choix final.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numba as nb
import numpy as np

from . import vrai_monde as v
from .donnees5m import ACTIFS

DOSSIER = Path("data/idee")
CACHE = DOSSIER / "cache"
K = 4
TFS = np.array([1, 3, 6, 12, 48, 288], dtype=np.int64)     # 5 min, 15 min, 30 min, 1 h, 4 h, 1 jour
NOMS_FLUX = ["acheteurs pressés 5 min", "acheteurs pressés 1 h", "acheteurs pressés 4 h", "financement moyen 3 jours"]
NOMS_X5 = v.NOMS_5M + NOMS_FLUX
F5, FH, FD = len(NOMS_X5), v.F_H, v.F_D
NOMS = NOMS_X5 + v.NOMS_1H + [n + " (jour)" for n in v.FEATURE_NAMES]
F_T = len(NOMS)
PRIX_5M = np.arange(0, len(v.NOMS_5M))                     # signaux de court terme, prix et volume
FLUX = np.arange(len(v.NOMS_5M), F5)
HEURE = np.arange(F5, F5 + FH)
JOUR = np.arange(F5 + FH, F_T)
I_AGIT = v.NOMS_5M.index("agitation 1 h / 1 jour")
PER = 30 * v.BPD

# gènes réels
TH_IN, TH_OUT, STOP, TRAIL, ACT, TP, F1, TVOL, MAXPOS, GROSS, RESERVE, PYR_K, PYR_F, ADA, VMIN = range(15)
NG = 15
BORNES = np.array([(0.02, 3.0), (-3.0, 3.0), (0.3, 30.0), (0.0, 30.0), (0.0, 30.0), (0.0, 30.0), (0.05, 1.0), (0.02, 5.0), (0.02, 10.0),
                   (1.0, 25.0), (0.0, 0.9), (0.0, 20.0), (0.0, 1.0), (0.0, 3.0), (-1.0, 2.0)])
# gènes entiers
I_TF, I_ECH, I_NPOS, I_HOLD, I_PAUSE, I_SHORT, I_PERP, I_MAKER, I_FILTRE, I_HMASK, I_BE, I_ACTIFS = range(12)
NI = 12


@dataclass
class Pop:
    G: np.ndarray
    I: np.ndarray
    Fid: np.ndarray
    W: np.ndarray

    def __len__(self):
        return len(self.G)

    def take(self, i) -> "Pop":
        return Pop(self.G[i].copy(), self.I[i].copy(), self.Fid[i].copy(), self.W[i].copy())

    @staticmethod
    def concat(ps: list["Pop"]) -> "Pop":
        return Pop(*(np.concatenate([getattr(p, f) for p in ps]) for f in ("G", "I", "Fid", "W")))


# ---------------------------------------------------------------- le moteur compilé
@nb.njit(nogil=True, cache=True)
def _un(p, O, H, L, C, Q, SIG, ATR5, VOL5, OK, X5, IH, ID, XH, ATRH, VOLH, UP, XD, ATRD, VOLD, TREND, FI, RATES, HB,
        G, I, Fid, W, t0, t1, per, DE, MMR, LEVA, FS, FP, FM, S, R, SF, TPF):
    A = O.shape[1]
    g, ii = G[p], I[p]
    th_in, th_out, stop_k, trail, act, tp, f1 = g[0], g[1], g[2], g[3], g[4], g[5], g[6]
    tvol, maxpos, gross, reserve, pyr_k, pyr_f, ada, vmin = g[7], g[8], g[9], g[10], g[11], g[12], g[13], g[14]
    tf = TFS[ii[0]]
    ech, npos, hold, pause = ii[1], ii[2], ii[3] * tf, ii[4]
    short = ii[5] > 0
    perp = ii[6] > 0 or short or gross > 1.0001
    maker = ii[7] > 0
    filtre, hmask = ii[8], ii[9]
    be = ii[10] > 0
    amask = ii[11]
    f_ent = FP if perp else FS
    f_mk = FM if perp else FS
    F5_, FH_ = X5.shape[2], XH.shape[2]
    side = np.zeros(A, np.int8)
    pend = np.zeros(A, np.int8)
    tpd = np.zeros(A, np.bool_)
    qty, ref, entry, best, stp, atre = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
    pn, plim, patr, score, addpx = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
    age = np.zeros(A, np.int64)
    cool = np.zeros(A, np.int64)
    eq, res, hwm, ema = 1000.0, 0.0, 1000.0, 0.0
    rt, frais, gliss, fund = 0.0, 0.0, 0.0, 0.0
    vivant = True
    s_i = 0
    t_mort = t1
    nb_ouv = 0
    nb_pend = 0
    for t in range(t0, t1):
        if (t - t0) % per == 0:
            S[p, s_i] = eq
            s_i += 1
        if nb_ouv > 0 or nb_pend > 0:
            pire = 0.0
            expo = 0.0
            maint = 0.0
            fi = FI[t]
            for a in range(A):
                if pend[a] != 0:
                    s = int(pend[a])
                    pend[a] = 0
                    nb_pend -= 1
                    n = pn[a]
                    o = float(O[t, a])
                    sg = float(SIG[t, a])
                    fait = False
                    px, frs = 0.0, f_ent
                    if maker:
                        lim = plim[a]
                        if (s > 0 and L[t, a] < lim) or (s < 0 and H[t, a] > lim):
                            px = lim
                            frs = f_mk
                            fait = True
                    else:
                        hs = DE[a] + 0.03 * sg
                        imp = sg * math.sqrt(n / max(float(Q[t, a]), 1.0))
                        px = o * (1 + s * (hs + imp))
                        gliss += n * (hs + imp)
                        frs = f_ent
                        fait = True
                    if fait:
                        q = n / px
                        c_ = n * frs
                        eq -= c_
                        frais += c_
                        if side[a] == 0:
                            nb_ouv += 1
                        side[a] = s
                        qty[a] = q
                        ref[a] = px
                        entry[a] = px
                        addpx[a] = px
                        best[a] = px
                        atre[a] = patr[a]
                        stp[a] = px * (1 - s * (SF if SF > 0 else stop_k * patr[a]))
                        tpd[a] = False
                        age[a] = 0
                if side[a] == 0:
                    continue
                s = int(side[a])
                o, h, l, c = float(O[t, a]), float(H[t, a]), float(L[t, a]), float(C[t, a])
                sg = float(SIG[t, a])
                if (s > 0 and l <= stp[a]) or (s < 0 and h >= stp[a]):
                    glis = DE[a] + 0.53 * sg
                    px = min(o, stp[a]) * (1 - glis) if s > 0 else max(o, stp[a]) * (1 + glis)
                    eq += s * qty[a] * (px - ref[a])
                    n = qty[a] * px
                    c_ = n * f_ent
                    eq -= c_
                    frais += c_
                    gliss += n * glis
                    ema = 0.9 * ema + 0.1 * math.log(max(1e-6, 1 + s * (px / entry[a] - 1)))
                    side[a] = 0
                    qty[a] = 0.0
                    nb_ouv -= 1
                    rt += 1
                    cool[a] = t + pause
                    continue
                if (tp > 0 or TPF > 0) and not tpd[a]:
                    cible = entry[a] * (1 + s * (TPF if TPF > 0 else tp * atre[a]))
                    if (s > 0 and h >= cible) or (s < 0 and l <= cible):
                        q = qty[a] * f1
                        eq += s * q * (cible - ref[a])
                        c_ = q * cible * f_mk
                        eq -= c_
                        frais += c_
                        qty[a] -= q
                        tpd[a] = True
                        if be:
                            stp[a] = max(stp[a], entry[a]) if s > 0 else min(stp[a], entry[a])
                        rt += f1
                eq += s * qty[a] * (c - ref[a])
                ref[a] = c
                if s > 0:
                    best[a] = max(best[a], c)
                    pire += qty[a] * (l - c)
                else:
                    best[a] = min(best[a], c)
                    pire += qty[a] * (c - h)
                gain = s * (best[a] / entry[a] - 1)
                if trail > 0 and gain >= act * atre[a]:
                    ns = best[a] * (1 - s * trail * atre[a])
                    stp[a] = max(stp[a], ns) if s > 0 else min(stp[a], ns)
                age[a] += 1
                expo += qty[a] * c
                maint += qty[a] * c * MMR[a]
                if fi >= 0 and perp:
                    d = s * qty[a] * c * float(RATES[fi, a])
                    eq -= d
                    fund += d
            if perp and expo > 0 and eq + pire <= maint:
                eq = res
                for a in range(A):
                    side[a] = 0
                    qty[a] = 0.0
                    pend[a] = 0
                nb_ouv = 0
                nb_pend = 0
            if eq > hwm:
                res = min(res + reserve * (eq - hwm), eq)
                hwm = eq
            if eq <= 100.0:
                vivant = False
                t_mort = t
                break
        if t % tf != 0:
            continue
        ih, id_ = IH[t], ID[t]
        expo = 0.0
        for a in range(A):
            if ((amask >> a) & 1) == 0 and side[a] == 0:
                score[a] = 0.0
                continue
            x = 0.0
            for k in range(Fid.shape[1]):
                w = W[p, k]
                if w != 0:
                    f = Fid[p, k]
                    if f < F5_:
                        x += w * X5[t, a, f]
                    elif f < F5_ + FH_:
                        x += w * XH[ih, a, f - F5_]
                    else:
                        x += w * XD[id_, a, f - F5_ - FH_]
            score[a] = x
            if side[a] == 0:
                continue
            s = int(side[a])
            if s * x < th_out or (hold > 0 and age[a] >= hold):
                c = float(C[t, a])
                hs = DE[a] + 0.03 * float(SIG[t, a])
                px = c * (1 - s * hs)
                eq += s * qty[a] * (px - ref[a])
                n = qty[a] * px
                c_ = n * f_ent
                eq -= c_
                frais += c_
                gliss += n * hs
                ema = 0.9 * ema + 0.1 * math.log(max(1e-6, 1 + s * (px / entry[a] - 1)))
                side[a] = 0
                qty[a] = 0.0
                nb_ouv -= 1
                rt += 1
                cool[a] = t + pause
            else:
                expo += qty[a] * float(C[t, a])
        eq_r = max(eq - res, 0.0)
        room = gross * eq_r - expo
        if pyr_k > 0:
            for a in range(A):
                if side[a] != 0 and room > 0:
                    s = int(side[a])
                    c = float(C[t, a])
                    if s * (c / addpx[a] - 1) >= pyr_k * atre[a]:
                        n = min(pyr_f * qty[a] * c, max(maxpos * eq_r - qty[a] * c, 0.0), room, 0.05 * float(Q[t, a]))
                        if n >= 5:
                            hs = DE[a] + 0.03 * float(SIG[t, a])
                            px = c * (1 + s * hs)
                            q = n / px
                            eq -= n * f_ent + s * q * (px - c)
                            frais += n * f_ent
                            gliss += n * hs
                            qty[a] += q
                            addpx[a] = c
                            room -= n
        if ((hmask >> int(HB[t])) & 1) == 0:
            continue
        mult = min(max(math.exp(ada * ema * 5), 0.25), 3.0)
        while nb_ouv + nb_pend < npos and room >= 5:
            top, ab = -1, th_in
            for a in range(A):
                if side[a] != 0 or pend[a] != 0 or not OK[t, a] or t < cool[a] or ((amask >> a) & 1) == 0:
                    continue
                x = score[a]
                if abs(x) <= ab or (x < 0 and not short):
                    continue
                s = 1 if x > 0 else -1
                if (filtre == 1 or filtre == 3) and (UP[ih] != (s > 0)):
                    continue
                if (filtre == 2 or filtre == 3) and TREND[id_, a] * s <= 0:
                    continue
                if X5[t, a, I_AGIT] < vmin:
                    continue
                top, ab = a, abs(x)
            if top < 0:
                break
            a = top
            s = 1 if score[a] > 0 else -1
            if ech == 0:
                atr, vol = float(ATR5[t, a]), float(VOL5[t, a])
            elif ech == 1:
                atr, vol = float(ATRH[ih, a]), float(VOLH[ih, a])
            else:
                atr, vol = float(ATRD[id_, a]), float(VOLD[id_, a])
            taille = min(tvol / max(vol, 0.05) * mult, maxpos, LEVA[a])
            n = min(taille * eq_r, room, 0.05 * float(Q[t, a]))
            score[a] = 0.0
            if n < 5:
                continue
            pend[a] = s
            nb_pend += 1
            pn[a] = n
            plim[a] = float(C[t, a])
            patr[a] = atr
            room -= n
    while s_i < S.shape[1]:
        S[p, s_i] = eq
        s_i += 1
    R[p, 0], R[p, 1], R[p, 2], R[p, 3], R[p, 4], R[p, 5] = rt, frais, gliss, fund, float(t_mort - t0), 1.0 if vivant else 0.0


@nb.njit(nogil=True, parallel=True, cache=True)
def _tous(O, H, L, C, Q, SIG, ATR5, VOL5, OK, X5, IH, ID, XH, ATRH, VOLH, UP, XD, ATRD, VOLD, TREND, FI, RATES, HB,
          G, I, Fid, W, t0, t1, per, DE, MMR, LEVA, FS, FP, FM, S, R, SF, TPF):
    for p in nb.prange(G.shape[0]):
        _un(p, O, H, L, C, Q, SIG, ATR5, VOL5, OK, X5, IH, ID, XH, ATRH, VOLH, UP, XD, ATRD, VOLD, TREND, FI, RATES, HB,
            G, I, Fid, W, t0, t1, per, DE, MMR, LEVA, FS, FP, FM, S, R, SF, TPF)


NOMS_D = ("O", "H", "L", "C", "Q", "SIG", "ATR5", "VOL5", "OK", "X5", "IH", "ID", "XH", "ATRH", "VOLH", "UP", "XD", "ATRD", "VOLD", "TREND", "FI", "RATES", "HB")


def simuler(D: dict, p: Pop, t0: int, t1: int, per: int = PER, stop_fixe: float = 0.0, cible_fixe: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie S : capital toutes les `per` bougies ; R : allers-retours, frais, écart, financement, durée de vie, vivant."""
    ns = (t1 - t0 - 1) // per + 2
    S = np.zeros((len(p), ns), np.float32)
    R = np.zeros((len(p), 6))
    _tous(*(D[n] for n in NOMS_D), p.G, p.I, p.Fid, p.W, t0, t1, per, v.DEMI_ECART[: D["O"].shape[1]].astype(np.float64),
          v.MMR_CRYPTO[: D["O"].shape[1]].astype(np.float64), v.LEV_CRYPTO[: D["O"].shape[1]].astype(np.float64),
          float(FRAIS["spot"]), float(FRAIS["perp"]), float(FRAIS["maker"]), S, R, float(stop_fixe), float(cible_fixe))
    return S, R


FRAIS = {"spot": v.FRAIS_SPOT, "perp": v.FRAIS_PERP, "maker": v.FRAIS_MAKER}


# ---------------------------------------------------------------- les données
def preparer(dossier: str = "data/5m") -> None:
    M = v.construire()
    Ac = M.Ac
    T = len(M.ts)
    t0 = int(M.ts[0])
    tq = np.zeros((T, Ac), np.float64)
    for j, a in enumerate(ACTIFS):
        f = np.load(f"{dossier}/{a}-flux.npz")
        i = (f["ts"] - t0) // v.STEP
        keep = (i >= 0) & (i < T)
        tq[i[keep], j] = f["tq"][keep]
    q = M.qv.astype(np.float64)

    def part(k):
        cs = np.vstack([np.zeros((k, Ac)), np.cumsum(q, axis=0)])
        ct = np.vstack([np.zeros((k, Ac)), np.cumsum(tq, axis=0)])
        return np.clip(2 * (ct[k:] - ct[:-k]) / np.maximum(cs[k:] - cs[:-k], 1.0) - 1, -1, 1) * 3
    p1, p12, p48 = part(1), part(12), part(48)
    r3 = np.zeros((T, Ac), np.float32)
    idx = np.flatnonzero(M.fund_i >= 0)
    if len(idx):
        moy = np.array([M.rates[max(0, i - 8): i + 1].mean(axis=0) for i in range(len(M.rates))], dtype=np.float32)
        pos = np.searchsorted(idx, np.arange(T), side="right") - 1
        r3 = np.clip(moy[np.maximum(pos, 0)] / 0.0003, -3, 3)
    X5 = np.concatenate([M.X5, p1[:, :, None].astype(np.float32), p12[:, :, None].astype(np.float32), p48[:, :, None].astype(np.float32), r3[:, :, None]], axis=2)
    d = v.FEATURE_NAMES
    mom = np.stack([M.Xd[:, :Ac, d.index(f"momentum {k} j")] for k in (14, 28, 56)], axis=2)
    trend = np.where((mom > 0).sum(2) >= 2, 1, -1).astype(np.int8)
    jours = (M.ts % v.D) // (4 * 3_600_000)
    D = {"O": M.o, "H": M.h, "L": M.l, "C": M.c, "Q": M.qv, "SIG": M.sig, "ATR5": M.atr5, "VOL5": M.vol5, "OK": M.ok, "X5": X5, "IH": M.ih, "ID": M.idg,
         "XH": M.Xh, "ATRH": M.atrh, "VOLH": M.volh, "UP": M.up_h, "XD": M.Xd[:, :Ac].copy(), "ATRD": M.atrd[:, :Ac].copy(), "VOLD": M.vold[:, :Ac].copy(),
         "TREND": trend, "FI": M.fund_i, "RATES": M.rates, "HB": jours.astype(np.int8), "TS": M.ts}
    CACHE.mkdir(parents=True, exist_ok=True)
    for k, x in D.items():
        np.save(CACHE / f"{k}.npy", np.asarray(x))


def charger() -> dict:
    D = {}
    for n in NOMS_D + ("TS",):
        x = np.load(CACHE / f"{n}.npy")
        if n in ("O", "H", "L", "C", "Q", "SIG", "ATR5", "VOL5", "ATRH", "VOLH", "ATRD", "VOLD", "RATES"):
            x = x.astype(np.float32)
        D[n] = x
    D["IH"], D["ID"], D["FI"] = D["IH"].astype(np.int64), D["ID"].astype(np.int64), D["FI"].astype(np.int64)
    D["OK"], D["UP"] = D["OK"].astype(np.bool_), D["UP"].astype(np.bool_)
    return D


def bougie(D: dict, date: str) -> int:
    t = int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp() * 1000)
    return int(np.searchsorted(D["TS"], t))


# ---------------------------------------------------------------- les pouvoirs, étape par étape
@dataclass
class Pouvoirs:
    nom: str
    feats: np.ndarray
    libres: tuple
    tf: tuple = (0, 1, 2, 3)
    levier: float = 1.0
    min_ar: float = 0.0             # allers-retours par jour imposés (idée de Dhia : beaucoup de trades)
    objectif: str = "robuste"       # robuste : moyenne des années + pire année ; gain : moyenne seule
    fixes: dict = field(default_factory=dict)
    stop_fixe: float = 0.0
    cible_fixe: float = 0.0


BASE = ("trail", "stop", "th", "taille", "npos", "maker", "short", "perp", "signaux")
ETAPES = {
    "0-base": dict(nom="Ton idée de base : signaux courts, stop suiveur, plusieurs positions", feats=np.concatenate([PRIX_5M]), libres=BASE),
    "1-tendance": dict(nom="+ tendance de fond : trader dans le sens du jour et de l'heure", feats=np.concatenate([PRIX_5M, HEURE, JOUR]),
                       libres=BASE + ("filtre", "ech"), tf=(0, 1, 2, 3, 4, 5)),
    "2-gains": dict(nom="+ prendre une partie du gain, laisser courir le reste", feats=np.concatenate([PRIX_5M, HEURE, JOUR]),
                    libres=BASE + ("filtre", "ech", "tp", "act", "be", "pause", "hold"), tf=(0, 1, 2, 3, 4, 5)),
    "3-flux": dict(nom="+ flux d'ordres, financement, heures et agitation", feats=np.arange(F_T),
                   libres=BASE + ("filtre", "ech", "tp", "act", "be", "pause", "hold", "hmask", "vmin"), tf=(0, 1, 2, 3, 4, 5)),
    "4-argent": dict(nom="+ gestion de l'argent : renfort, réserve, taille selon les résultats", feats=np.arange(F_T),
                     libres=BASE + ("filtre", "ech", "tp", "act", "be", "pause", "hold", "hmask", "vmin", "pyr", "reserve", "ada"), tf=(0, 1, 2, 3, 4, 5)),
}


def pouvoirs(cle: str, levier: float = 1.0, min_ar: float = 0.0, objectif: str = "robuste") -> Pouvoirs:
    e = dict(ETAPES[cle])
    if levier > 1:
        e["libres"] = e["libres"] + ("gross",)
    return Pouvoirs(levier=levier, min_ar=min_ar, objectif=objectif, **e)


def imposer(p: Pop, pv: Pouvoirs) -> Pop:
    """Remet aux valeurs neutres ce que cette étape n'a pas le droit d'utiliser, et borne tout le reste."""
    G, I = p.G, p.I
    G[:] = np.clip(G, BORNES[:, 0], BORNES[:, 1])
    L = pv.libres
    if "act" not in L:
        G[:, ACT] = 0
    if "tp" not in L:
        G[:, TP] = 0
    if "pyr" not in L:
        G[:, PYR_K] = 0
    if "reserve" not in L:
        G[:, RESERVE] = 0
    if "ada" not in L:
        G[:, ADA] = 0
    if "vmin" not in L:
        G[:, VMIN] = -1
    if "gross" not in L:
        G[:, GROSS] = 1.0
    G[:, GROSS] = np.minimum(G[:, GROSS], pv.levier)
    G[:, TH_OUT] = np.minimum(G[:, TH_OUT], G[:, TH_IN] - 0.01)
    G[:, MAXPOS] = np.minimum(G[:, MAXPOS], G[:, GROSS])
    if "filtre" not in L:
        I[:, I_FILTRE] = 0
    if "ech" not in L:
        I[:, I_ECH] = 0
    if "be" not in L:
        I[:, I_BE] = 0
    if "pause" not in L:
        I[:, I_PAUSE] = 0
    if "hold" not in L:
        I[:, I_HOLD] = 0
    if "hmask" not in L:
        I[:, I_HMASK] = 63
    I[:, I_NPOS] = np.clip(I[:, I_NPOS], 1, 9)
    I[:, I_TF] = np.array(pv.tf)[np.minimum(np.searchsorted(pv.tf, I[:, I_TF]), len(pv.tf) - 1)]
    I[I[:, I_ACTIFS] == 0, I_ACTIFS] = 511
    for k, val in pv.fixes.get("G", {}).items():
        G[:, k] = val
    for k, val in pv.fixes.get("I", {}).items():
        I[:, k] = val
    return p


def au_hasard(n: int, pv: Pouvoirs, rng: np.random.Generator) -> Pop:
    G = rng.uniform(BORNES[:, 0], BORNES[:, 1], (n, NG))
    G[:, TH_IN] = np.exp(rng.uniform(math.log(0.05), math.log(2.5), n))
    G[:, STOP] = np.exp(rng.uniform(math.log(0.5), math.log(20), n))
    G[:, TRAIL] = np.where(rng.random(n) < 0.15, 0.0, np.exp(rng.uniform(math.log(0.5), math.log(20), n)))
    G[:, ACT] = np.where(rng.random(n) < 0.5, 0.0, np.exp(rng.uniform(math.log(0.5), math.log(10), n)))
    G[:, TP] = np.where(rng.random(n) < 0.3, 0.0, np.exp(rng.uniform(math.log(0.5), math.log(20), n)))
    G[:, TVOL] = np.exp(rng.uniform(math.log(0.05), math.log(5), n))
    G[:, MAXPOS] = np.exp(rng.uniform(math.log(0.05), math.log(max(pv.levier, 1.0)), n))
    G[:, GROSS] = np.exp(rng.uniform(0, math.log(max(pv.levier, 1.0001)), n))
    G[:, PYR_K] = np.where(rng.random(n) < 0.6, 0.0, rng.uniform(0.5, 10, n))
    G[:, RESERVE] = np.where(rng.random(n) < 0.6, 0.0, rng.uniform(0, 0.8, n))
    G[:, ADA] = np.where(rng.random(n) < 0.6, 0.0, rng.uniform(0, 3, n))
    G[:, VMIN] = np.where(rng.random(n) < 0.6, -1.0, rng.uniform(-0.5, 1.5, n))
    I = np.zeros((n, NI), np.int64)
    I[:, I_TF] = rng.choice(pv.tf, n)
    I[:, I_ECH] = rng.integers(0, 3, n)
    I[:, I_NPOS] = rng.integers(1, 10, n)
    I[:, I_HOLD] = np.where(rng.random(n) < 0.6, 0, rng.integers(2, 200, n))
    I[:, I_PAUSE] = np.where(rng.random(n) < 0.6, 0, rng.integers(1, 288, n))
    I[:, I_SHORT] = rng.random(n) < 0.5
    I[:, I_PERP] = rng.random(n) < 0.7
    I[:, I_MAKER] = rng.random(n) < 0.6
    I[:, I_FILTRE] = rng.integers(0, 4, n)
    I[:, I_HMASK] = rng.integers(1, 64, n)
    I[:, I_BE] = rng.random(n) < 0.5
    I[:, I_ACTIFS] = rng.integers(1, 512, n)
    Fid = rng.choice(pv.feats, (n, K)).astype(np.int64)
    W = (rng.uniform(-1, 1, (n, K)) * (rng.random((n, K)) < 0.8)).astype(np.float64)
    return imposer(Pop(G, I, Fid, W), pv)


def enfants(pop: Pop, fit: np.ndarray, n: int, pv: Pouvoirs, rng: np.random.Generator) -> Pop:
    def tour():
        c = rng.integers(0, len(pop), (n, 4))
        return c[np.arange(n), np.argmax(fit[c], axis=1)]
    a, b = tour(), tour()
    m = rng.random((n, NG)) < 0.5
    G = np.where(m, pop.G[a], pop.G[b])
    m = rng.random((n, NI)) < 0.5
    I = np.where(m, pop.I[a], pop.I[b])
    sig = rng.random(n) < 0.5                  # les signaux passent en bloc, sinon ils perdent leur sens
    Fid = np.where(sig[:, None], pop.Fid[a], pop.Fid[b])
    W = np.where(sig[:, None], pop.W[a], pop.W[b])
    ech = (BORNES[:, 1] - BORNES[:, 0]) * 0.06
    mu = rng.normal(0, 1, (n, NG)) * ech * (rng.random((n, NG)) < 0.2)
    G = G + mu
    W = W + rng.normal(0, 0.25, W.shape) * (rng.random(W.shape) < 0.25)
    Fid = np.where(rng.random(Fid.shape) < 0.08, rng.choice(pv.feats, Fid.shape), Fid)
    fr = rng.random((n, NI)) < 0.06
    bruit = au_hasard(n, pv, rng).I
    I = np.where(fr, bruit, I)
    return imposer(Pop(G, I, Fid.astype(np.int64), np.clip(W, -1, 1)), pv)


def note(S: np.ndarray, R: np.ndarray, duree_bougies: int, pv: Pouvoirs) -> np.ndarray:
    pas = min(12, S.shape[1] - 1)
    idx = np.arange(0, S.shape[1] - pas, pas)
    logs = np.log(np.maximum(S[:, idx + pas], 1.0) / np.maximum(S[:, idx], 1.0))
    logs = np.clip(logs, -3, math.log(1000))
    moy, pire = logs.mean(1), logs.min(1)
    f = moy if pv.objectif == "gain" else moy + 0.25 * pire
    f = f - 3 * (R[:, 5] == 0)
    jours = np.maximum(R[:, 4] / v.BPD, 1)
    ar = R[:, 0] / jours
    f = f + 0.1 * np.log1p(ar) * (moy > 0)      # la règle de Dhia : à gain égal, le plus de trades possible
    if pv.min_ar > 0:                           # règle stricte : sous le rythme imposé, seul le nombre de trades compte
        f = np.where(ar >= pv.min_ar, f, -50 + ar)
    return f


GROUPE_A = 0b001010011       # BTC, ETH, BNB, ADA... : la moitié des cryptos
GROUPE_B = 0b110101100       # l'autre moitié


def note_croisee(D: dict, pop: Pop, t0: int, t1: int, pv: Pouvoirs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Un cerveau doit réussir sur deux groupes de cryptos séparés : ce qu'il apprend par coeur sur l'un ne marche pas sur l'autre."""
    notes, Ss, Rs = [], [], []
    for m in (GROUPE_A, GROUPE_B):
        p = pop.take(np.arange(len(pop)))
        p.I[:, I_ACTIFS] = m
        S, R = simuler(D, p, t0, t1, stop_fixe=pv.stop_fixe, cible_fixe=pv.cible_fixe)
        notes.append(note(S, R, t1 - t0, pv))
        Ss.append(S)
        Rs.append(R)
    return np.minimum(notes[0], notes[1]), Ss[0], Rs[0]


def evoluer(D: dict, pv: Pouvoirs, n: int, gens: int, t0: int, t1: int, seed: int = 1, graines: Pop | None = None, log=None, croise: bool = False) -> tuple[Pop, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop = au_hasard(n, pv, rng)
    if graines is not None and len(graines):
        k = min(len(graines), n // 10)
        g = imposer(graines.take(rng.permutation(len(graines))[:k]), pv)
        pop = Pop.concat([g, pop.take(np.arange(n - k))])
    for g_ in range(gens):
        t = time.time()
        if croise:
            fit, S, R = note_croisee(D, pop, t0, t1, pv)
        else:
            S, R = simuler(D, pop, t0, t1, stop_fixe=pv.stop_fixe, cible_fixe=pv.cible_fixe)
            fit = note(S, R, t1 - t0, pv)
        o = np.argsort(-fit)
        if log and (g_ % 5 == 0 or g_ == gens - 1):
            b = o[0]
            jours = max(R[b, 4] / v.BPD, 1)
            log(f"    gén {g_:3d} | meilleur 1 000 -> {S[b, -1]:,.0f} | {R[b, 0] / jours:.1f} a/r par jour | vivants {R[:, 5].mean() * 100:.0f} % | "
                f"au-dessus de 1 000 : {(S[:, -1] > 1000).mean() * 100:.1f} % | {time.time() - t:.0f} s")
        if g_ == gens - 1:
            return pop.take(o), fit[o]
        ne, ni = int(n * 0.05), int(n * 0.10)
        pop = Pop.concat([pop.take(o[:ne]), enfants(pop, fit, n - ne - ni, pv, rng), au_hasard(ni, pv, rng)])
    raise RuntimeError("aucune génération")


def bot_crypto(D: dict) -> Pop:
    """Le bot actuel sur BTC, ETH et SOL, dans ce moteur : sert à vérifier le moteur et à comparer chaque étape."""
    d = v.FEATURE_NAMES
    Fid = np.array([[F5 + FH + d.index(f"momentum {k} j") for k in (14, 28, 56)] + [0]], dtype=np.int64)
    G = np.zeros((1, NG))
    G[0, [TH_IN, TH_OUT, STOP, TVOL, MAXPOS, GROSS, VMIN]] = (0.0, -1e-6, 3.0, 0.15, 0.3, 1.0, -9.0)
    I = np.zeros((1, NI), np.int64)
    I[0, [I_TF, I_ECH, I_NPOS, I_MAKER, I_FILTRE, I_HMASK, I_ACTIFS]] = (5, 2, 3, 0, 0, 63, 0b111)
    return Pop(G, I, Fid, np.array([[1.0, 1.0, 1.0, 0.0]]))


def decrire(p: Pop, i: int) -> str:
    g, ii = p.G[i], p.I[i]
    r = ["5 min", "15 min", "30 min", "1 h", "4 h", "1 jour"][int(ii[I_TF])]
    sig = " ; ".join(f"{p.W[i, k]:+.2f} x {NOMS[p.Fid[i, k]]}" for k in range(K) if abs(p.W[i, k]) > 0.02) or "aucun"
    ac = ", ".join(a.removesuffix("USDT") for j, a in enumerate(ACTIFS) if (ii[I_ACTIFS] >> j) & 1)
    f = ["aucun filtre", "sens du Bitcoin", "sens du jour de l'actif", "Bitcoin et jour"][int(ii[I_FILTRE])]
    return (f"décide toutes les {r} | {'ordres limite' if ii[I_MAKER] else 'ordres au marché'} | {'contrats' if ii[I_PERP] or ii[I_SHORT] else 'spot'} | "
            f"{'achat et vente' if ii[I_SHORT] else 'achat seul'} | {sig} | entrer si > {g[TH_IN]:.2f}, sortir si < {g[TH_OUT]:.2f} | {f} | {ii[I_NPOS]} position(s), "
            f"max {g[MAXPOS] * 100:.0f} %, levier x{g[GROSS]:.1f} | stop {g[STOP]:.1f} ATR, suiveur {g[TRAIL]:.1f} dès {g[ACT]:.1f} ATR, prise de gain {g[F1] * 100:.0f} % à {g[TP]:.1f} ATR | "
            f"pause {ii[I_PAUSE]} bougies, durée max {ii[I_HOLD]} décisions | réserve {g[RESERVE] * 100:.0f} %, renfort {g[PYR_F] * 100:.0f} % à {g[PYR_K]:.1f} ATR | actifs : {ac}")


def fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def essai(D: dict, cle: str, nom: str, n: int, gens: int, levier: float, min_ar: float, objectif: str, graines, fen: dict, seed: int, sortie: Path, croise: bool = False) -> tuple[Pop, str]:
    if cle == "G-dhia":
        pv = version_dhia(levier)
    elif cle.startswith("H-cible"):
        pv = version_cible(levier, float(cle.split("-")[-1]) / 100)
    else:
        pv = pouvoirs(cle, levier, min_ar, objectif)
    print(f"\n=== {nom} | {pv.nom} | levier max x{levier:g} | {min_ar:g} a/r par jour imposés | note {objectif}", flush=True)
    t = time.time()
    pop, fit = evoluer(D, pv, n, gens, fen["app"][0], fen["app"][1], seed, graines, log=lambda s: print(s, flush=True), croise=croise)
    top = pop.take(np.arange(min(200, n)))
    if croise:
        top.I[:, I_ACTIFS] = 511
    np.savez_compressed(sortie / f"{nom}.npz", G=top.G, I=top.I, Fid=top.Fid, W=top.W, fit=fit[:200])
    tv = top.take(np.arange(min(20, len(top))))
    S1, R1 = simuler(D, tv, *fen["app"], stop_fixe=pv.stop_fixe, cible_fixe=pv.cible_fixe)
    S2, R2 = simuler(D, tv, *fen["val"], stop_fixe=pv.stop_fixe, cible_fixe=pv.cible_fixe)
    an1 = np.exp(np.log(np.maximum(S2[:, 12], 1) / 1000)).max()
    b = bot_crypto(D)
    b1, _ = simuler(D, b, *fen["app"])
    b2, _ = simuler(D, b, *fen["val"])
    ar = R2[:, 0] / np.maximum(R2[:, 4] / v.BPD, 1)
    ligne = (f"{nom:26} | apprentissage : médiane {fmt(np.median(S1[:, -1])):>11}, meilleur {fmt(S1[:, -1].max()):>11} (bot {fmt(b1[0, -1])}) | 2023-2024 jamais vu : médiane {fmt(np.median(S2[:, -1])):>9}, "
             f"moyenne {fmt(S2[:, -1].mean()):>9}, meilleur {fmt(S2[:, -1].max()):>9} (bot {fmt(b2[0, -1])}) | battent le bot {int((S2[:, -1] > b2[0, -1]).sum())}/20 | ruinés {int((R2[:, 5] == 0).sum())}/20 | "
             f"{np.median(ar):.1f} a/r par jour | coûts médians {fmt(np.median(R2[:, 1] + R2[:, 2] + R2[:, 3]))} | meilleure 1re année inconnue x{an1:.2f} | {time.time() - t:.0f} s")
    print(ligne, flush=True)
    print("    meilleur : " + decrire(top, 0)[:400], flush=True)
    return top, ligne


def fenetres(D: dict) -> dict:
    return {"app": (bougie(D, "2018-02-01"), bougie(D, "2023-01-01")), "val": (bougie(D, "2023-01-01"), bougie(D, "2025-01-01")),
            "coffre": (bougie(D, "2025-01-01"), len(D["TS"]))}


def vitesse() -> None:
    """Vérifie le moteur contre le bot actuel, puis mesure sa vitesse."""
    D = charger()
    b = bot_crypto(D)
    t = time.time()
    S, R = simuler(D, b, 0, len(D["TS"]))
    print(f"bot BTC/ETH/SOL 2018-2026 dans ce moteur : 1 000 -> {fmt(S[0, -1])} (vrai_monde : 59 129) | {R[0, 0] / (len(D['TS']) / v.BPD):.3f} a/r par jour | compilation + calcul {time.time() - t:.0f} s", flush=True)
    f = fenetres(D)
    for cle, n in (("0-base", 800), ("4-argent", 800)):
        p = au_hasard(n, pouvoirs(cle, 3.0), np.random.default_rng(1))
        t = time.time()
        S, R = simuler(D, p, *f["app"])
        d = time.time() - t
        print(f"{cle} : {n} cerveaux sur 5 ans en {d:.1f} s, soit {d / n * 1000:.1f} ms par cerveau | vivants {R[:, 5].mean() * 100:.0f} % | a/r par jour médian {np.median(R[:, 0] / np.maximum(R[:, 4] / v.BPD, 1)):.1f}", flush=True)


PROGRAMME = (
    # nom, étape, levier max, allers-retours imposés par jour, note
    ("A0-idee-de-base", "0-base", 1.0, 0.0, "robuste"),
    ("A1-tendance", "1-tendance", 1.0, 0.0, "robuste"),
    ("A2-prise-de-gains", "2-gains", 1.0, 0.0, "robuste"),
    ("A3-flux-ordres", "3-flux", 1.0, 0.0, "robuste"),
    ("A4-gestion-argent", "4-argent", 1.0, 0.0, "robuste"),
    ("B1-10-trades-jour", "4-argent", 1.0, 10.0, "robuste"),
    ("B2-25-trades-jour", "4-argent", 1.0, 25.0, "robuste"),
    ("B3-50-trades-jour", "4-argent", 1.0, 50.0, "robuste"),
    ("C1-levier-x3", "4-argent", 3.0, 0.0, "gain"),
    ("C2-levier-x10", "4-argent", 10.0, 0.0, "gain"),
    ("C3-levier-x25", "4-argent", 25.0, 0.0, "gain"),
    ("C4-x3-50-trades-jour", "4-argent", 3.0, 50.0, "gain"),
    ("C5-x10-50-trades-jour", "4-argent", 10.0, 50.0, "gain"),
)


def programme(n: int, gens: int, seed: int = 1) -> None:
    D = charger()
    f = fenetres(D)
    DOSSIER.mkdir(parents=True, exist_ok=True)
    rapport = DOSSIER / "idee-report.txt"
    lignes = ["=" * 150, f"  LE LABORATOIRE DE L'IDÉE — {n} cerveaux x {gens} générations par étape, 5 min, 9 cryptos, frais réels, SANS IMPÔTS, 1 000 au départ de chaque fenêtre", "=" * 150, ""]
    graines = None
    for nom, cle, lev, ar, obj in PROGRAMME:
        if nom[0] in "BC" and nom[1] == "1":
            graines = None if graines is None else graines
        pop, ligne = essai(D, cle, nom, n, gens, lev, ar, obj, graines, f, seed, DOSSIER)
        graines = pop
        lignes.append(ligne)
        rapport.write_text("\n".join(lignes), encoding="utf-8")


PROGRAMME_2 = (
    # nom, étape, levier max, allers-retours imposés, note, double épreuve sur deux groupes de cryptos
    ("D1-10-trades-strict", "2-gains", 1.0, 10.0, "robuste", False),
    ("D2-25-trades-strict", "2-gains", 1.0, 25.0, "robuste", False),
    ("D3-50-trades-strict", "2-gains", 1.0, 50.0, "robuste", False),
    ("D4-50-trades-flux", "3-flux", 1.0, 50.0, "robuste", False),
    ("D5-x3-25-trades-strict", "2-gains", 3.0, 25.0, "gain", False),
    ("E1-tendance-croisee", "1-tendance", 1.0, 0.0, "robuste", True),
    ("E2-10-trades-croisee", "2-gains", 1.0, 10.0, "robuste", True),
)


ELAN = np.array([i for i, n in enumerate(NOMS) if n.startswith("momentum") and (n.endswith("min") or (n.endswith(" h") and i >= F5 and i < F5 + FH))])


def version_dhia(levier: float) -> Pouvoirs:
    """Les réponses de Dhia du 03/11/2026 : entrer quand le prix accélère (hausse ou baisse), stop de 3 %, sortie et nombre de
    trades choisis par l'évolution, 5 % du capital par trade, plus de 4 trades complets par heure, aucune limite de chute."""
    return Pouvoirs(nom="Ta version : accélération, stop 3 %, 5 % par trade, plus de 100 trades par jour, deux sens",
                    feats=ELAN, libres=("trail", "th", "npos", "maker", "signaux", "tp", "act", "be", "pause", "hold", "ech", "gross"),
                    tf=(0, 1, 2, 3), levier=levier, min_ar=100.0, objectif="gain", stop_fixe=0.03,
                    fixes={"G": {MAXPOS: 0.05 * levier, TVOL: 5.0, GROSS: levier, PYR_K: 0.0, RESERVE: 0.0, ADA: 0.0, VMIN: -1.0},
                           "I": {I_SHORT: 1, I_PERP: 1, I_FILTRE: 0, I_HMASK: 63, I_ACTIFS: 511}})


def version_cible(levier: float, cible: float) -> Pouvoirs:
    """Question de Dhia du 03/11/2026 : seulement quand le marché bouge vite, chaque trade coupé en entier à +10, +20 ou +30 %,
    stop de 3 %, entrée sur l'accélération dans les deux sens, 5 % par trade. Le rythme est libre : il dépend du marché."""
    return Pouvoirs(nom=f"Cible fixe +{cible * 100:.0f} %, stop 3 %, marchés rapides seulement", feats=ELAN,
                    libres=("th", "npos", "maker", "signaux", "pause", "hold", "vmin", "gross"), tf=(0, 1, 2, 3, 4), levier=levier,
                    objectif="gain", stop_fixe=0.03, cible_fixe=cible,
                    fixes={"G": {MAXPOS: 0.05 * levier, TVOL: 5.0, GROSS: levier, TRAIL: 0.0, ACT: 0.0, F1: 1.0, TH_OUT: -3.0, PYR_K: 0.0, RESERVE: 0.0, ADA: 0.0},
                           "I": {I_SHORT: 1, I_PERP: 1, I_FILTRE: 0, I_HMASK: 63, I_BE: 0}})


def programme4(n: int, gens: int, seed: int = 5) -> None:
    D = charger()
    f = fenetres(D)
    rapport = DOSSIER / "idee-report-4.txt"
    lignes = ["=" * 150, f"  CIBLE FIXE DE GAIN — {n} cerveaux x {gens} générations, 9 cryptos, frais réels, SANS IMPÔTS", "=" * 150, ""]
    for lev in (1.0, 3.0):
        for c in (10, 20, 30):
            _, ligne = essai(D, f"H-cible-{c}", f"H-cible{c}-x{lev:g}", n, gens, lev, 0.0, "gain", None, f, seed, DOSSIER)
            lignes.append(ligne)
            rapport.write_text("\n".join(lignes), encoding="utf-8")


def programme3(n: int, gens: int, seed: int = 4) -> None:
    D = charger()
    f = fenetres(D)
    rapport = DOSSIER / "idee-report-3.txt"
    lignes = ["=" * 150, f"  TA VERSION EXACTE — {n} cerveaux x {gens} générations, 9 cryptos, frais réels, SANS IMPÔTS", "=" * 150, ""]
    faits = rapport.read_text(encoding="utf-8").splitlines()[4:] if rapport.exists() else []
    lignes += faits                                 # reprise : les essais déjà finis sont gardés
    for lev in (1.0, 3.0, 10.0):
        for cr in (False, True):
            nom = f"G-x{lev:g}{'-croisee' if cr else ''}"
            if any(l.startswith(nom + " ") for l in faits):
                continue
            _, ligne = essai(D, "G-dhia", nom, n, gens, lev, 100.0, "gain", None, f, seed, DOSSIER, croise=cr)
            lignes.append(ligne)
            rapport.write_text("\n".join(lignes), encoding="utf-8")


def marche_avant(D: dict, cle: str, nom: str, n: int, gens: int, levier: float, min_ar: float, seed: int = 2) -> str:
    """L'apprentissage continu : chaque année, l'évolution repart de tout le passé vécu, puis trade l'année suivante, inconnue."""
    pv = pouvoirs(cle, levier, min_ar)
    bot = bot_crypto(D)
    val, val_b, lignes = 1000.0, 1000.0, []
    debut = bougie(D, "2018-02-01")
    graines = None
    for an in (2021, 2022, 2023, 2024):
        t0, t1 = bougie(D, f"{an}-01-01"), bougie(D, f"{an + 1}-01-01")
        print(f"  {nom} : apprend jusqu'à fin {an - 1}, trade {an}", flush=True)
        pop, _ = evoluer(D, pv, n, gens, debut, t0, seed + an, graines)
        graines = pop.take(np.arange(200))
        top = pop.take(np.arange(20))
        S, R = simuler(D, top, t0, t1)
        b, _ = simuler(D, bot, t0, t1)
        m, mb = float(S[:, -1].mean()) / 1000, float(b[0, -1]) / 1000
        val *= m
        val_b *= mb
        ar = float(np.median(R[:, 0] / np.maximum(R[:, 4] / v.BPD, 1)))
        lignes.append(f"{an} : x{m:.2f} (bot x{mb:.2f}, {ar:.1f} a/r par jour, ruinés {int((R[:, 5] == 0).sum())}/20)")
    return f"{nom:26} | apprend chaque année, 2021-2024 : 1 000 -> {fmt(val)} (bot {fmt(val_b)}) | " + " | ".join(lignes)


def programme2(n: int, gens: int, seed: int = 3) -> None:
    D = charger()
    f = fenetres(D)
    rapport = DOSSIER / "idee-report-2.txt"
    lignes = ["=" * 150, f"  L'IDÉE, 2E PASSE — {n} cerveaux x {gens} générations, rythme imposé strict, double épreuve, apprentissage continu", "=" * 150, ""]
    for nom, cle, lev, ar, obj, cr in PROGRAMME_2:
        _, ligne = essai(D, cle, nom, n, gens, lev, ar, obj, None, f, seed, DOSSIER, croise=cr)
        lignes.append(ligne)
        rapport.write_text("\n".join(lignes), encoding="utf-8")
    for nom, cle, lev, ar in (("F1-continu-tendance", "1-tendance", 1.0, 0.0), ("F2-continu-10-trades", "2-gains", 1.0, 10.0)):
        ligne = marche_avant(D, cle, nom, n * 3 // 4, gens * 4 // 5, lev, ar)
        print(ligne, flush=True)
        lignes.append(ligne)
        rapport.write_text("\n".join(lignes), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if a and a[0] == "preparer":
        preparer()
    elif a and a[0] == "vitesse":
        vitesse()
    elif a and a[0] == "programme":
        programme(int(a[1]), int(a[2]))
    elif a and a[0] == "programme2":
        programme2(int(a[1]), int(a[2]))
    elif a and a[0] == "programme3":
        programme3(int(a[1]), int(a[2]))
    elif a and a[0] == "programme4":
        programme4(int(a[1]), int(a[2]))
    print("FIN", flush=True)
