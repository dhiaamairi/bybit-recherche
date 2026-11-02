"""La grande finale : des cerveaux qui vivent sur un vrai broker, bougie de 5 minutes par bougie de 5 minutes.

Demande de Dhia (02/11/2026) : reprendre les meilleurs cerveaux, leur donner le contrôle total de l'argent, du temps
et des actifs, plus de 50 trades par jour permis, tout retenir de leur passé, dans un marché réel : Bybit pour les
cryptos, Interactive Brokers pour le reste, sans impôts. SIMULATION SEULEMENT.

Ce que le simulateur copie d'un vrai broker :
  - ordre décidé à la clôture d'une bougie, exécuté à l'ouverture de la suivante (jamais au prix déjà connu) ;
  - écart achat/vente, glissement selon l'agitation, impact selon la taille face au volume, plafond de 5 % du volume ;
  - Bybit : spot 0,10 %, contrats 0,055 % (preneur) ou 0,02 % (ordre limite, exécuté seulement si le prix traverse) ;
  - financement réel des contrats toutes les 8 h (fichiers Binance, 0,01 % avant septembre 2019) ;
  - IBKR : 0,005 $ par action, 1 $ minimum, 1 % maximum ; devises 0,2 point de base, 2 $ minimum ; intérêts sur le prêt ;
  - liquidation sur le plus bas ou le plus haut de la bougie, levier limité actif par actif, ordre minimum de 5 $.
Hypothèses à connaître : un seul compte de marge pour les deux brokers (optimiste), volumes de Binance pris pour ceux de
Bybit, aucun impôt (choix de Dhia, donc résultats flatteurs).

Les gènes nouveaux : tf (décider toutes les 5, 15 min, 1 h, 4 h ou 1 jour), maker (ordres limite).
La mémoire : une archive garde les meilleurs de chaque époque et les fait revenir ; rien n'est oublié.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arene import RUIN
from .donnees5m import ACTIFS
from .ecosysteme import yearly
from .evolution import FEATURE_NAMES, K, _names_intraday, LOOKBACKS_INTRADAY, _rolling_ext, _rolling_mean, _rolling_std, _rsi, build_market

STEP = 300_000                      # 5 minutes en ms
BPD = 288                           # bougies de 5 minutes par jour
D = 86_400_000
NAISSANCE = 1000.0
TFS = np.array([1, 3, 12, 48, 288])  # décide toutes les 5 min, 15 min, 1 h, 4 h, 1 jour
COUCHE = np.array([0, 0, 1, 1, 2])   # échelle de l'ATR et de la volatilité pour chaque rythme
AUTRES = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "JNJ", "KO", "WMT", "INTC", "GE", "C", "BA", "IBM",
          "PFE", "F", "SPY", "QQQ", "IWM", "EEM", "TLT", "GC=F", "SI=F", "USO", "EURUSD=X", "JPY=X"]
DEVISES = ("EURUSD=X", "JPY=X")
MATIERES = ("GC=F", "SI=F", "USO")
# demi-écart de base par actif (fraction du prix) : BTC et ETH très serrés, petites cryptos plus larges
DEMI_ECART = np.array([0.00003, 0.00003, 0.00005, 0.0001, 0.0001, 0.0001, 0.00015, 0.00015, 0.0002])
MMR_CRYPTO = np.array([0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
LEV_CRYPTO = np.array([100.0, 100.0, 50.0, 50.0, 50.0, 50.0, 25.0, 25.0, 25.0])
FRAIS_SPOT, FRAIS_PERP, FRAIS_MAKER = 0.0010, 0.00055, 0.0002
PLAFOND_VOLUME = 0.05
ORDRE_MIN = 5.0
INTERET_AN = 0.05                   # hypothèse : prêt IBKR en dollars
EMPRUNT_TITRES_AN = 0.003           # hypothèse : vente à découvert d'actions
FUNDING_DEFAUT = 0.0001
NOMS_5M = ([f"momentum {m} min" for m in (15, 30, 60, 120, 240)] + [f"écart à la moyenne {m} min" for m in (15, 30, 60, 120, 240)]
           + ["position dans le canal 1 h", "position dans le canal 4 h", "RSI 14 (5 min)", "agitation 1 h / 1 jour", "pic de volume (1 jour)"])
NOMS_1H = [n if n.split()[0] in ("momentum", "écart", "position") else n + " (1 h)" for n in _names_intraday(LOOKBACKS_INTRADAY[24], 24)]
NOMS = FEATURE_NAMES + NOMS_1H + NOMS_5M
F_T = len(NOMS)
F_D, F_H = len(FEATURE_NAMES), len(NOMS_1H)


@dataclass
class Marche5:
    ts: np.ndarray          # (T,) ouverture de chaque bougie
    o: np.ndarray           # (T, Ac) prix des cryptos
    h: np.ndarray
    l: np.ndarray           # noqa: E741
    c: np.ndarray
    qv: np.ndarray          # volume en dollars de la bougie
    ok: np.ndarray
    X5: np.ndarray          # (T, Ac, f5)
    atr5: np.ndarray
    vol5: np.ndarray        # volatilité annualisée
    sig: np.ndarray         # volatilité d'une bougie
    ih: np.ndarray          # (T,) dernière heure terminée
    idg: np.ndarray         # (T,) dernier jour terminé (pour les signaux)
    idl: np.ndarray         # (T,) même jour, dans les prix journaliers de ce marché
    fund_i: np.ndarray      # (T,) -1, ou numéro du paiement de financement
    rates: np.ndarray       # (N, Ac)
    Xh: np.ndarray          # (Th, Ac, F_H)
    atrh: np.ndarray
    volh: np.ndarray
    up_h: np.ndarray        # tendance du Bitcoin, par heure
    Xd: np.ndarray          # (Td, A, F_D) cryptos puis autres actifs
    atrd: np.ndarray
    vold: np.ndarray
    up_d: np.ndarray        # tendance du S&P 500, par jour
    Do: np.ndarray          # (Td', Ao) prix journaliers des autres actifs
    Dh: np.ndarray
    Dl: np.ndarray
    Dc: np.ndarray
    Dok: np.ndarray
    assets: list
    Ac: int
    jours: np.ndarray       # jour (entier) de chaque ligne journalière de signaux

    @property
    def A(self) -> int:
        return len(self.assets)


def _complete(x: np.ndarray, reel: np.ndarray) -> np.ndarray:
    """Répète le dernier prix connu dans les trous ; avant la cotation, le prix vaut 1 (non négociable)."""
    i = np.where(reel, np.arange(len(x))[:, None], 0)
    i = np.maximum.accumulate(i, axis=0)
    return np.nan_to_num(x[i, np.arange(x.shape[1])], nan=1.0)


def signaux_5m(o, h, l, c, q) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # noqa: E741
    """Signaux de court terme, connus à la clôture de chaque bougie."""
    lc = np.log(c.astype(np.float64))
    r = np.diff(lc, axis=0, prepend=lc[:1])
    sig = _rolling_std(r, BPD)
    sig = np.where(np.isnan(sig) | (sig <= 0), 0.002, sig)
    fe = []
    for k in (3, 6, 12, 24, 48):
        past = np.vstack([lc[:1].repeat(k, axis=0), lc[:-k]])
        fe.append((lc - past) / (sig * math.sqrt(k) + 1e-9) / 3)
    for k in (3, 6, 12, 24, 48):
        fe.append((c / _rolling_mean(c.astype(np.float64), k) - 1) / (sig * math.sqrt(k) + 1e-9) / 3)
    for k in (12, 48):
        hh = np.vstack([np.full((1, c.shape[1]), np.nan), _rolling_ext(h.astype(np.float64), k, np.max)[:-1]])
        ll = np.vstack([np.full((1, c.shape[1]), np.nan), _rolling_ext(l.astype(np.float64), k, np.min)[:-1]])
        fe.append(2 * (c - ll) / np.maximum(hh - ll, 1e-12) - 1)
    fe.append((_rsi(c.astype(np.float64), 14) - 50) / 50)
    fe.append(np.clip(_rolling_std(r, 12) / (sig + 1e-12) - 1, -1, 2))
    qq = q.astype(np.float64)
    fe.append(np.clip(np.log((qq + 1) / (_rolling_mean(qq, BPD) + 1)), -2, 2))
    X = np.clip(np.nan_to_num(np.stack(fe, axis=2), nan=0.0), -3, 3).astype(np.float32)
    prev = np.vstack([c[:1], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = np.nan_to_num(_rolling_mean(tr.astype(np.float64), 20) / c, nan=0.003)
    return X, np.maximum(atr, 1e-4).astype(np.float32), (sig * math.sqrt(365 * BPD)).astype(np.float32), sig.astype(np.float32)


def construire(dossier: str = "data/5m", debut: str = "2018-01-02", fin: str | None = None, actifs: list[str] = ACTIFS,
               h1: str = "data/history-binance-1h.json.gz", j1: str = "data/history-binance.json.gz",
               marche: str = "data/history-marche.json.gz") -> Marche5:
    from datetime import datetime, timezone

    from .dataset import Dataset
    ms = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)  # noqa: E731
    t0 = ms(debut)
    dl = [np.load(f"{dossier}/{a}.npz") for a in actifs]
    t1 = ms(fin) if fin else max(int(d["ts"][-1]) for d in dl) + STEP
    n, Ac = (t1 - t0) // STEP, len(actifs)
    ts = t0 + STEP * np.arange(n, dtype=np.int64)
    arr = {k: np.full((n, Ac), np.nan, np.float32) for k in ("o", "h", "l", "c", "qv")}
    for j, d in enumerate(dl):
        i = (d["ts"] - t0) // STEP
        keep = (i >= 0) & (i < n)
        for k in arr:
            arr[k][i[keep], j] = d[k][keep]
    reel = ~np.isnan(arr["c"])
    c = _complete(arr["c"], reel)
    o, h, l = (np.where(reel, arr[k], c) for k in ("o", "h", "l"))  # noqa: E741
    q = np.nan_to_num(arr["qv"], nan=0.0)
    X5, atr5, vol5, sig = signaux_5m(o, h, l, c, q)
    # financement : un paiement toutes les 8 h
    pos8 = np.flatnonzero(ts % (8 * 3_600_000) == 0)
    fund_i = np.full(n, -1, np.int32)
    fund_i[pos8] = np.arange(len(pos8))
    rates = np.full((len(pos8), Ac), FUNDING_DEFAUT, np.float32)
    for j, a in enumerate(actifs):
        f = np.load(f"{dossier}/{a}-fund.npz")
        if len(f["ts"]):
            s = (f["ts"] + 4 * 3_600_000 - ts[pos8[0]]) // (8 * 3_600_000)      # arrondi au paiement le plus proche
            keep = (s >= 0) & (s < len(pos8))
            somme = np.zeros(len(pos8))
            np.add.at(somme, s[keep], f["rate"][keep])
            vu = np.zeros(len(pos8), bool)
            vu[s[keep]] = True
            rates[vu, j] = somme[vu]
    bars_h = Dataset.load(h1).bars
    mh = build_market(bars_h, actifs, regime=actifs[0], bpd=24)
    cles_h = np.array(sorted({b.ts // 3_600_000 for a in actifs for b in bars_h[a]}), dtype=np.int64)
    del bars_h
    j = Dataset.load(j1).bars
    bars_d = {a: list(j[a]) for a in actifs}
    bars_d.update({a: v for a, v in Dataset.load(marche).bars.items() if a in AUTRES})
    autres = [a for a in AUTRES if bars_d.get(a)]
    md = build_market(bars_d, actifs + autres, regime="SPY", bpd=1)
    idg = np.clip(np.searchsorted(md.days, ts // D) - 1, 0, len(md.days) - 1).astype(np.int32)
    ih = np.clip(np.searchsorted(cles_h, ts // 3_600_000) - 1, 0, len(cles_h) - 1).astype(np.int32)
    ok = reel & mh.ok[ih] & md.ok[idg, :Ac]
    return Marche5(ts, o, h, l, c, q, ok, X5, atr5, vol5, sig, ih, idg, idg.copy(), fund_i, rates, mh.X, mh.atr.astype(np.float32),
                   mh.vol.astype(np.float32), np.asarray(mh.btc_up), md.X, md.atr.astype(np.float32), md.vol.astype(np.float32),
                   np.asarray(md.btc_up), np.nan_to_num(md.open[:, Ac:], nan=1.0), np.nan_to_num(md.high[:, Ac:], nan=1.0),
                   np.nan_to_num(md.low[:, Ac:], nan=1.0), np.nan_to_num(md.close[:, Ac:], nan=1.0), md.ok[:, Ac:],
                   list(actifs) + autres, Ac, md.days)


def scenario(M: Marche5, premier: int, dernier: int, jours: int, rng: np.random.Generator, bloc: int = 3) -> Marche5:
    """Un futur inventé : des blocs de 3 jours entiers tirés dans le passé déjà vécu (bougies `premier` à `dernier`), recollés."""
    jmin, jmax = premier // BPD, max(premier // BPD + 1, dernier // BPD - bloc)
    n = -(-jours // bloc)
    debuts = rng.integers(jmin, jmax, n) * BPD
    idx = np.concatenate([np.arange(a, a + bloc * BPD) for a in debuts])
    idx = idx[idx < len(M.ts)]
    prev = M.c[np.maximum(idx - 1, 0)].astype(np.float64)

    def rel(a):
        return a[idx].astype(np.float64) / prev
    chemin = np.cumprod(np.vstack([np.ones((1, M.c.shape[1])), rel(M.c)]), axis=0)
    base = chemin[:-1]
    dg = M.idg[idx[::BPD]]
    pd = M.Dc[np.maximum(dg - 1, 0)]

    def reld(a):
        return np.nan_to_num(a[dg] / pd, nan=1.0)
    cd = np.cumprod(np.vstack([np.ones((1, M.Dc.shape[1])), reld(M.Dc)]), axis=0)
    bd = cd[:-1]
    idl = np.repeat(np.arange(len(dg)), BPD)[: len(idx)] + 1
    Do = np.vstack([bd[:1], bd * reld(M.Do)])
    return Marche5(np.arange(len(idx)) * STEP, base * rel(M.o), base * rel(M.h), base * rel(M.l), chemin[1:], M.qv[idx], M.ok[idx],
                   M.X5[idx], M.atr5[idx], M.vol5[idx], M.sig[idx], M.ih[idx], M.idg[idx], idl.astype(np.int32), M.fund_i[idx], M.rates,
                   M.Xh, M.atrh, M.volh, M.up_h, M.Xd, M.atrd, M.vold, M.up_d, Do, np.vstack([bd[:1], bd * reld(M.Dh)]),
                   np.vstack([bd[:1], bd * reld(M.Dl)]), cd, np.vstack([M.Dok[dg[:1]], M.Dok[dg]]), M.assets, M.Ac, M.jours)


# ---------------------------------------------------------------- les cerveaux
@dataclass
class Cerveaux:
    fb: np.ndarray
    wb: np.ndarray
    fr: np.ndarray
    wr: np.ndarray
    regime: np.ndarray
    th_in: np.ndarray
    th_out: np.ndarray
    short: np.ndarray
    stop: np.ndarray
    trail: np.ndarray
    tvol: np.ndarray
    maxpos: np.ndarray
    gross: np.ndarray
    npos: np.ndarray
    hold: np.ndarray
    assets: np.ndarray
    reserve: np.ndarray
    pyr_k: np.ndarray
    pyr_frac: np.ndarray
    ada: np.ndarray
    tf: np.ndarray          # indice dans TFS : à quel rythme il décide
    maker: np.ndarray       # entre avec des ordres limite (0,02 %, mais pas toujours exécutés)

    def __len__(self):
        return len(self.th_in)

    def take(self, i) -> "Cerveaux":
        return Cerveaux(*(np.array(getattr(self, f)[i]) for f in self.__dataclass_fields__))

    @staticmethod
    def concat(bs: list["Cerveaux"]) -> "Cerveaux":
        return Cerveaux(*(np.concatenate([getattr(b, f) for b in bs]) for f in Cerveaux.__dataclass_fields__))


def nettoyer(b: Cerveaux, levier: float) -> Cerveaux:
    b.th_out = np.minimum(b.th_out, b.th_in - 0.01)
    b.stop = np.clip(b.stop, 0.3, 8)
    b.trail = np.clip(b.trail, 0, 8)
    b.tvol = np.clip(b.tvol, 0.02, 5.0)
    b.maxpos = np.clip(b.maxpos, 0.05, 10.0)
    b.gross = np.clip(b.gross, 1.0, levier)
    b.npos = np.clip(b.npos, 1, 9)
    b.hold = np.clip(b.hold, 0, 200)
    b.wb = np.clip(b.wb, -1, 1)
    b.wr = np.clip(b.wr, -1, 1)
    b.reserve = np.clip(b.reserve, 0, 0.9)
    b.pyr_k = np.clip(b.pyr_k, 0, 8)
    b.pyr_frac = np.clip(b.pyr_frac, 0, 1)
    b.ada = np.clip(b.ada, 0, 3)
    b.tf = np.clip(b.tf, 0, len(TFS) - 1)
    b.assets[~b.assets.any(axis=1), 0] = True
    return b


def au_hasard(n: int, A: int, levier: float, rng: np.random.Generator) -> Cerveaux:
    th = rng.uniform(0.0, 0.8, n)
    return nettoyer(Cerveaux(
        fb=rng.integers(0, F_T, (n, K)), wb=rng.uniform(-1, 1, (n, K)) * (rng.random((n, K)) < 0.8),
        fr=rng.integers(0, F_T, (n, K)), wr=rng.uniform(-1, 1, (n, K)) * (rng.random((n, K)) < 0.8),
        regime=rng.random(n) < 0.5, th_in=th, th_out=th - rng.uniform(0.05, 1.0, n), short=rng.random(n) < 0.5,
        stop=rng.uniform(0.5, 8, n), trail=np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.5, 8, n)),
        tvol=np.exp(rng.uniform(math.log(0.05), math.log(3.0), n)), maxpos=np.exp(rng.uniform(math.log(0.1), math.log(10), n)),
        gross=np.exp(rng.uniform(0, math.log(max(levier, 1.0001)), n)), npos=rng.integers(1, 10, n),
        hold=np.where(rng.random(n) < 0.6, 0, rng.integers(2, 120, n)), assets=rng.random((n, A)) < 0.6,
        reserve=np.where(rng.random(n) < 0.4, 0.0, rng.uniform(0, 0.8, n)),
        pyr_k=np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.5, 5, n)), pyr_frac=rng.uniform(0, 1, n),
        ada=np.where(rng.random(n) < 0.4, 0.0, rng.uniform(0, 2, n)), tf=rng.integers(0, len(TFS), n), maker=rng.random(n) < 0.5), levier)


def bot_actuel(assets: list[str]) -> Cerveaux:
    """Le bot actuel sur ce broker : momentum 14/28/56 j, une décision par jour, sans levier."""
    i = [FEATURE_NAMES.index(f"momentum {k} j") for k in (14, 28, 56)] + [0]
    un = lambda v: np.array([v])   # noqa: E731
    return Cerveaux(fb=np.array([i]), wb=np.array([[1.0, 1, 1, 0]]), fr=np.array([i]), wr=np.array([[1.0, 1, 1, 0]]), regime=un(False),
                    th_in=un(0.0), th_out=un(-1e-6), short=un(False), stop=un(3.0), trail=un(0.0), tvol=un(0.15), maxpos=un(0.3),
                    gross=un(1.0), npos=un(3), hold=un(0), assets=np.array([[a in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "NVDA", "TSLA") for a in assets]]),
                    reserve=un(0.0), pyr_k=un(0.0), pyr_frac=un(0.0), ada=un(0.0), tf=un(4), maker=un(False))


def decrire(b: Cerveaux, i: int, assets: list[str]) -> str:
    def sig(f, w):
        return " ; ".join(f"{w[i, k]:+.2f} x {NOMS[f[i, k]]}" for k in range(K) if abs(w[i, k]) > 0.02) or "aucun"
    tete = f"cerveau hausse : {sig(b.fb, b.wb)} | cerveau baisse : {sig(b.fr, b.wr)}" if b.regime[i] else f"signaux : {sig(b.fb, b.wb)}"
    rythme = ["5 min", "15 min", "1 h", "4 h", "1 jour"][int(b.tf[i])]
    return (f"décide toutes les {rythme}, {'ordres limite' if b.maker[i] else 'ordres au marché'} | {tete} | entrer si |score| > {b.th_in[i]:.2f}, "
            f"sortir si < {b.th_out[i]:.2f} | {'achète ET vend à découvert' if b.short[i] else 'achète seulement'} | LEVIER max x{b.gross[i]:.1f}, "
            f"position max {b.maxpos[i] * 100:.0f} %, volatilité visée {b.tvol[i] * 100:.0f} %, {b.npos[i]} position(s) | stop {b.stop[i]:.1f} x ATR"
            f" | réserve {b.reserve[i] * 100:.0f} % | actifs : " + ", ".join(a for a, ok in zip(assets, b.assets[i]) if ok))


# ---------------------------------------------------------------- le broker
class VraiMonde:
    """Tous les comptes, en dollars. Colonnes : les cryptos (mises à jour à chaque bougie), puis les autres actifs (chaque jour)."""

    def __init__(self, M: Marche5, b: Cerveaux):
        self.M, self.b = M, b
        P, A = len(b), M.A
        self.side = np.zeros((P, A), np.int8)
        self.pos = np.zeros((P, A))             # exposition en dollars (toujours positive)
        self.stop_px = np.zeros((P, A))
        self.ext = np.zeros((P, A))
        self.add_px = np.zeros((P, A))
        self.age = np.zeros((P, A), np.int32)
        self.pend = np.zeros((P, A), np.int8)   # ordres en attente de la prochaine ouverture
        self.psz = np.zeros((P, A))
        self.plim = np.zeros((P, A))
        self.patr = np.zeros((P, A))
        self.eq = np.full(P, NAISSANCE)
        self.res = np.zeros(P)                  # réserve : argent mis de côté, jamais risqué
        self.hwm = np.full(P, NAISSANCE)
        self.eq_jour = np.full(P, NAISSANCE)
        self.ema = np.zeros(P)
        self.alive = np.ones(P, bool)
        self.cause = np.zeros(P, np.int8)
        self.birth = np.zeros(P, np.int64)
        self.trades = np.zeros(P, np.int64)
        self.frais = np.zeros(P)                # commissions payées
        self.fin = np.zeros(P)                  # financement et intérêts payés
        self.gliss = np.zeros(P)                # écart et impact payés
        self.vus = [(0, M.Ac), (M.Ac, A)]
        self._gel()
        self.mmr = np.concatenate([MMR_CRYPTO[: M.Ac], np.full(A - M.Ac, 0.25)])
        self.lev = np.concatenate([LEV_CRYPTO[: M.Ac], [20.0 if a in DEVISES else 5.0 if a in MATIERES else 2.0 for a in M.assets[M.Ac:]]])
        self.autres_fx = np.array([a in DEVISES for a in M.assets[M.Ac:]])

    def _gel(self):
        b = self.b
        self.perp = (b.gross > 1.0001) | b.short
        self.tfv = TFS[b.tf]
        self.lay = COUCHE[b.tf]
        self.a_autres = b.assets[:, self.M.Ac:].any(axis=1)

    def renaitre(self, idx: np.ndarray, kids: Cerveaux, k: int):
        for f in Cerveaux.__dataclass_fields__:
            getattr(self.b, f)[idx] = getattr(kids, f)
        for arr in (self.side, self.pos, self.stop_px, self.ext, self.add_px, self.age, self.pend, self.psz, self.plim, self.patr):
            arr[idx] = 0
        for arr in (self.eq, self.hwm, self.eq_jour):
            arr[idx] = NAISSANCE
        for arr in (self.res, self.ema, self.cause, self.trades, self.frais, self.fin, self.gliss):
            arr[idx] = 0
        self.alive[idx] = True
        self.birth[idx] = k
        self._gel()

    # ------------------------------------------------------------ frais
    def _com(self, notionnel: np.ndarray, g: int, d: int, maker=None) -> np.ndarray:
        if g == 0:
            taux = np.where(self.perp, FRAIS_PERP, FRAIS_SPOT)
            if maker is not None:
                taux = np.where(maker & self.perp, FRAIS_MAKER, taux)
            return (notionnel * taux[:, None]).sum(1)
        px = self.M.Dc[d]
        taux = np.where(self.autres_fx, 0.00002, 0.005 / np.maximum(px, 0.5))
        mini = np.where(self.autres_fx, 2.0, 1.0)
        f = np.clip(notionnel * taux, mini, 0.01 * notionnel)
        return np.where(notionnel > 0, f, 0.0).sum(1)

    # ------------------------------------------------------------ une bougie sur un groupe d'actifs
    def _regler(self, g, o, h, l, c, pc, hs, glis, sig, q, d, fi=None, ouvert=None):  # noqa: E741
        a0, a1 = self.vus[g]
        side, pos, stop, ext, pend = (x[:, a0:a1] for x in (self.side, self.pos, self.stop_px, self.ext, self.pend))
        ref = np.broadcast_to(pc, pos.shape)
        if pend.any():
            psz, plim = self.psz[:, a0:a1], self.plim[:, a0:a1]
            mk = self.b.maker[:, None] if g == 0 else np.zeros((len(pos), 1), bool)
            impact = sig * np.sqrt(psz / np.maximum(q, 1.0))
            px = o * (1 + pend * (hs + impact))
            px = np.where(mk, plim, px)
            prix_ok = np.where(pend > 0, l < plim, h > plim)
            fait = (pend != 0) & (~mk | prix_ok)
            if ouvert is not None:                  # marché fermé : l'ordre attend la vraie ouverture
                fait = fait & ouvert
            stop_mult = self.b.stop[:, None] * self.patr[:, a0:a1]
            n = psz * fait
            cout = (n * np.where(mk, 0.0, hs + impact)).sum(1)
            com = self._com(n, g, d, maker=self.b.maker)
            self.eq -= cout + com
            self.gliss += cout
            self.frais += com
            side[:] = np.where(fait, pend, side)
            pos[:] = np.where(fait, psz, pos)
            ini = pend * stop_mult
            stop[:] = np.where(fait, px * (1 - ini), stop)
            ext[:] = np.where(fait, px, ext)
            self.add_px[:, a0:a1] = np.where(fait, px, self.add_px[:, a0:a1])
            self.age[:, a0:a1] = np.where(fait, 0, self.age[:, a0:a1])
            ref = np.where(fait, px, ref)
            self.trades += fait.sum(1)
            pend[:] = 0 if ouvert is None else np.where(ouvert, 0, pend)
        if not pos.any():
            return
        lng, sht = side > 0, side < 0
        hit_l, hit_s = lng & (l <= stop), sht & (h >= stop)
        hit = hit_l | hit_s
        sortie = np.where(hit_l, np.minimum(o, stop) * (1 - glis), np.maximum(o, stop) * (1 + glis))
        r = np.where(hit, sortie / ref - 1, c / ref - 1) * side
        pire = np.where(hit, r, np.where(lng, l / ref - 1, np.where(sht, -(h / ref - 1), 0.0)))
        liq = self.alive & self.perp & ((self.eq + (pos * pire).sum(1)) <= (pos * self.mmr[a0:a1]).sum(1)) & (self.eq > 0) & pos.any(axis=1)
        self.eq += (pos * r).sum(1)
        if hit.any():
            n = pos * (1 + r) * hit
            com = self._com(n, g, d)
            self.eq -= com
            self.frais += com
            self.trades += hit.sum(1)
        garde = (side != 0) & ~hit
        pos[:] = np.where(garde, pos * (1 + r), 0.0)
        side[:] = np.where(garde, side, 0)
        ext[:] = np.where(side > 0, np.maximum(ext, c), np.where(side < 0, np.minimum(ext, c), 0.0))
        self.age[:, a0:a1] += garde
        if fi is not None and fi >= 0 and g == 0:
            du = ((pos * side) * self.M.rates[fi]).sum(1) * self.perp
            self.eq -= du
            self.fin += du
        if liq.any():                               # le compte de marge est perdu, la réserve reste
            self.cause[liq] = 2
            self.eq[liq] = self.res[liq]
            pos[liq] = 0.0
            side[liq] = 0

    def _fin_de_bougie(self):
        gain = np.maximum(self.eq - self.hwm, 0)
        self.res = np.minimum(self.res + self.b.reserve * gain, np.maximum(self.eq, 0))
        self.hwm = np.maximum(self.hwm, self.eq)
        mort = self.alive & (self.eq <= NAISSANCE * RUIN)
        if mort.any():
            self.cause[mort & (self.cause == 0)] = 1
            self.alive[mort] = False
            self.side[mort] = 0
            self.pos[mort] = 0
            self.pend[mort] = 0

    # ------------------------------------------------------------ décisions
    def _decider(self, g, idx, x, atr, vol, c, hs, ok, baisse, sig, q, d):
        if not len(idx):
            return
        a0, a1 = self.vus[g]
        b = self.b
        fb, wb = b.fb[idx], b.wb[idx]
        score = (x[:, fb] * wb[None]).sum(2).T
        rg = np.flatnonzero(b.regime[idx])
        if baisse and len(rg):
            score[rg] = (x[:, b.fr[idx[rg]]] * b.wr[idx[rg]][None]).sum(2).T
        side = self.side[idx, a0:a1]
        pos = self.pos[idx, a0:a1]
        th_in, th_out = b.th_in[idx, None], b.th_out[idx, None]
        old = (b.hold[idx, None] > 0) & (self.age[idx, a0:a1] >= b.hold[idx, None] * self.tfv[idx, None])
        out = ((side > 0) & ((score < th_out) | old)) | ((side < 0) & ((score > -th_out) | old))
        if out.any():
            n = np.zeros_like(self.pos[:, a0:a1])
            n[idx] = pos * out
            cout = (n * hs).sum(1)
            com = self._com(n, g, d)
            self.eq -= cout + com
            self.gliss += cout
            self.frais += com
            self.trades += n.astype(bool).sum(1)
            side = np.where(out, 0, side)
            pos = np.where(out, 0.0, pos)
            self.side[idx, a0:a1] = side
            self.pos[idx, a0:a1] = pos
        if (b.trail[idx] > 0).any():
            tr = b.trail[idx, None]
            stop = self.stop_px[idx, a0:a1]
            ext = self.ext[idx, a0:a1]
            stop = np.where((side > 0) & (tr > 0), np.maximum(stop, ext * (1 - tr * atr)), stop)
            stop = np.where((side < 0) & (tr > 0), np.minimum(stop, ext * (1 + tr * atr)), stop)
            self.stop_px[idx, a0:a1] = stop
        libre_m = (side == 0) & ok[None, :] & b.assets[idx, a0:a1] & (self.pend[idx, a0:a1] == 0)
        want_l = libre_m & (score > th_in)
        want_s = libre_m & b.short[idx, None] & (score < -th_in)
        eq_risque = np.maximum(self.eq[idx] - self.res[idx], 0)
        expo = self.pos[idx].sum(1)
        place = np.maximum(b.gross[idx] * eq_risque - expo, 0)
        if (b.pyr_k[idx] > 0).any():
            tenu = side != 0
            prog = np.where(tenu, side * (c / np.maximum(self.add_px[idx, a0:a1], 1e-12) - 1), 0)
            go = tenu & (b.pyr_k[idx, None] > 0) & (prog >= b.pyr_k[idx, None] * atr)
            if go.any():
                eqr = eq_risque[:, None]
                extra = np.where(go, np.minimum(b.pyr_frac[idx, None] * pos, np.maximum(b.maxpos[idx, None] * eqr - pos, 0)), 0.0)
                extra *= np.minimum(1, place / np.maximum(extra.sum(1), 1e-12))[:, None]
                n = np.zeros_like(self.pos[:, a0:a1])
                n[idx] = extra
                cout = (n * hs).sum(1)
                com = self._com(n, g, d)
                self.eq -= cout + com
                self.gliss += cout
                self.frais += com
                self.pos[idx, a0:a1] = pos + extra
                self.add_px[idx, a0:a1] = np.where(extra > 0, c, self.add_px[idx, a0:a1])
                place = np.maximum(place - extra.sum(1), 0)
        cand = want_l | want_s
        if not cand.any():
            return
        libre = b.npos[idx] - (self.side[idx] != 0).sum(1)
        force = np.where(cand, np.abs(score), -np.inf)
        rang = np.argsort(np.argsort(-force, axis=1), axis=1)
        new = cand & (rang < libre[:, None])
        mult = np.clip(np.exp(b.ada[idx] * self.ema[idx] * 20), 0.25, 3.0)
        taille = np.minimum(b.tvol[idx, None] / np.maximum(vol, 0.05) * mult[:, None], b.maxpos[idx, None])
        taille = np.minimum(taille, self.lev[a0:a1]) * eq_risque[:, None] * new
        taille = np.minimum(taille, PLAFOND_VOLUME * q)
        taille *= np.minimum(1, place / np.maximum(taille.sum(1), 1e-12))[:, None]
        new &= taille >= ORDRE_MIN
        self.pend[idx, a0:a1] = np.where(new, np.where(want_l, 1, -1), self.pend[idx, a0:a1])
        self.psz[idx, a0:a1] = np.where(new, taille, self.psz[idx, a0:a1])
        self.plim[idx, a0:a1] = np.where(new, c, self.plim[idx, a0:a1])
        self.patr[idx, a0:a1] = np.where(new, atr, self.patr[idx, a0:a1])

    def bougie(self, k: int):
        M = self.M
        premier = k == 0
        o, h, l, c = M.o[k].astype(np.float64), M.h[k].astype(np.float64), M.l[k].astype(np.float64), M.c[k].astype(np.float64)  # noqa: E741
        sig = M.sig[k].astype(np.float64)
        q = M.qv[k].astype(np.float64)
        hs = DEMI_ECART[: M.Ac] + 0.03 * sig
        if not premier:
            self._regler(0, o, h, l, c, M.c[k - 1].astype(np.float64), hs, hs + 0.5 * sig, sig, q, 0, int(M.fund_i[k]))
        nouveau_jour = k % BPD == 0
        if nouveau_jour:
            self._jour(k, premier)
        self._fin_de_bougie()
        idx = np.flatnonzero(self.alive & (k % self.tfv == 0))
        if len(idx):
            ih, id_ = M.ih[k], M.idg[k]
            x = np.concatenate([M.Xd[id_, : M.Ac], M.Xh[ih], M.X5[k]], axis=1)
            couche = np.stack([M.atr5[k], M.atrh[ih], M.atrd[id_, : M.Ac]])[self.lay[idx]]
            volc = np.stack([M.vol5[k], M.volh[ih], M.vold[id_, : M.Ac]])[self.lay[idx]]
            self._decider(0, idx, x, couche, volc, c, hs, M.ok[k], not bool(M.up_h[ih]), sig, q, 0)

    def _jour(self, k: int, premier: bool):
        M = self.M
        if M.A == M.Ac:
            return
        d, id_ = int(M.idl[k]), int(M.idg[k])
        hs = np.where(self.autres_fx, 0.00001, 0.0002)
        if not premier and d > 0:
            nul = np.full(M.A - M.Ac, 0.0)
            grand = np.full(M.A - M.Ac, 1e12)
            self._regler(1, M.Do[d], M.Dh[d], M.Dl[d], M.Dc[d], M.Dc[d - 1], hs, hs + 0.0003, nul, grand, d, ouvert=np.asarray(M.Dok[d], bool))
            emprunt = np.maximum(self.pos[:, M.Ac:].sum(1) - np.maximum(self.eq, 0), 0) * INTERET_AN / 365
            titres = (self.pos[:, M.Ac:] * (self.side[:, M.Ac:] < 0)).sum(1) * EMPRUNT_TITRES_AN / 365
            self.eq -= emprunt + titres
            self.fin += emprunt + titres
        self.ema = 0.94 * self.ema + 0.06 * np.log(np.maximum(self.eq, 1.0) / np.maximum(self.eq_jour, 1.0))
        self.eq_jour = self.eq.copy()
        idx = np.flatnonzero(self.alive & self.a_autres)
        if len(idx):
            x = np.concatenate([M.Xd[id_, M.Ac:], np.zeros((M.A - M.Ac, F_T - F_D), np.float32)], axis=1)
            ok = np.asarray(M.Dok[d], bool)
            self._decider(1, idx, x, np.broadcast_to(M.atrd[id_, M.Ac:], (len(idx), M.A - M.Ac)),
                          np.broadcast_to(M.vold[id_, M.Ac:], (len(idx), M.A - M.Ac)), M.Dc[d], hs, ok, not bool(M.up_d[id_]),
                          np.zeros(M.A - M.Ac), np.full(M.A - M.Ac, 1e12), d)


def jouer(M: Marche5, b: Cerveaux, k0: int = 0, k1: int | None = None) -> VraiMonde:
    """Fait vivre des cerveaux fixes sur le marché, sans évolution."""
    if k0:
        raise ValueError("utiliser une vue du marché pour partir d'une autre date")
    w = VraiMonde(M, b)
    for k in range(k1 if k1 is not None else len(M.ts)):
        w.bougie(k)
    return w


# ---------------------------------------------------------------- l'écosystème
@dataclass
class Reglage:
    pop: int = 8000
    levier: float = 25.0
    seed: int = 1
    heritage: bool = True           # part des meilleurs cerveaux du passé (ils ont vu ces années : résultat flatté)
    objectif: str = "x1000"
    cand: int = 100
    scenarios: int = 8
    jours_scenario: int = 15
    periode: int = 30
    memoire: int = 90
    tue: float = 0.3
    fonds_k: int = 20
    budget_s: float = 19_000.0      # s'arrête proprement avant la limite de 6 h des machines

    def nom(self) -> str:
        return (f"{'héritiers' if self.heritage else 'nés de zéro'} | levier x{self.levier:g} | {self.objectif} | tirage {self.seed} | "
                f"{self.pop} cerveaux")


def note_scenarios(M: Marche5, cand: Cerveaux, premier: int, dernier: int, s: Reglage, rng: np.random.Generator) -> np.ndarray:
    logs = []
    for _ in range(s.scenarios):
        ms = scenario(M, premier, dernier, s.jours_scenario, rng)
        w = VraiMonde(ms, cand.take(np.arange(len(cand))))
        for k in range(len(ms.ts)):
            w.bougie(k)
        logs.append(np.log(np.maximum(np.where(w.alive, w.eq, np.minimum(w.eq, NAISSANCE * RUIN)), 10.0) / NAISSANCE))
    logs = np.array(logs)
    if s.objectif == "x1000":
        cible = math.log(1000) * s.jours_scenario / 365
        return (logs >= cible).mean(axis=0) + 0.1 * np.median(logs, axis=0)
    return 0.5 * np.median(logs, axis=0) + 0.5 * np.quantile(logs, 0.05, axis=0)


def croiser(b: Cerveaux, parents: np.ndarray, fit: np.ndarray, n: int, levier: float, rng: np.random.Generator) -> Cerveaux:
    def tour():
        c = parents[rng.integers(0, len(parents), (n, 4))]
        return c[np.arange(n), np.argmax(fit[c], axis=1)]
    pa, pb = b.take(tour()), b.take(tour())
    ch = b.take(tour())
    for f in Cerveaux.__dataclass_fields__:
        va, vb = getattr(pa, f), getattr(pb, f)
        setattr(ch, f, np.where(rng.random(va.shape) < 0.5, va, vb))
    return muter(ch, levier, rng)


def muter(ch: Cerveaux, levier: float, rng: np.random.Generator) -> Cerveaux:
    n = len(ch)
    mut = lambda sh, s: rng.normal(0, s, sh) * (rng.random(sh) < 0.25)  # noqa: E731
    for f in ("wb", "wr"):
        setattr(ch, f, getattr(ch, f) + mut(getattr(ch, f).shape, 0.3))
    for f in ("fb", "fr"):
        v = getattr(ch, f)
        setattr(ch, f, np.where(rng.random(v.shape) < 0.08, rng.integers(0, F_T, v.shape), v))
    ch.th_in = ch.th_in + mut(n, 0.1)
    ch.th_out = ch.th_out + mut(n, 0.1)
    ch.stop = ch.stop + mut(n, 0.5)
    ch.trail = np.where(rng.random(n) < 0.05, 0, ch.trail + mut(n, 0.5))
    for f in ("tvol", "maxpos", "gross"):
        setattr(ch, f, getattr(ch, f) * np.exp(mut(n, 0.25)))
    ch.npos = ch.npos + np.round(mut(n, 0.8)).astype(int)
    ch.hold = np.where(rng.random(n) < 0.05, 0, ch.hold + np.round(mut(n, 10)).astype(int))
    for f in ("regime", "short", "maker"):
        setattr(ch, f, np.where(rng.random(n) < 0.05, ~getattr(ch, f), getattr(ch, f)))
    ch.assets = np.where(rng.random(ch.assets.shape) < 0.05, ~ch.assets, ch.assets)
    ch.reserve = ch.reserve + mut(n, 0.1)
    ch.pyr_k = np.where(rng.random(n) < 0.05, 0, ch.pyr_k + mut(n, 0.5))
    ch.pyr_frac = ch.pyr_frac + mut(n, 0.15)
    ch.ada = np.where(rng.random(n) < 0.05, 0, ch.ada + mut(n, 0.3))
    ch.tf = np.where(rng.random(n) < 0.08, rng.integers(0, len(TFS), n), ch.tf)
    return nettoyer(ch, levier)


def vivre(M: Marche5, s: Reglage, graines: Cerveaux | None = None, sortie: str | None = None, log=None) -> dict:
    """L'écosystème vit sur tout le marché. Le fonds suit les meilleurs du moment, sans jamais voir l'avenir."""
    debut_h = time.time()
    rng = np.random.default_rng(s.seed)
    A, T = M.A, len(M.ts)
    pop = au_hasard(s.pop, A, s.levier, rng)
    if graines is not None and len(graines):
        n = min(len(graines), s.pop // 10)
        pop_idx = rng.permutation(s.pop)
        ordre = rng.permutation(len(graines))[:n]
        g = nettoyer(graines.take(ordre), s.levier)
        for f in Cerveaux.__dataclass_fields__:
            getattr(pop, f)[pop_idx[:n]] = getattr(g, f)
        m = min(s.pop // 2 - n, s.pop - n)
        kids = muter(g.take(rng.integers(0, n, m)), s.levier, rng)
        for f in Cerveaux.__dataclass_fields__:
            getattr(pop, f)[pop_idx[n:n + m]] = getattr(kids, f)
    w = VraiMonde(M, pop)
    periode, an = s.periode * BPD, 365 * BPD
    regard = max(1, s.memoire // s.periode)
    fonds, courbe = 1000.0, [(int(M.ts[0] // D), 1000.0)]
    choisis, eq_ch = np.array([], int), np.array([])
    captures: list[np.ndarray] = []
    naissances: list[np.ndarray] = []
    meilleur, atteint, nes, stats = (0.0, None), 0, s.pop, []
    archive: Cerveaux | None = None
    interrompu = None
    tr_fonds, cout_fonds, jours_fonds = 0.0, 0.0, 0.0
    tr0, cout0 = np.array([]), np.array([])

    def resultat() -> dict:
        haut = np.argsort(-np.where(w.alive, w.eq, 0))[:3]
        res = {"reglage": s.__dict__, "courbe": courbe, "nes": nes, "meilleur_1an": meilleur[0], "meilleur_1an_cerveau": meilleur[1],
               "x1000": atteint, "chefs": [decrire(w.b, int(i), M.assets) for i in haut], "stats": stats, "interrompu": interrompu,
               "annees": {int(y): v for y, v in yearly(courbe).items()}, "trades_jour": tr_fonds / max(jours_fonds, 1),
               "frais_part": cout_fonds / max(jours_fonds, 1)}
        if sortie:
            Path(sortie).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return res

    for k in range(T):
        w.bougie(k)
        if k % (7 * BPD) == 0:
            age = k - w.birth
            fini = (age >= an) & (age < an + 7 * BPD)
            if fini.any():
                mult = np.where(w.alive[fini], (w.eq[fini]) / NAISSANCE, 0)
                atteint += int((mult >= 1000).sum())
                j = int(np.argmax(mult))
                if mult[j] > meilleur[0]:
                    meilleur = (float(mult[j]), decrire(w.b, int(np.flatnonzero(fini)[j]), M.assets))
        if k == 0 or k % periode != 0:
            continue
        if len(choisis):
            r = np.where(w.birth[choisis] < k - periode + 1, w.eq[choisis] / np.maximum(eq_ch, 1e-9), 1.0)
            fonds *= max(float(np.mean(r)) - 0.002, 0)
            courbe.append((int(M.ts[k] // D), fonds))
            tr_fonds += float(np.mean(np.maximum(w.trades[choisis] - tr0, 0)))
            cout_fonds += float(np.mean(np.maximum(w.frais[choisis] + w.gliss[choisis] + w.fin[choisis] - cout0, 0) / np.maximum(eq_ch, 1e-9)))
            jours_fonds += s.periode
        captures.append(w.eq.copy())
        naissances.append(w.birth.copy())
        del captures[:-(regard + 1)], naissances[:-(regard + 1)]
        i0 = max(0, len(captures) - 1 - regard)
        ref = np.where(naissances[i0] == w.birth, captures[i0], NAISSANCE)
        perf = np.log(np.maximum(w.eq, 1.0) / np.maximum(ref, 1.0))
        assez = (k - w.birth) >= min(s.memoire * BPD, periode)
        fit = np.where(w.alive, perf, -99.0)
        if w.alive.any():
            v = w.alive
            stats.append({"jour": int(M.ts[k] // D), "ruines": float(1 - v.mean()), "mediane": float(np.median(perf[v])),
                          "levier": float(np.median(w.b.gross[v])), "tf5": float((w.b.tf[v] == 0).mean()), "tf1j": float((w.b.tf[v] == 4).mean()),
                          "limite": float(w.b.maker[v].mean()), "trades_jour": float(np.median(w.trades[v] / np.maximum((k - w.birth[v]) / BPD, 1)))})
        sc = None
        vivier = np.flatnonzero(w.alive & assez & (w.birth < k))
        if len(vivier):
            cand = vivier[np.argsort(-fit[vivier])[: s.cand]]
            sc = (cand, note_scenarios(M, w.b.take(cand), 0, k, s, rng))
            top = cand[np.argsort(-sc[1])[:5]]
            archive = w.b.take(top) if archive is None else Cerveaux.concat([archive, w.b.take(top)])
            if len(archive) > 1000:
                archive = archive.take(rng.permutation(len(archive))[:1000])
        morts = ~w.alive
        if sc is not None:
            fit = fit.copy()
            fit[sc[0]] = fit.max() + 10 + sc[1]
        morts[np.argsort(fit)[: int(s.tue * s.pop)]] = True
        vivants = np.flatnonzero(~morts)
        idx = np.flatnonzero(morts)
        if len(idx) and len(vivants):
            n_imm = int(len(idx) * 0.15)
            parts = [croiser(w.b, vivants, fit, len(idx) - n_imm, s.levier, rng)]
            if archive is not None and n_imm // 2:                       # la mémoire : les anciens reviennent
                parts.append(muter(archive.take(rng.integers(0, len(archive), n_imm // 2)), s.levier, rng))
                n_imm -= n_imm // 2
            parts.append(au_hasard(n_imm, A, s.levier, rng))
            w.renaitre(idx, Cerveaux.concat(parts), k)
            for c in captures:
                c[idx] = NAISSANCE
            nes += len(idx)
        elig = np.flatnonzero(w.alive & assez & (w.birth < k))
        if sc is not None:
            elig = sc[0][w.alive[sc[0]] & (w.birth[sc[0]] < k)]
        if len(elig):
            choisis = elig[np.argsort(-fit[elig])[: s.fonds_k]]
            eq_ch = w.eq[choisis].copy()
            tr0, cout0 = w.trades[choisis].copy(), (w.frais + w.gliss + w.fin)[choisis].copy()
        else:
            choisis = np.array([], int)
        resultat()
        if log:
            log(f"    jour {k // BPD} : fonds {fonds:,.0f} | vivants {int(w.alive.sum())} | nés {nes:,} | {time.time() - debut_h:.0f} s".replace(",", " "))
        if time.time() - debut_h > s.budget_s:
            interrompu = int(M.ts[k] // D)
            break
    res = resultat()
    if sortie:
        haut = np.argsort(-np.where(w.alive, w.eq, 0))[:50]
        g = w.b.take(haut) if archive is None else Cerveaux.concat([w.b.take(haut), archive])
        np.savez_compressed(Path(sortie).with_suffix(".npz"), **{f: getattr(g, f) for f in Cerveaux.__dataclass_fields__})
    return res
