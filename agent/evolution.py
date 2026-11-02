"""Laboratoire d'évolution : des milliers de « cerveaux » (stratégies) qui vivent, meurent et se reproduisent.

Demande de Dhia : objectif 1 000 -> 1 000 000 en un an (x1000). Les cerveaux qui survivent et avancent vers
l'objectif se reproduisent ; leurs enfants héritent d'un mélange de leurs règles, avec des mutations ; à chaque
génération, des stratégies totalement nouvelles apparaissent. Algorithme génétique (Holland 1975 ; appliqué au
trading par Allen & Karjalainen 1999, qui mettent en garde contre le surapprentissage).

LE CERVEAU (génome) :
  - signaux : jusqu'à 4 parmi 38 (momentum 3-112 j, écart à la moyenne, position dans le canal, RSI 2 et 14,
    agitation de l'actif, tendance et agitation du Bitcoin), chacun avec un poids de -1 à +1
  - seuils : entrer si score > seuil_entrée, sortir si score < seuil_sortie
  - risque : stop (1-6 x ATR), stop suiveur, volatilité visée (5-80 %), taille max (10-100 %), 1-5 positions,
    durée max, feu vert du Bitcoin, et les actifs qu'il accepte de trader (parmi 9 cryptos)
LA VIE : année par année, 1 000 USDT au départ de chaque année, frais réels (0,1 % + 0,05 % de glissement par
ordre), spot sans levier. Arrêt d'urgence (-25 % du départ ou -35 % du plus haut) = MORT.
LA NOTE (fitness) : moyenne sur les années d'apprentissage du log(capital final / 1 000) ; une mort coûte cher.
L'HONNÊTETÉ : apprentissage sur des années passées, puis test sur l'année suivante, jamais vue (walk-forward).

Simplification assumée (pour simuler 80 000 cerveaux par minute) : les décisions se prennent à la clôture
journalière, comme le vrai bot, mais la simulation est vectorisée (numpy) ; un champion prometteur doit
ensuite être revérifié avec le vrai moteur avant toute utilisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

D = 86_400_000
LOOKBACKS = [3, 5, 7, 10, 14, 20, 28, 40, 56, 80, 112]
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"]
K = 4                       # signaux par cerveau
LOOKBACKS_INTRADAY = {6: [1, 2, 3, 6, 12, 24, 42, 84, 168, 336, 504],          # bougies de 4 h (4 h -> 84 j)
                      24: [2, 4, 8, 16, 24, 48, 96, 168, 336, 672, 1344]}     # bougies de 1 h (2 h -> 56 j)
COST = 0.0015               # par ordre : frais 0,1 % + glissement 0,05 %
FEATURE_NAMES = ([f"momentum {k} j" for k in LOOKBACKS] + [f"écart à la moyenne {k} j" for k in LOOKBACKS]
                 + [f"position dans le canal {k} j" for k in LOOKBACKS]
                 + ["RSI 2", "RSI 14", "agitation de l'actif", "tendance du marché (BTC ou S&P 500)",
                    "agitation du marché (BTC ou S&P 500)", "force relative face au marché (28 j)",
                    "distance au plus haut de l'année", "pic de volume (20 j)"]
                 # sens « intermarchés » : les relations entre les actifs (Murphy, Intermarket Analysis)
                 + ["largeur du marché (part des actifs en hausse)", "variation de la largeur (10 j)",
                    "corrélation moyenne entre actifs (30 j)", "recul du marché depuis son plus haut",
                    "dernier mouvement du marché (BTC ou S&P 500)", "tendance de l'or (28 j)",
                    "tendance des obligations (28 j)", "tendance du dollar (28 j)",
                    "lien de l'actif avec le marché (corrélation 30 j)"])
N_INTER = 9                 # les 9 derniers signaux sont les sens intermarchés
F = len(FEATURE_NAMES)


# ---------------------------------------------------------------- données et signaux (calculés une fois)
@dataclass
class Market:
    days: np.ndarray        # jour (entier) de chaque ligne
    open: np.ndarray        # (T, A)
    low: np.ndarray
    close: np.ndarray
    ok: np.ndarray          # (T, A) actif négociable ce jour-là (historique suffisant)
    X: np.ndarray           # (T, A, F) signaux connus à la clôture
    vol: np.ndarray         # (T, A) volatilité annualisée 30 j
    atr: np.ndarray         # (T, A) ATR 20 j en fraction du prix
    btc_up: np.ndarray      # (T,) tendance du Bitcoin (momentum 14/28/56 majoritaire)
    assets: list
    high: np.ndarray | None = None   # (T, A) plus haut du jour (ventes à découvert : stops et liquidations)
    names: list | None = None        # noms des signaux (en heures pour les bougies de moins d'un jour)
    bpd: int = 1                     # bougies par jour (1 = journalier, 6 = 4 h, 24 = 1 h)


def _rolling_mean(x, n):
    c = np.cumsum(np.nan_to_num(x), axis=0)
    out = np.full_like(x, np.nan)
    out[n - 1:] = (c[n - 1:] - np.vstack([np.zeros((1,) + x.shape[1:]), c[:-n]])) / n
    return out


def _rolling_std(x, n):
    m = _rolling_mean(x, n)
    m2 = _rolling_mean(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def _rolling_ext(x, n, fn):
    out = np.full_like(x, np.nan)
    if len(x) >= n:
        win = np.lib.stride_tricks.sliding_window_view(x, n, axis=0)
        out[n - 1:] = fn(win, axis=-1)
    return out


def _rsi(c, n):
    d = np.diff(c, axis=0, prepend=c[:1])
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au, ad = np.zeros_like(c), np.zeros_like(c)
    a = 1 / n
    for t in range(1, len(c)):
        au[t] = au[t - 1] * (1 - a) + np.nan_to_num(up[t]) * a
        ad[t] = ad[t - 1] * (1 - a) + np.nan_to_num(dn[t]) * a
    return 100 - 100 / (1 + au / np.maximum(ad, 1e-12))


def build_market(bars: dict, assets: list[str] = ASSETS, regime: str = "BTCUSDT", bpd: int = 1,
                 inter: bool = True) -> Market:
    """regime : actif qui donne la tendance générale du marché (Bitcoin pour la crypto, SPY pour tout le marché).
    bpd : bougies par jour (1 journalier, 6 = 4 h, 24 = 1 h). Les fenêtres « en jours » sont converties en bougies."""
    assets = [a for a in assets if bars.get(a)]
    step = D // bpd
    keys = sorted({b.ts // step for a in assets for b in bars[a]})
    idx = {k: i for i, k in enumerate(keys)}
    days = [k * step // D for k in keys]
    lookbacks = LOOKBACKS if bpd == 1 else LOOKBACKS_INTRADAY[bpd]
    T, A = len(keys), len(assets)
    O, H, L, C, V = (np.full((T, A), np.nan) for _ in range(5))
    for j, a in enumerate(assets):
        for b in bars[a]:
            i = idx[b.ts // step]
            O[i, j], H[i, j], L[i, j], C[i, j], V[i, j] = b.open, b.high, b.low, b.close, b.volume
    real = ~np.isnan(C)                                   # jour de cotation réel (pas de trading le week-end en bourse)
    # jours manquants au milieu : on répète la veille (pas de trou dans les calculs)
    for t in range(1, T):
        miss = np.isnan(C[t]) & ~np.isnan(C[t - 1])
        for arr in (O, H, L, C):
            arr[t, miss] = C[t - 1, miss]
    r = np.diff(np.log(C), axis=0, prepend=np.log(C[:1]))
    vol20 = _rolling_std(r, 20 * bpd)
    vol30 = _rolling_std(r, 30 * bpd) * math.sqrt(365 * bpd)
    vol100 = _rolling_std(r, 100 * bpd)
    feats = []
    for k in lookbacks:                                   # momentum normalisé par la volatilité
        past = np.vstack([np.full((k, A), np.nan), C[:-k]])
        feats.append(np.log(C / past) / (vol20 * math.sqrt(k) + 1e-9) / 3)
    for k in lookbacks:                                   # écart à la moyenne
        feats.append((C / _rolling_mean(C, k) - 1) / (vol20 * math.sqrt(k) + 1e-9) / 3)
    for k in lookbacks:                                   # position dans le canal des k bougies précédentes (-1 bas, +1 haut)
        hh = np.vstack([np.full((1, A), np.nan), _rolling_ext(H, k, np.max)[:-1]])
        ll = np.vstack([np.full((1, A), np.nan), _rolling_ext(L, k, np.min)[:-1]])
        feats.append(2 * (C - ll) / np.maximum(hh - ll, 1e-12) - 1)
    feats.append((_rsi(C, 2) - 50) / 50)
    feats.append((_rsi(C, 14) - 50) / 50)
    feats.append(np.clip(vol20 / (vol100 + 1e-12) - 1, -1, 2))
    b = assets.index(regime) if regime in assets else 0
    btc_mom = sum(np.sign(np.log(C[:, b] / np.concatenate([np.full(k * bpd, np.nan), C[:-k * bpd, b]])))
                  for k in (14, 28, 56))
    btc_up = btc_mom > 0
    feats.append(np.repeat((btc_mom / 3)[:, None], A, axis=1))
    feats.append(np.repeat(np.clip(vol20[:, b] / (vol100[:, b] + 1e-12) - 1, -1, 2)[:, None], A, axis=1))
    past28 = np.vstack([np.full((28 * bpd, A), np.nan), C[:-28 * bpd]])
    mom28 = np.log(C / past28)
    feats.append((mom28 - mom28[:, b:b + 1]) / (vol20 * math.sqrt(28 * bpd) + 1e-9) / 3)   # force relative
    hi252 = _rolling_ext(np.nan_to_num(H, nan=-np.inf), 252 * bpd, np.max)
    feats.append(np.log(C / hi252) / (vol20 * math.sqrt(252 * bpd) + 1e-9))               # distance au plus haut de l'année
    vv = np.nan_to_num(V, nan=0.0)
    feats.append(np.clip(np.log((vv + 1) / (_rolling_mean(vv, 20) + 1)), -2, 2))      # pic de volume
    feats += _intermarket(C, H, r, vol20, assets, b, bpd, real) if inter else [np.zeros((T, A))] * N_INTER
    X = np.clip(np.nan_to_num(np.stack(feats, axis=2), nan=0.0), -3, 3)
    tr = np.maximum(H - L, np.maximum(abs(H - np.vstack([C[:1], C[:-1]])), abs(L - np.vstack([C[:1], C[:-1]]))))
    atr = _rolling_mean(tr, 20) / C
    first = np.argmax(~np.isnan(C), axis=0)
    ok = np.zeros((T, A), bool)
    for j in range(A):
        ok[first[j] + 130 * bpd:, j] = True               # 112 j de signaux + marge
    ok &= real
    names = FEATURE_NAMES if bpd == 1 else _names_intraday(lookbacks, bpd)
    return Market(np.array(days), O, L, C, ok, X.astype(np.float32), np.nan_to_num(vol30, nan=1.0),
                  np.nan_to_num(atr, nan=0.05), btc_up, assets, H, names, bpd)


def _intermarket(C, H, r, vol20, assets, b, bpd, real) -> list[np.ndarray]:
    """Les relations entre les marchés, connues à la clôture (jamais l'avenir)."""
    T, A = C.shape
    bc = lambda v: np.repeat(np.nan_to_num(v)[:, None], A, axis=1)   # noqa: E731  (même valeur pour tous les actifs)
    rr = np.where(real, r, np.nan)
    sma50 = _rolling_mean(C, 50 * bpd)
    up = np.where(np.isnan(rr) | np.isnan(sma50), np.nan, (C > sma50).astype(float))
    have = np.maximum((~np.isnan(up)).sum(1), 1)
    breadth = np.nansum(up, axis=1) / have * 2 - 1                         # -1 tout baisse ... +1 tout monte
    lag = 10 * bpd
    breadth_chg = breadth - np.concatenate([np.full(lag, np.nan), breadth[:-lag]])
    w = 30 * bpd
    sd = _rolling_std(np.nan_to_num(rr), w) + 1e-12
    z = np.nan_to_num(rr) / sd
    n = np.maximum((~np.isnan(rr)).sum(1), 2)
    s_all = z.sum(1)
    avg_corr = (_rolling_mean((s_all ** 2)[:, None], w)[:, 0] - _rolling_mean((z ** 2).sum(1)[:, None], w)[:, 0]) / (n * (n - 1))
    hi = _rolling_ext(np.nan_to_num(H[:, b:b + 1], nan=-np.inf), 252 * bpd, np.max)[:, 0]
    dd_mkt = np.log(C[:, b] / hi) / (vol20[:, b] * math.sqrt(252 * bpd) + 1e-9)
    last_mkt = np.nan_to_num(r[:, b]) / (vol20[:, b] + 1e-9) / 3

    def trend(name, sign=1.0):
        if name not in assets:
            return np.zeros(T)
        j = assets.index(name)
        past = np.concatenate([np.full(28 * bpd, np.nan), C[:-28 * bpd, j]])
        return sign * np.log(C[:, j] / past) / (vol20[:, j] * math.sqrt(28 * bpd) + 1e-9) / 3
    rb = np.nan_to_num(r[:, b:b + 1])
    cov = _rolling_mean(np.nan_to_num(r) * rb, w) - _rolling_mean(np.nan_to_num(r), w) * _rolling_mean(rb, w)
    corr_mkt = cov / (_rolling_std(np.nan_to_num(r), w) * _rolling_std(rb, w) + 1e-12)
    return [bc(breadth), bc(breadth_chg * 2), bc((avg_corr - 0.3) / 0.3), bc(dd_mkt), bc(last_mkt),
            bc(trend("GC=F")), bc(trend("TLT")), bc(trend("EURUSD=X", -1.0)), np.nan_to_num(corr_mkt)]


def _names_intraday(lookbacks: list[int], bpd: int) -> list[str]:
    def h(k):
        hours = k * 24 // bpd
        return f"{hours} h" if hours < 48 else f"{hours // 24} j"
    base = FEATURE_NAMES[3 * len(LOOKBACKS):]
    return ([f"momentum {h(k)}" for k in lookbacks] + [f"écart à la moyenne {h(k)}" for k in lookbacks]
            + [f"position dans le canal {h(k)}" for k in lookbacks] + base)


# ---------------------------------------------------------------- population
GENES_RISK = ["seuil_entree", "seuil_sortie", "stop_atr", "suiveur_atr", "vol_visee", "taille_max", "positions",
              "duree_max", "feu_vert_btc"]


@dataclass
class Population:
    feat: np.ndarray        # (P, K) indices des signaux
    w: np.ndarray           # (P, K) poids
    th_in: np.ndarray       # (P,)
    th_out: np.ndarray
    stop: np.ndarray        # multiple d'ATR
    trail: np.ndarray       # 0 = pas de stop suiveur
    tvol: np.ndarray        # volatilité visée
    maxpos: np.ndarray      # taille max d'une position (fraction)
    npos: np.ndarray        # nombre max de positions
    hold: np.ndarray        # durée max (jours, 0 = illimitée)
    gate: np.ndarray        # feu vert du Bitcoin
    assets: np.ndarray      # (P, A) actifs autorisés

    def __len__(self):
        return len(self.th_in)

    def take(self, i) -> "Population":
        return Population(*(getattr(self, f)[i] for f in self.__dataclass_fields__))

    @staticmethod
    def concat(ps: list["Population"]) -> "Population":
        return Population(*(np.concatenate([getattr(p, f) for p in ps]) for f in Population.__dataclass_fields__))


def random_population(n: int, A: int, rng: np.random.Generator) -> Population:
    th_in = rng.uniform(0.0, 0.8, n)
    return Population(
        feat=rng.integers(0, F, (n, K)), w=rng.uniform(-1, 1, (n, K)) * (rng.random((n, K)) < 0.8),
        th_in=th_in, th_out=th_in - rng.uniform(0.05, 1.0, n), stop=rng.uniform(1, 6, n),
        trail=np.where(rng.random(n) < 0.5, 0.0, rng.uniform(1, 6, n)), tvol=rng.uniform(0.05, 0.8, n),
        maxpos=rng.uniform(0.1, 1.0, n), npos=rng.integers(1, 6, n), hold=np.where(rng.random(n) < 0.6, 0, rng.integers(5, 120, n)),
        gate=rng.random(n) < 0.5, assets=rng.random((n, A)) < 0.6)


def reference_bot(A: int, assets: list[str]) -> Population:
    """Le bot actuel traduit en cerveau : momentum 14/28/56 j, 3 positions de 30 % max, volatilité visée 15 %."""
    i = [LOOKBACKS.index(k) for k in (14, 28, 56)]
    return Population(feat=np.array([i + [0]]), w=np.array([[1.0, 1.0, 1.0, 0.0]]), th_in=np.array([0.0]),
                      th_out=np.array([-1e-6]), stop=np.array([3.0]), trail=np.array([0.0]), tvol=np.array([0.15]),
                      maxpos=np.array([0.3]), npos=np.array([3]), hold=np.array([0]), gate=np.array([False]),
                      assets=np.array([[a in ("BTCUSDT", "ETHUSDT", "SOLUSDT") for a in assets]]))


LIMITS = {"max_size": 1.0, "min_assets": 1}      # règles imposées à tous les cerveaux (voir run_robust)


def clean(p: Population) -> Population:
    p.th_out = np.minimum(p.th_out, p.th_in - 0.01)
    p.stop = np.clip(p.stop, 0.5, 8)
    p.trail = np.clip(p.trail, 0, 8)
    p.tvol = np.clip(p.tvol, 0.02, 1.0)
    p.maxpos = np.clip(p.maxpos, 0.05, LIMITS["max_size"])
    p.npos = np.clip(p.npos, 1, 5)
    p.hold = np.clip(p.hold, 0, 200)
    p.w = np.clip(p.w, -1, 1)
    rng = np.random.default_rng(len(p))
    for _ in range(p.assets.shape[1]):                   # au moins `min_assets` actifs autorisés
        few = p.assets.sum(axis=1) < LIMITS["min_assets"]
        if not few.any():
            break
        cols = rng.integers(0, p.assets.shape[1], few.sum())
        p.assets[np.where(few)[0], cols] = True
    return p


# ---------------------------------------------------------------- simulation de toute la population
def simulate(m: Market, p: Population, t0: int, t1: int, start: float = 1000.0) -> dict:
    """Une période (une année) : tous les cerveaux en même temps. Décision à chaque clôture, stops pendant la
    journée suivante (au prix du stop, ou à l'ouverture si le prix a sauté dessous). Renvoie capital final, etc."""
    P, A = len(p), len(m.assets)
    held = np.zeros((P, A), bool)
    wgt = np.zeros((P, A))          # part du capital dans chaque actif
    stop_px = np.zeros((P, A))      # prix du stop (absolu)
    high_px = np.zeros((P, A))      # plus haute clôture depuis l'entrée (stop suiveur)
    age = np.zeros((P, A), int)
    eq = np.full(P, start)
    peak = eq.copy()
    alive = np.ones(P, bool)
    trades = np.zeros(P, int)
    worst = np.zeros(P)
    for t in range(t0, t1):
        if t > t0:                                        # 1) la journée t : stops, rendements
            o, lo, c, pc = np.nan_to_num(m.open[t]), m.low[t], m.close[t], m.close[t - 1]
            ret = np.nan_to_num(c / pc - 1)
            hit = held & (np.nan_to_num(lo, nan=np.inf)[None, :] <= stop_px)
            exit_px = np.minimum(o[None, :], stop_px)
            r_pos = np.where(hit, exit_px / np.nan_to_num(pc, nan=1.0)[None, :] - 1, ret[None, :])
            pnl = (wgt * r_pos).sum(1)
            fees = (wgt * hit).sum(1) * COST
            growth = np.maximum(1 + pnl - fees, 1e-9)
            eq = np.where(alive, eq * growth, eq)
            wgt = np.where(held & ~hit, wgt * (1 + r_pos) / growth[:, None], 0)
            trades += hit.sum(1)
            held &= ~hit
            high_px = np.where(held, np.maximum(high_px, np.nan_to_num(c)[None, :]), 0)
            age = np.where(held, age + 1, 0)
            peak = np.maximum(peak, eq)
            worst = np.maximum(worst, 1 - eq / peak)
            dead = alive & ((eq <= start * 0.75) | (eq <= peak * 0.65))        # arrêt d'urgence = mort
            alive &= ~dead
            held &= alive[:, None]
            wgt *= held
        # 2) décision à la clôture t
        x = m.X[t]                                        # (A, F)
        score = (x[:, p.feat] * p.w[None, :, :]).sum(2).T  # (P, A)
        tradable = m.ok[t][None, :] & p.assets & alive[:, None]
        gate_ok = ~p.gate | bool(m.btc_up[t])
        want_out = held & ((score < p.th_out[:, None]) | ((p.hold[:, None] > 0) & (age >= p.hold[:, None])))
        fees = (wgt * want_out).sum(1) * COST
        trades += want_out.sum(1)
        held &= ~want_out
        wgt *= held
        trail = held & (p.trail[:, None] > 0)
        stop_px = np.where(trail, np.maximum(stop_px, high_px * (1 - p.trail[:, None] * m.atr[t][None, :])), stop_px)
        cand = ~held & tradable & (score > p.th_in[:, None]) & gate_ok[:, None]
        free = p.npos - held.sum(1)
        s = np.where(cand, score, -np.inf)
        rank = np.argsort(np.argsort(-s, axis=1), axis=1)
        new = cand & (rank < free[:, None])
        size = np.minimum(p.tvol[:, None] / np.maximum(m.vol[t][None, :], 0.05), p.maxpos[:, None]) * new
        cash = np.maximum(1 - wgt.sum(1) - 0.005, 0)
        size *= np.minimum(1, cash / np.maximum(size.sum(1), 1e-12))[:, None]
        new &= size > 0.01
        fees += (size * new).sum(1) * COST
        eq = np.where(alive, eq * (1 - fees), eq)
        wgt = np.where(new, size, wgt)
        held |= new
        c_now = np.nan_to_num(m.close[t])[None, :]
        stop_px = np.where(new, c_now * (1 - p.stop[:, None] * m.atr[t][None, :]), np.where(held, stop_px, 0))
        high_px = np.where(new, c_now, high_px)
        trades += new.sum(1)
    eq = np.where(alive, eq * (1 - wgt.sum(1) * COST), eq)    # tout est vendu à la fin de la période
    return {"final": eq, "alive": alive, "trades": trades, "worst": worst}


# ---------------------------------------------------------------- évolution
def year_windows(m: Market, years: list[int]) -> list[tuple[int, int, int]]:
    out = []
    for y in years:
        a = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() // 86400)
        b = int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() // 86400)
        idx = np.where((m.days >= a) & (m.days < b))[0]
        if len(idx) > 30:
            out.append((y, int(idx[0]), int(idx[-1]) + 1))
    return out


def fitness(m: Market, p: Population, windows, objective: str = "max") -> tuple[np.ndarray, dict]:
    """objective = "max"     : gain moyen (l'objectif x1000 de Dhia)
                 "robuste" : la PIRE année compte le plus (survivre et gagner chaque année)."""
    logs, deaths, finals = [], np.zeros(len(p)), []
    for _, a, b in windows:
        r = simulate(m, p, a, b)
        lg = np.log(r["final"] / 1000.0)
        lg = np.where(r["alive"], lg, lg - 1.0)          # mourir coûte cher
        logs.append(lg)
        deaths += ~r["alive"]
        finals.append(r["final"])
    logs = np.array(logs)
    if objective == "robuste":
        fit = 0.7 * logs.min(axis=0) + 0.3 * logs.mean(axis=0) - 0.5 * (deaths > 0)
    else:
        fit = logs.mean(axis=0) - 0.05 * (deaths > 0)
    return fit, {"finals": np.array(finals), "deaths": deaths}


def evolve(m: Market, windows, pop_size: int = 2000, generations: int = 40, elite: float = 0.05,
           immigrants: float = 0.10, seed: int = 7, log=print, objective: str = "max") -> tuple[Population, np.ndarray, list]:
    rng = np.random.default_rng(seed)
    A = len(m.assets)
    pop = clean(Population.concat([reference_bot(A, m.assets), random_population(pop_size - 1, A, rng)]))
    history = []
    for g in range(generations):
        fit, info = fitness(m, pop, windows, objective)
        order = np.argsort(-fit)
        best = order[0]
        mult = np.exp(np.mean(np.log(np.maximum(info["finals"][:, best], 1) / 1000)))
        reached = int(((info["finals"] >= 1_000_000).any(axis=0)).sum())
        history.append({"gen": g, "best": float(fit[best]), "median": float(np.median(fit)),
                        "dead_pct": float((info["deaths"] > 0).mean() * 100), "best_mult": float(mult), "reached": reached})
        if log and (g % 5 == 0 or g == generations - 1):
            log(f"    génération {g:3d} : meilleur x{mult:.2f} par an en moyenne | morts {history[-1]['dead_pct']:.0f} % "
                f"| objectif x1000 atteint : {reached}")
        if g == generations - 1:
            return pop.take(order), fit[order], history
        n_elite = int(pop_size * elite)
        n_imm = int(pop_size * immigrants)
        n_child = pop_size - n_elite - n_imm

        def tournament(n):
            c = rng.integers(0, pop_size, (n, 4))
            return c[np.arange(n), np.argmax(fit[c], axis=1)]
        a, b = tournament(n_child), tournament(n_child)
        pa, pb = pop.take(a), pop.take(b)
        child = pop.take(a)
        for f in Population.__dataclass_fields__:        # croisement : chaque gène vient d'un des deux parents
            va, vb = getattr(pa, f), getattr(pb, f)
            mask = rng.random(va.shape) < 0.5
            setattr(child, f, np.where(mask, va, vb))
        mut = lambda shape, s: rng.normal(0, s, shape) * (rng.random(shape) < 0.25)  # noqa: E731
        child.w = child.w + mut(child.w.shape, 0.3)
        swap = rng.random(child.feat.shape) < 0.08
        child.feat = np.where(swap, rng.integers(0, F, child.feat.shape), child.feat)
        child.th_in = child.th_in + mut(n_child, 0.1)
        child.th_out = child.th_out + mut(n_child, 0.1)
        child.stop = child.stop + mut(n_child, 0.5)
        child.trail = np.where(rng.random(n_child) < 0.05, 0, child.trail + mut(n_child, 0.5))
        child.tvol = child.tvol * np.exp(mut(n_child, 0.2))
        child.maxpos = child.maxpos + mut(n_child, 0.1)
        child.npos = child.npos + np.round(mut(n_child, 0.7)).astype(int)
        child.hold = np.where(rng.random(n_child) < 0.05, 0, child.hold + np.round(mut(n_child, 10)).astype(int))
        child.gate = np.where(rng.random(n_child) < 0.05, ~child.gate, child.gate)
        child.assets = np.where(rng.random(child.assets.shape) < 0.05, ~child.assets, child.assets)
        pop = clean(Population.concat([pop.take(order[:n_elite]), child, random_population(n_imm, A, rng)]))


def describe(p: Population, i: int, assets: list[str]) -> str:
    sig = [f"{p.w[i, k]:+.2f} x {FEATURE_NAMES[p.feat[i, k]]}" for k in range(K) if abs(p.w[i, k]) > 0.02]
    acts = [a.removesuffix("USDT") for a, ok in zip(assets, p.assets[i]) if ok]
    return (f"signaux : {' ; '.join(sig) or 'aucun'} | entrer si score > {p.th_in[i]:.2f}, sortir si < {p.th_out[i]:.2f} | "
            f"stop {p.stop[i]:.1f} x ATR{f', suiveur {p.trail[i]:.1f} x ATR' if p.trail[i] > 0 else ''} | "
            f"volatilité visée {p.tvol[i] * 100:.0f} %, taille max {p.maxpos[i] * 100:.0f} %, {p.npos[i]} position(s)"
            f"{f', durée max {p.hold[i]} j' if p.hold[i] else ''}{', feu vert BTC' if p.gate[i] else ''} | actifs : {', '.join(acts)}")


# ---------------------------------------------------------------- l'expérience complète (python -m agent evolution)
REPORT_PATH = "data/evolution-report.txt"


def run_report(bars: dict, pop_size: int = 3000, generations: int = 50, log=print) -> dict:
    m = build_market(bars)
    A = len(m.assets)
    ref = reference_bot(A, m.assets)
    log("=" * 110)
    log(f"  LABORATOIRE D'ÉVOLUTION — {pop_size} cerveaux x {generations} générations par étape "
        f"(objectif de Dhia : 1 000 -> 1 000 000 en un an)")
    log("=" * 110)
    log("Chaque cerveau commence chaque année avec 1 000 USDT. Mort = arrêt d'urgence (-25 % / -35 %). Frais réels, sans levier.")
    log(f"Actifs possibles : {', '.join(a.removesuffix('USDT') for a in m.assets)}\n")
    folds = [([2022, 2023], 2024), ([2022, 2023, 2024], 2025), ([2022, 2023, 2024, 2025], 2026)]
    out = {"folds": []}
    for train, test in folds:
        log(f"ÉTAPE : apprentissage sur {train[0]}-{train[-1]}, puis test sur {test} (année jamais vue)")
        tw, sw = year_windows(m, train), year_windows(m, [test])
        pop, fit, hist = evolve(m, tw, pop_size, generations, log=log)
        _, info = fitness(m, pop.take(np.arange(min(200, len(pop)))), tw)
        best_train = info["finals"][:, 0]
        top = pop.take(np.arange(20))
        r = simulate(m, top, sw[0][1], sw[0][2])
        rr = simulate(m, ref, sw[0][1], sw[0][2])
        reached = int((info["finals"] >= 1_000_000).any(axis=0).sum())
        log(f"  Champion pendant l'apprentissage : " + ", ".join(f"{y} : 1 000 -> {v:,.0f}".replace(",", " ")
                                                               for (y, _, _), v in zip(tw, best_train)))
        log(f"  Objectif x1000 atteint pendant l'apprentissage par : {reached} cerveau(x) sur les 200 meilleurs")
        log(f"  TEST {test} (jamais vu) : champion 1 000 -> {r['final'][0]:,.0f}{'' if r['alive'][0] else ' (MORT)'} | "
            f"20 meilleurs : médiane {np.median(r['final']):,.0f}, morts {int((~r['alive']).sum())}/20 | "
            f"bot actuel {rr['final'][0]:,.0f}".replace(",", " "))
        log(f"  Champion : {describe(pop, 0, m.assets)}\n")
        out["folds"].append({"train": train, "test": test, "train_best": best_train.tolist(),
                             "test_champion": float(r["final"][0]), "test_champion_alive": bool(r["alive"][0]),
                             "test_top20_median": float(np.median(r["final"])), "test_top20_dead": int((~r["alive"]).sum()),
                             "test_reference": float(rr["final"][0]), "reached": reached})
    log("BILAN SUR LES ANNÉES JAMAIS VUES (ce qui compte vraiment)")
    ch = [f["test_champion"] for f in out["folds"]]
    tp = [f["test_top20_median"] for f in out["folds"]]
    rf = [f["test_reference"] for f in out["folds"]]
    for f in out["folds"]:
        log(f"  {f['test']} : champion {f['test_champion']:8,.0f} | 20 meilleurs (médiane) {f['test_top20_median']:8,.0f} | "
            f"bot actuel {f['test_reference']:8,.0f}".replace(",", " "))
    gm = lambda v: math.exp(np.mean(np.log(np.maximum(v, 1) / 1000))) * 1000   # noqa: E731
    log(f"  Moyenne par an : champion {gm(ch):,.0f} | 20 meilleurs {gm(tp):,.0f} | bot actuel {gm(rf):,.0f}".replace(",", " "))
    best_ever = max(max(f["train_best"]) for f in out["folds"])
    log(f"  Meilleure année d'un champion PENDANT l'apprentissage : 1 000 -> {best_ever:,.0f} ; "
        f"objectif 1 000 000 : {'atteint' if best_ever >= 1e6 else 'jamais atteint'}".replace(",", " "))
    out["summary"] = {"champion": gm(ch), "top20": gm(tp), "reference": gm(rf), "best_train_year": best_ever}
    log("=" * 110)
    return out


def run_robust(bars: dict, pop_size: int = 2500, generations: int = 40, log=print, max_size: float = 1.0,
               min_assets: int = 1) -> dict:
    """2e expérience : note « robuste » (la pire année compte) + examen sur une année mise de côté avant le test.
    max_size / min_assets : interdire de tout miser (taille max par position, nombre minimum d'actifs)."""
    LIMITS.update(max_size=max_size, min_assets=min_assets)
    m = build_market(bars)
    ref = reference_bot(len(m.assets), m.assets)
    log("=" * 110)
    log(f"  2e ÉVOLUTION « ROBUSTE » — {pop_size} cerveaux x {generations} générations ; note = surtout la PIRE année ;")
    log("  champion choisi par un examen sur une année non apprise, puis testé sur l'année suivante (jamais vue)")
    log(f"  règles imposées : taille max {max_size * 100:.0f} % par position, au moins {min_assets} actif(s)")
    log("=" * 110)
    folds = [([2022], 2023, 2024), ([2022, 2023], 2024, 2025), ([2022, 2023, 2024], 2025, 2026)]
    res = []
    for train, exam, test in folds:
        log(f"ÉTAPE : apprentissage {train[0]}-{train[-1]}, examen {exam}, test {test}")
        pop, fit, _ = evolve(m, year_windows(m, train), pop_size, generations, log=log, objective="robuste")
        top = pop.take(np.arange(100))
        ew, tw = year_windows(m, [exam])[0], year_windows(m, [test])[0]
        ex = simulate(m, top, ew[1], ew[2])
        score = np.where(ex["alive"], np.log(ex["final"] / 1000), -9)
        best = int(np.argmax(score))
        champ = top.take(np.array([best]))
        te = simulate(m, champ, tw[1], tw[2])
        rr = simulate(m, ref, tw[1], tw[2])
        log(f"  Examen {exam} : champion choisi 1 000 -> {ex['final'][best]:,.0f} | TEST {test} : champion 1 000 -> "
            f"{te['final'][0]:,.0f}{'' if te['alive'][0] else ' (MORT)'} | bot actuel {rr['final'][0]:,.0f}".replace(",", " "))
        log(f"  Champion : {describe(top, best, m.assets)}\n")
        res.append((test, float(te["final"][0]), bool(te["alive"][0]), float(rr["final"][0])))
    gm = lambda v: math.exp(np.mean(np.log(np.maximum(v, 1) / 1000))) * 1000   # noqa: E731
    log(f"BILAN : moyenne par an sur les années jamais vues : champion robuste {gm([r[1] for r in res]):,.0f} | "
        f"bot actuel {gm([r[3] for r in res]):,.0f}".replace(",", " "))
    log("=" * 110)
    return {"folds": res}
