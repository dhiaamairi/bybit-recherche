"""Stratégies de suivi de tendance, achat seulement (spot, sans levier).

Choix avec strategy.type dans config.yaml :

  ema_cross  (version d'origine)
      Achat : prix > EMA200, EMA20 > EMA50, croisement ou rebond, RSI 40-70.
      Vente : EMA20 < EMA50, prix < EMA200, ou stop suiveur.

  sma_trend  (Faber 2007 « A Quantitative Approach to Tactical Asset Allocation » ;
              la moyenne 50 jours sert de référence dans les tests crypto récents)
      Achat : clôture au-dessus de la moyenne simple sur N bougies.
      Vente : clôture en dessous.

  donchian   (règles des « Turtles », Dennis & Eckhardt 1983, décrites par Curtis Faith,
              « Way of the Turtle », 2007)
      Achat : clôture au-dessus du plus haut des N dernières bougies (20 = Système 1, 55 = Système 2).
      Vente : clôture sous le plus bas des M dernières bougies (10 ou 20), ou stop à 2 x ATR.

  tsmom      (« time-series momentum » : Moskowitz, Ooi & Pedersen 2012 ;
              Liu & Tsyvinski 2021 pour la crypto, horizons de quelques semaines)
      Achat : la majorité des rendements passés (ex : 14, 28, 56 jours) sont positifs.
      Vente : la majorité sont négatifs.

  combo      (vote de 3 signaux indépendants, pour ne pas dépendre d'une seule règle)
      Signal 1 : momentum (majorité des rendements 14/28/56 j positifs)
      Signal 2 : prix au-dessus de la moyenne 50 (Faber)
      Signal 3 : prix dans la moitié haute du canal 20 bougies (Turtles)
      Achat : au moins `combo_enter` signaux sur 3 (2 par défaut).
      Vente : `combo_exit` signaux ou moins (1 par défaut).

  rebond     (retour à la moyenne : RSI 2 de Larry Connors, « Short Term Trading Strategies
              That Work », 2008)
      Achat : tendance de fond haussière (prix > moyenne 100) ET forte baisse sur 2-3 jours (RSI 2 < 10).
      Vente : le prix repasse au-dessus de la moyenne 5 jours (le rebond a eu lieu).

  --- méthodes tirées des livres du dépôt « some-investment-books » ---

  elder      (« Triple Screen » + « Impulse System » d'Alexander Elder, « Trading for a Living » 1993,
              « Come Into My Trading Room » 2002)
      Écran 1 (marée) : la moyenne EMA 65 jours (≈ 13 semaines) monte depuis une semaine.
      Écran 2 (vague) : repli récent, l'indice de force (Force Index EMA 2) est passé sous 0.
      Écran 3 (entrée) : Impulse « vert » = EMA 13 ET histogramme MACD en hausse.
      Vente : la marée tourne (EMA 65 en baisse et prix dessous) ou Impulse « rouge » sous l'EMA 13.

  ichimoku   (Ichimoku Kinko Hyo, Manesh Patel, « Trading with Ichimoku Clouds » 2010)
      Achat : prix au-dessus du nuage, Tenkan (9) > Kijun (26), Chikou au-dessus du prix d'il y a 26 jours.
      Vente : clôture sous la Kijun ou sous le nuage.

  vol_breakout (cassure confirmée par le volume : Anna Coulling, « Volume Price Analysis » 2013 ;
              William O'Neil, « How to Make Money in Stocks » : volume >= 1,5 x la moyenne)
      Achat : cassure du plus haut N bougies AVEC un volume >= vol_mult x la moyenne 50 bougies.
      Vente : cassure du plus bas M bougies.

  Filtre commun adx_min (Welles Wilder 1978, repris par John Murphy, « Technical Analysis of the
  Financial Markets ») : si adx_min > 0, un achat n'est accepté que si l'ADX dépasse ce seuil
  (le marché est vraiment en tendance, pas en zigzag).

  Options communes vsa_filter / vsa_exit (Tom Williams, « Master the Markets », voir vsa.py) :
  vsa_filter : pas d'achat si des signes de faiblesse récents (up-thrust, climax d'achat, pas de
               demande...) ne sont pas suivis de signes de force.
  vsa_exit   : vente si la dernière bougie est un climax d'achat ou un up-thrust.
"""
from __future__ import annotations

from dataclasses import dataclass

from .indicators import adx, atr, ema, rsi


@dataclass
class Bar:
    ts: int        # ouverture de la bougie (ms)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Analysis:
    ready: bool
    close: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_trend: float = 0.0
    rsi: float = 50.0
    atr: float = 0.0
    enter: bool = False
    exit: bool = False
    reason: str = ""


STRATEGY_NAMES = {
    "ema_cross": "croisement de moyennes EMA",
    "sma_trend": "prix au-dessus de la moyenne (Faber)",
    "donchian": "cassure de canal (Turtles)",
    "tsmom": "momentum multi-horizons (TSMOM)",
    "combo": "vote de 3 signaux (momentum + moyenne + canal)",
    "rebond": "achat des replis dans une tendance haussière (RSI 2)",
    "champion": "système complet (14 méthodes + un chef qui choisit)",
    "elder": "Triple Screen + Impulse (Elder)",
    "ichimoku": "nuage Ichimoku",
    "vol_breakout": "cassure confirmée par le volume (Coulling / O'Neil)",
}


def sma(values: list[float], n: int) -> float | None:
    return sum(values[-n:]) / n if len(values) >= n else None


def required_bars(p: dict) -> int:
    t = p.get("type", "ema_cross")
    if t == "sma_trend":
        n = p["sma_period"]
    elif t == "donchian":
        n = max(p["entry_period"], p["exit_period"])
    elif t == "tsmom":
        n = max(p["lookbacks"])
    elif t == "combo":
        n = max(max(p["lookbacks"]), p["sma_period"], p["entry_period"])
    elif t == "rebond":
        n = max(p.get("mr_trend_sma", 100), p.get("mr_exit_sma", 5))
    elif t == "elder":
        n = max(p.get("elder_tide", 65) + 6, 26 + 9 + 2)
    elif t == "ichimoku":
        n = p.get("ichi_slow", 52) + p.get("ichi_mid", 26) + 1
    elif t == "vol_breakout":
        n = max(p["entry_period"], p["exit_period"], p.get("vol_avg", 50)) + 1
    elif t == "champion":   # système complet : assez d'historique pour toutes les méthodes
        from .ensemble import EXPERTS
        return max(required_bars(e.strategy(p)) for e in EXPERTS)
    else:
        n = p["ema_trend"]
    if p.get("adx_min"):
        n = max(n, 2 * p.get("adx_period", 14) + 1)
    if p.get("vsa_filter") or p.get("vsa_exit"):
        n = max(n, 34)
    return max(n, p["atr_period"], p["rsi_period"]) + 2


def analyze(bars: list[Bar], p: dict) -> Analysis:
    need = required_bars(p)
    if len(bars) < need:
        return Analysis(ready=False, reason=f"pas assez d'historique ({len(bars)}/{need})")

    closes = [b.close for b in bars]
    r = rsi(closes, p["rsi_period"])
    a = atr([b.high for b in bars], [b.low for b in bars], closes, p["atr_period"])
    if r[-1] is None or a[-1] is None:
        return Analysis(ready=False, reason="indicateurs en cours de calcul")

    t = p.get("type", "ema_cross")
    fn = {"ema_cross": _ema_cross, "sma_trend": _sma_trend, "donchian": _donchian, "tsmom": _tsmom,
          "combo": _combo, "rebond": _rebond, "elder": _elder, "ichimoku": _ichimoku,
          "vol_breakout": _vol_breakout}.get(t)
    if fn is None:
        raise ValueError(f"Stratégie inconnue : {t}")
    res = Analysis(ready=True, close=closes[-1], rsi=r[-1], atr=a[-1])
    res = fn(bars, closes, r, res, p)
    if res.enter and p.get("adx_min"):
        x = adx([b.high for b in bars], [b.low for b in bars], closes, p.get("adx_period", 14))[-1]
        if x is None or x < p["adx_min"]:
            res.enter = False
            res.reason = f"tendance trop faible (ADX {x or 0:.0f} < {p['adx_min']}) — {res.reason}"
    if p.get("vsa_filter") or p.get("vsa_exit"):
        from .vsa import read, weakness_blocks
        v = read(bars)
        if res.enter and p.get("vsa_filter") and weakness_blocks(v, p.get("vsa_max_score", -1.0)):
            res.enter = False
            res.reason = f"faiblesse récente sur le volume (VSA : {v.summary()}) — {res.reason}"
        last = v.bars[-1].signals if v.ready and v.bars else []
        if p.get("vsa_exit") and ("buying_climax" in last or "upthrust" in last):
            res.enter, res.exit = False, True
            what = "climax d'achat" if "buying_climax" in last else "up-thrust"
            res.reason = f"VSA : {what} sur la dernière bougie"
    return res


# ------------------------------------------------------------------ stratégies
def _ema_cross(bars, closes, r, res, p) -> Analysis:
    ef = ema(closes, p["ema_fast"])
    es = ema(closes, p["ema_slow"])
    et = ema(closes, p["ema_trend"])
    if None in (ef[-1], ef[-2], es[-1], es[-2], et[-1]):
        return Analysis(ready=False, reason="indicateurs en cours de calcul")
    c, pc = closes[-1], closes[-2]
    res.ema_fast, res.ema_slow, res.ema_trend = ef[-1], es[-1], et[-1]

    uptrend = c > et[-1] and ef[-1] > es[-1]
    crossed_up = ef[-2] <= es[-2] and ef[-1] > es[-1]
    lb = p.get("pullback_lookback", 6)
    recent_rsi = [x for x in r[-lb:] if x is not None]
    # rebond seulement après un vrai repli (RSI récemment descendu sous pullback_rsi)
    bounce = pc <= ef[-2] and c > ef[-1] and bool(recent_rsi) and min(recent_rsi) <= p.get("pullback_rsi", 45)
    rsi_ok = p["rsi_min"] <= r[-1] <= p["rsi_max"]

    if uptrend and rsi_ok and (crossed_up or bounce):
        res.enter = True
        res.reason = "croisement EMA" if crossed_up else "rebond sur EMA20"
    elif ef[-1] < es[-1]:
        res.exit = True
        res.reason = "EMA20 sous EMA50"
    elif c < et[-1]:
        res.exit = True
        res.reason = "prix sous EMA200"
    return res


def _sma_trend(bars, closes, r, res, p) -> Analysis:
    n = p["sma_period"]
    m = sma(closes, n)
    res.ema_fast = res.ema_slow = res.ema_trend = m
    band = p.get("sma_band_pct", 0.0) / 100
    if closes[-1] > m * (1 + band):
        res.enter = True
        res.reason = f"prix au-dessus de la moyenne {n}"
    elif closes[-1] < m * (1 - band):
        res.exit = True
        res.reason = f"prix sous la moyenne {n}"
    return res


def _donchian(bars, closes, r, res, p) -> Analysis:
    ne, nx = p["entry_period"], p["exit_period"]
    prior = bars[:-1]
    hh = max(b.high for b in prior[-ne:])
    ll = min(b.low for b in prior[-nx:])
    res.ema_fast, res.ema_slow, res.ema_trend = hh, ll, sma(closes, ne)
    if closes[-1] > hh:
        res.enter = True
        res.reason = f"cassure du plus haut {ne} bougies"
    elif closes[-1] < ll:
        res.exit = True
        res.reason = f"cassure du plus bas {nx} bougies"
    return res


def _tsmom(bars, closes, r, res, p) -> Analysis:
    c = closes[-1]
    rets = [c / closes[-lb - 1] - 1 for lb in p["lookbacks"]]
    score = sum(1 if x > 0 else -1 for x in rets) / len(rets)
    res.ema_fast = res.ema_slow = res.ema_trend = sma(closes, max(p["lookbacks"]))
    unit = "j" if p.get("_interval", 1440) == 1440 else "bougies"
    detail = ", ".join(f"{lb} {unit} : {x * 100:+.1f} %".replace(".", ",") for lb, x in zip(p["lookbacks"], rets))
    if score > 0:
        res.enter = True
        res.reason = f"momentum positif ({detail})"
    elif score < 0:
        res.exit = True
        res.reason = f"momentum négatif ({detail})"
    return res


def _combo(bars, closes, r, res, p) -> Analysis:
    c = closes[-1]
    rets = [c / closes[-lb - 1] - 1 for lb in p["lookbacks"]]
    s1 = sum(1 if x > 0 else -1 for x in rets) > 0
    m = sma(closes, p["sma_period"])
    s2 = c > m
    n = p["entry_period"]
    hh = max(b.high for b in bars[-n - 1:-1])
    ll = min(b.low for b in bars[-n - 1:-1])
    s3 = c > (hh + ll) / 2
    votes = int(s1) + int(s2) + int(s3)
    res.ema_fast, res.ema_slow, res.ema_trend = m, (hh + ll) / 2, m
    detail = (f"momentum {'✓' if s1 else '✗'}, moyenne {p['sma_period']} {'✓' if s2 else '✗'}, "
              f"canal {n} {'✓' if s3 else '✗'}")
    if votes >= p.get("combo_enter", 2):
        res.enter = True
        res.reason = f"{votes}/3 signaux haussiers ({detail})"
    elif votes <= p.get("combo_exit", 1):
        res.exit = True
        res.reason = f"{votes}/3 signaux haussiers seulement ({detail})"
    else:
        res.reason = f"{votes}/3 signaux ({detail})"
    return res


def _rebond(bars, closes, r, res, p) -> Analysis:
    c = closes[-1]
    trend = sma(closes, p.get("mr_trend_sma", 100))
    back = sma(closes, p.get("mr_exit_sma", 5))
    r2 = rsi(closes[-60:], p.get("mr_rsi_period", 2))[-1]
    res.ema_fast, res.ema_slow, res.ema_trend = back, back, trend
    if r2 is None:
        return res
    if c > trend and r2 < p.get("mr_entry", 10):
        res.enter = True
        res.reason = f"repli dans une tendance haussière (RSI2 {r2:.0f}, prix > moyenne {p.get('mr_trend_sma', 100)})"
    elif c > back:
        res.exit = True
        res.reason = f"rebond terminé (prix au-dessus de la moyenne {p.get('mr_exit_sma', 5)} jours)"
    return res


def _macd_hist(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> list[float]:
    ef, es = ema(closes, fast), ema(closes, slow)
    line = [f - s for f, s in zip(ef, es) if f is not None and s is not None]
    sig = ema(line, signal)
    return [m - g for m, g in zip(line, sig) if g is not None]


def _elder(bars, closes, r, res, p) -> Analysis:
    tide_n = p.get("elder_tide", 65)
    tide = ema(closes, tide_n)
    e13 = ema(closes, 13)
    hist = _macd_hist(closes)
    if None in (tide[-1], tide[-6], e13[-1], e13[-2]) or len(hist) < 2:
        return Analysis(ready=False, reason="indicateurs en cours de calcul")
    c = closes[-1]
    res.ema_fast, res.ema_slow, res.ema_trend = e13[-1], e13[-1], tide[-1]
    tide_up = tide[-1] > tide[-6]
    green = e13[-1] > e13[-2] and hist[-1] > hist[-2]
    red = e13[-1] < e13[-2] and hist[-1] < hist[-2]
    start = max(1, len(closes) - 30)
    force = [(closes[i] - closes[i - 1]) * bars[i].volume for i in range(start, len(closes))]
    fi2 = [x for x in ema(force, 2) if x is not None]
    pullback = bool(fi2) and min(fi2[-p.get("elder_wave_bars", 3):]) < 0
    if tide_up and green and pullback:
        res.enter = True
        res.reason = f"Elder : marée haussière (EMA {tide_n}), repli puis Impulse vert"
    elif not tide_up and c < tide[-1]:
        res.exit = True
        res.reason = f"Elder : la marée tourne (EMA {tide_n} en baisse, prix dessous)"
    elif red and c < e13[-1]:
        res.exit = True
        res.reason = "Elder : Impulse rouge sous l'EMA 13"
    else:
        res.reason = "Elder : " + ("marée haussière" if tide_up else "marée baissière") + \
            (", Impulse vert" if green else ", Impulse rouge" if red else ", Impulse neutre")
    return res


def _ichimoku(bars, closes, r, res, p) -> Analysis:
    fast, mid, slow = p.get("ichi_fast", 9), p.get("ichi_mid", 26), p.get("ichi_slow", 52)

    def midpoint(end: int, n: int) -> float:       # (plus haut + plus bas) / 2 des n bougies finissant à end
        w = bars[end - n + 1: end + 1]
        return (max(b.high for b in w) + min(b.low for b in w)) / 2

    last = len(bars) - 1
    tenkan, kijun = midpoint(last, fast), midpoint(last, mid)
    past = last - mid                                # le nuage d'aujourd'hui a été tracé il y a 26 bougies
    span_a = (midpoint(past, fast) + midpoint(past, mid)) / 2
    span_b = midpoint(past, slow)
    top, bottom = max(span_a, span_b), min(span_a, span_b)
    c = closes[-1]
    res.ema_fast, res.ema_slow, res.ema_trend = tenkan, kijun, top
    chikou_ok = c > closes[-mid - 1]
    if c > top and tenkan > kijun and chikou_ok:
        res.enter = True
        res.reason = "Ichimoku : prix au-dessus du nuage, Tenkan > Kijun, Chikou confirmé"
    elif c < kijun or c < bottom:
        res.exit = True
        res.reason = "Ichimoku : " + ("prix sous le nuage" if c < bottom else "clôture sous la Kijun")
    else:
        res.reason = "Ichimoku : " + ("au-dessus du nuage" if c > top else "dans le nuage")
    return res


def _vol_breakout(bars, closes, r, res, p) -> Analysis:
    ne, nx, nv = p["entry_period"], p["exit_period"], p.get("vol_avg", 50)
    prior = bars[:-1]
    hh = max(b.high for b in prior[-ne:])
    ll = min(b.low for b in prior[-nx:])
    avg_v = sum(b.volume for b in prior[-nv:]) / nv
    ratio = bars[-1].volume / avg_v if avg_v > 0 else 0.0
    res.ema_fast, res.ema_slow, res.ema_trend = hh, ll, sma(closes, ne)
    mult = p.get("vol_mult", 1.5)
    if closes[-1] > hh and ratio >= mult:
        res.enter = True
        res.reason = f"cassure du plus haut {ne} bougies avec volume x{ratio:.1f}".replace(".", ",")
    elif closes[-1] < ll:
        res.exit = True
        res.reason = f"cassure du plus bas {nx} bougies"
    elif closes[-1] > hh:
        res.reason = f"cassure sans volume (x{ratio:.1f} < x{mult}) : ignorée".replace(".", ",")
    return res


def momentum_score(bars: list[Bar], lookback: int = 28, vol_lookback: int = 30) -> float:
    """Force de la tendance ajustée du risque : rendement sur `lookback` / volatilité.
    Sert à choisir les meilleurs actifs quand il y a plus de signaux que de places."""
    closes = [b.close for b in bars]
    if len(closes) <= max(lookback, vol_lookback) + 1:
        return 0.0
    ret = closes[-1] / closes[-lookback - 1] - 1
    rets = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - vol_lookback, len(closes))]
    mean = sum(rets) / len(rets)
    sd = (sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)) ** 0.5
    return ret / sd if sd else 0.0
