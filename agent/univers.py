"""Tout le marché : actions, indices, obligations, matières premières, devises et cryptos (Yahoo Finance, gratuit).

Sert seulement à la recherche (arène / écosystème). 26 ans d'historique pour les marchés traditionnels :
bulle internet (2000), crise financière (2008), Covid (2020), inflation (2022), et toutes les phases crypto.
On y met volontairement des gagnantes ET des perdantes connues aujourd'hui, pour limiter le biais de survie.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import requests

from .dataset import Dataset
from .strategy import Bar

PATH = "data/history-marche.json.gz"
UNIVERSE = {
    "actions": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "JNJ", "WMT", "KO",
                "INTC", "GE", "C", "BA", "IBM", "PFE", "F"],
    "indices": ["SPY", "QQQ", "IWM", "EEM", "TLT"],
    "matieres": ["GC=F", "SI=F", "USO"],
    "devises": ["EURUSD=X", "JPY=X"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD"],
}
NAMES = {"GC=F": "OR", "SI=F": "ARGENT", "USO": "PETROLE", "EURUSD=X": "EURUSD", "JPY=X": "USDJPY",
         "BTC-USD": "BTC", "ETH-USD": "ETH", "SOL-USD": "SOL", "XRP-USD": "XRP", "BNB-USD": "BNB", "DOGE-USD": "DOGE"}


def label(sym: str) -> str:
    return NAMES.get(sym, sym)


def yahoo_daily(sym: str, start: str = "1999-01-01") -> list[Bar]:
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                     params={"period1": p1, "period2": int(time.time()), "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    bars, seen = [], set()
    for i, ts in enumerate(res.get("timestamp") or []):
        o, h, lo, c, v = (q[k][i] for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, lo, c) or min(o, h, lo, c) <= 0:
            continue
        day = ts // 86400
        if day in seen:
            continue
        seen.add(day)
        bars.append(Bar(day * 86_400_000, float(o), float(max(h, o, c)), float(min(lo, o, c)), float(c), float(v or 0)))
    return bars


def download(log=print) -> Dataset:
    ds = Dataset(interval=1440)
    for group, syms in UNIVERSE.items():
        for s in syms:
            try:
                ds.bars[s] = yahoo_daily(s)
                b = ds.bars[s]
                log(f"  {group:9} {label(s):8} {len(b):5d} jours depuis le "
                    f"{datetime.fromtimestamp(b[0].ts / 1000, tz=timezone.utc):%d/%m/%Y}")
            except Exception as e:  # noqa: BLE001
                log(f"  {group:9} {label(s):8} ERREUR : {e}")
            time.sleep(0.3)
    ds.save(PATH)
    return ds


def group_of(sym: str) -> str:
    return next(g for g, ss in UNIVERSE.items() if sym in ss)
