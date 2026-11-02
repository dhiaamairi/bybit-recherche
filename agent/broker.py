"""Extrait de broker.py pour la recherche : le regroupement des bougies seulement, aucun code d'ordre."""
from __future__ import annotations

from .strategy import Bar


def resample(bars: list[Bar], tf_minutes: int) -> list[Bar]:
    """Regroupe des bougies (ex: 1h) en bougies plus longues (ex: 4h, 1 jour UTC)."""
    size = tf_minutes * 60_000
    out: list[Bar] = []
    for b in bars:
        start = b.ts // size * size
        if out and out[-1].ts == start:
            o = out[-1]
            o.high = max(o.high, b.high)
            o.low = min(o.low, b.low)
            o.close = b.close
            o.volume += b.volume
        else:
            out.append(Bar(start, b.open, b.high, b.low, b.close, b.volume))
    return out
