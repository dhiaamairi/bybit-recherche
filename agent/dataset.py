"""Téléchargement et sauvegarde de l'historique (bougies 1h + Fear & Greed + funding rates).

Le fichier data/history.json.gz peut être réutilisé par 'backtest' et 'compare' sans tout
retélécharger, et peut être envoyé pour analyse.
"""
from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .strategy import Bar

DEFAULT_PATH = "data/history.json.gz"

# Cryptos supplémentaires téléchargées pour la recherche (grandes capitalisations, très liquides sur Bybit).
# Les études trouvent le momentum surtout sur les grandes cryptos (Grobys et al. 2025).
EXTRA_CRYPTOS = ["XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"]


@dataclass
class Dataset:
    interval: int = 60
    bars: dict[str, list[Bar]] = field(default_factory=dict)
    fng: list[tuple[int, int]] = field(default_factory=list)
    funding: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    created: int = 0

    def save(self, path: str = DEFAULT_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "interval": self.interval,
            "created": self.created or int(time.time()),
            "bars": {s: [[b.ts, b.open, b.high, b.low, b.close, b.volume] for b in bars]
                     for s, bars in self.bars.items()},
            "fng": [list(x) for x in self.fng],
            "funding": {s: [list(x) for x in rows] for s, rows in self.funding.items()},
        }
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "Dataset":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            interval=d["interval"],
            bars={s: [Bar(int(r[0]), *map(float, r[1:6])) for r in rows] for s, rows in d["bars"].items()},
            fng=[(int(t), int(v)) for t, v in d.get("fng", [])],
            funding={s: [(int(t), float(v)) for t, v in rows] for s, rows in d.get("funding", {}).items()},
            created=d.get("created", 0),
        )

    def summary(self) -> str:
        lines = []
        for s, bars in self.bars.items():
            days = (bars[-1].ts - bars[0].ts) / 86_400_000 if bars else 0
            lines.append(f"  {s:12} {len(bars):6d} bougies ({days:.0f} jours, depuis le "
                         f"{time.strftime('%d/%m/%Y', time.gmtime(bars[0].ts / 1000)) if bars else '?'})")
        lines.append(f"  Fear & Greed : {len(self.fng)} jours | Funding : "
                     + ", ".join(f"{s} {len(r)}" for s, r in self.funding.items()))
        return "\n".join(lines)


def all_symbols(cfg: dict, extra: bool = True) -> tuple[list[str], list[str]]:
    """(symboles à télécharger, symboles crypto pour le funding)"""
    crypto = list(cfg["symbols"].get("crypto") or [])
    syms = crypto + list(cfg["symbols"].get("stocks") or [])
    if extra:
        syms += [s for s in EXTRA_CRYPTOS if s not in syms]
    reg = cfg["filters"]["regime"]
    for ref in (reg.get("crypto_symbol"), reg.get("stocks_symbol")):
        if ref and ref not in syms:
            syms.append(ref)
    return syms, crypto


def download(cfg: dict, days: int, log=print, interval: int = 1440, extra: bool = True) -> Dataset:
    from .backtest import fetch_history
    from .context import fetch_fear_greed_history, fetch_funding_history

    syms, crypto = all_symbols(cfg, extra)
    ds = Dataset(interval=interval, created=int(time.time()))
    label = "journalières" if interval == 1440 else f"{interval} min"
    for s in syms:
        log(f"Téléchargement {s} ({days} jours, bougies {label})...")
        try:
            bars = fetch_history(s, days, interval)
        except Exception as e:  # noqa: BLE001
            log(f"  ignoré : {e}")
            continue
        if bars:
            ds.bars[s] = bars
            log(f"  {len(bars)} bougies")
        else:
            log("  aucune donnée (symbole inconnu ?)")
    if not ds.bars:
        raise RuntimeError("Aucune donnée téléchargée.")
    start_ms = min(b[0].ts for b in ds.bars.values())
    try:
        ds.fng = fetch_fear_greed_history()
        log(f"Fear & Greed : {len(ds.fng)} jours")
    except Exception as e:  # noqa: BLE001
        log(f"Fear & Greed indisponible : {e}")
    for s in crypto:
        try:
            ds.funding[s] = fetch_funding_history(s, start_ms)
            log(f"Funding {s} : {len(ds.funding[s])} valeurs")
        except Exception as e:  # noqa: BLE001
            log(f"Funding {s} indisponible : {e}")
    return ds
