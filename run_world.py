"""Lance un seul monde (une machine GitHub Actions par monde). Usage : run_world.py <4|5> <numéro> [cerveaux]
Mêmes réglages et mêmes numéros que sur le PC : 4 = expérience 2 (monde 4), 5 = expérience 3 (monde 5)."""
import sys
from pathlib import Path

from agent.ecosysteme import GRID_MONDE4, GRID_MONDE5, _run_monde4, _run_tout

exp, i = int(sys.argv[1]), int(sys.argv[2])
pop = int(sys.argv[3]) if len(sys.argv) > 3 else None
Path("resultats").mkdir(exist_ok=True)
if exp == 4:
    s = GRID_MONDE4[i]
    s.pop = pop or s.pop
    print(f"Expérience 2, monde {i} : {s.name()} | {s.pop} cerveaux", flush=True)
    res = _run_monde4((30 + i, s, "data/history-binance-1h.json.gz", "resultats"))
else:
    s = GRID_MONDE5[i]
    s.pop = pop or 40000
    print(f"Expérience 3, monde {i} : {s.name()} | {s.pop} cerveaux", flush=True)
    res = _run_tout((50 + i, s, "data/history-marche.json.gz", "resultats"))
print(f"Terminé : fonds 1 000 -> {res['fund_curve'][-1][1]:,.0f}", flush=True)
