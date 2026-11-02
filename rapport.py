"""Assemble le rapport d'une expérience, comme sur le PC. Usage : rapport.py <4|5>"""
import json
import sys
from pathlib import Path

from agent.ecosysteme import generations_report, report_tout

exp = sys.argv[1]
base = {"4": 30, "5": 50}[exp]
files = [Path(f"resultats/tout-{base + i:02d}.json") for i in range(8)]
res = [json.loads(f.read_text(encoding="utf-8")) for f in files if f.exists()]
print(f"{len(res)} mondes sur 8", flush=True)
path = f"resultats/monde{exp}-report.txt"
text = report_tout(res, path)
Path(path).write_text(text + "\n\n" + generations_report(res), encoding="utf-8")
if exp == "4":
    best = max(res, key=lambda r: r["fund_curve"][-1][1])
    Path("resultats/cerveau-champion.json").write_text(json.dumps(
        {"settings": best["settings"], "leaders": best["final_leaders"], "fund_final": best["fund_curve"][-1][1],
         "yearly": best["yearly"]}, ensure_ascii=False, indent=1), encoding="utf-8")
print(text)
