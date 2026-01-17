#!/usr/bin/env python3
# analysis/tools/analyze_thresholds.py

import json
import csv
import statistics
from pathlib import Path
from typing import List, Dict

PROMOTED = Path("results/promotions/faseA_promoted.json")
REJECTED = Path("results/promotions/why_rejected_promotion_rules_A.csv")

METRICS = [
    "trades_min",
    "pf_mean",
    "winrate_mean",
    "dd_max",
    "expectancy_mean",
    "robust_score_mean",
]

def load_rejected() -> List[Dict]:
    rows = []
    with open(REJECTED, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(r[k]) for k in METRICS})
    return rows

def load_promoted() -> List[Dict]:
    rows = []
    with open(PROMOTED, encoding="utf-8") as f:
        data = json.load(f)
    for p in data:
        for win in p["windows"].values():
            for _ in win["scores"]:
                rows.append({
                    "robust_score_mean": statistics.mean(p["scores"]),
                    # resto ya está implícito en promoted
                })
    return rows

def percentile(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = int(len(xs) * q)
    return xs[min(k, len(xs) - 1)]

def analyze(label: str, rows: List[Dict]):
    print(f"\n=== {label} ===")
    for m in METRICS:
        vals = [r[m] for r in rows if m in r]
        if not vals:
            continue
        print(
            f"{m:22s} | "
            f"p50={percentile(vals,0.50):7.3f} "
            f"p60={percentile(vals,0.60):7.3f} "
            f"p70={percentile(vals,0.70):7.3f} "
            f"p80={percentile(vals,0.80):7.3f}"
        )

def main():
    rejected = load_rejected()
    analyze("REJECTED", rejected)

if __name__ == "__main__":
    main()
