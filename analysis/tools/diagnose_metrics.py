#!/usr/bin/env python3
# analysis/tools/diagnose_metrics.py

from __future__ import annotations

import os
import json
import glob
import csv
import statistics
from typing import Dict, Any, List

ROBUST_DIR = "results/robust"
OUT_DIR = "results/diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = [
    "trades",
    "profit_factor",
    "winrate",
    "max_drawdown_r",
    "expectancy",
    "robust_score",
]

def normalize_from_folds(rec: Dict[str, Any]) -> Dict[str, float]:
    folds = rec.get("folds", []) or []

    def vals(key, default=0.0):
        return [
            float(f.get("metrics", {}).get(key, default))
            for f in folds
            if isinstance(f, dict)
        ]

    def mean(xs, d=0.0):
        return statistics.mean(xs) if xs else d

    def minv(xs, d=0.0):
        return min(xs) if xs else d

    def maxv(xs, d=0.0):
        return max(xs) if xs else d

    return {
        "trades_min": minv(vals("trades"), 0),
        "trades_mean": mean(vals("trades"), 0),

        "pf_mean": mean(vals("profit_factor"), 0.0),
        "winrate_mean": mean(vals("winrate"), 0.0),
        "dd_max": max(abs(v) for v in vals("max_drawdown_r", 999)) if folds else 999,
        "expectancy_mean": mean(vals("expectancy"), 0.0),
        "robust_score_mean": mean(vals("robust_score"), -1e9),
    }

def main():
    files = glob.glob(os.path.join(ROBUST_DIR, "robust_*_seed*.json"))
    if not files:
        print("[DIAG] No robust files found.")
        return

    rows_all: List[Dict[str, Any]] = []
    rows_top: List[Dict[str, Any]] = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        for i, rec in enumerate(data):
            m = normalize_from_folds(rec)
            row = {
                "file": os.path.basename(fp),
                "rank": i,
                **m,
            }
            rows_all.append(row)

            if i < 20:
                rows_top.append(row)

    def write_csv(name: str, rows: List[Dict[str, Any]]):
        if not rows:
            return
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"[DIAG] wrote -> {path}")

    write_csv("metrics_all.csv", rows_all)
    write_csv("metrics_top20.csv", rows_top)

    # Summary
    print("\n========== METRIC SUMMARY ==========")
    for k in [
        "trades_mean",
        "pf_mean",
        "winrate_mean",
        "dd_max",
        "expectancy_mean",
        "robust_score_mean",
    ]:
        all_vals = [r[k] for r in rows_all]
        top_vals = [r[k] for r in rows_top]

        print(f"\n{k}")
        print(f"  ALL : mean={statistics.mean(all_vals):.4f} "
              f"p50={statistics.median(all_vals):.4f}")
        print(f"  TOP : mean={statistics.mean(top_vals):.4f} "
              f"p50={statistics.median(top_vals):.4f}")

    print("\n[DIAG] Done.")

if __name__ == "__main__":
    main()
