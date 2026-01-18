#!/usr/bin/env python3
# analysis/tools/analyze_thresholds.py

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Dict, List

WHY_REJECTED = "results/promotions/why_rejected_promotion_rules_A.csv"
OUT_STATS = "results/promotions/rejection_stats.csv"
OUT_SUGGESTED = "results/promotions/suggested_thresholds.json"

# -------------------------------------------------
# CANONICAL COLUMN MAP (NEW PIPELINE)
# -------------------------------------------------

METRIC_COLUMNS = {
    "trades": "trades",
    "profit_factor": "profit_factor",
    "winrate": "winrate",
    "max_drawdown_r": "max_drawdown_r",
    "expectancy": "expectancy",
    "robust_score": "robust_score",
}

# -------------------------------------------------
# LOAD
# -------------------------------------------------

def load_rejected() -> List[Dict[str, str]]:
    with open(WHY_REJECTED, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------

def main() -> None:
    rows = load_rejected()
    if not rows:
        print("[THRESHOLDS] No rejected rows found.")
        return

    failed_counter = Counter()
    values: Dict[str, List[float]] = defaultdict(list)

    for r in rows:
        # failed rules
        failed = r.get("failed_rules", "")
        for rule in failed.split(";"):
            if rule:
                failed_counter[rule] += 1

        # metric values
        for metric, col in METRIC_COLUMNS.items():
            v = r.get(col)
            if v is None or v == "":
                continue
            try:
                values[metric].append(float(v))
            except Exception:
                pass

    # -------------------------------------------------
    # STATS CSV
    # -------------------------------------------------

    with open(OUT_STATS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "min", "p25", "median", "p75", "max"])
        for m, vals in values.items():
            if not vals:
                continue
            vals_sorted = sorted(vals)
            w.writerow([
                m,
                len(vals_sorted),
                min(vals_sorted),
                percentile(vals_sorted, 25),
                statistics.median(vals_sorted),
                percentile(vals_sorted, 75),
                max(vals_sorted),
            ])

    # -------------------------------------------------
    # SUGGESTED THRESHOLDS (DATA-DRIVEN)
    # -------------------------------------------------

    suggested_filters = {}

    for m, vals in values.items():
        if not vals:
            continue

        if m == "max_drawdown_r":
            suggested_filters[m] = {
                "max": percentile(sorted(vals), 75)
            }
        else:
            suggested_filters[m] = {
                "min": percentile(sorted(vals), 25)
            }

    with open(OUT_SUGGESTED, "w", encoding="utf-8") as f:
        json.dump({
            "source": os.path.basename(WHY_REJECTED),
            "suggested_filters": suggested_filters,
            "failed_rules_rank": dict(failed_counter.most_common())
        }, f, indent=2)

    print("====================================")
    print(f"[THRESHOLDS] Analyzed: {WHY_REJECTED}")
    print(f"[THRESHOLDS] Stats -> {OUT_STATS}")
    print(f"[THRESHOLDS] Suggested -> {OUT_SUGGESTED}")
    print("====================================")

# -------------------------------------------------

def percentile(vals: List[float], p: int) -> float:
    if not vals:
        return 0.0
    k = int(len(vals) * p / 100)
    return vals[min(k, len(vals) - 1)]

# -------------------------------------------------

if __name__ == "__main__":
    main()



