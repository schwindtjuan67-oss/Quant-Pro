#!/usr/bin/env python3
# analysis_post_robust.py

from __future__ import annotations

import json
import os
import glob
import statistics
from typing import Dict, List, Any, Tuple

# ===============================
# CONFIG
# ===============================

ROBUST_DIR = "results/robust"
OUT_DIR = "results/promotions"

SEEDS = {1337, 2024, 777}
MIN_SEED_PASSES = 2          # 2 de 3 seeds
MIN_WINDOWS_PASSES = 2       # 2 de N ventanas

# Hard gates (FASE A)
GATES = {
    "min_trades": 300,
    "min_pf": 1.10,
    "min_winrate": 0.35,
    "max_dd_r": 12.0,
    "min_score_worst": -0.20,
}

TOP_K_PROMOTED = 30


# ===============================
# HELPERS
# ===============================

def _passes_hard_gates(r: Dict[str, Any]) -> bool:
    agg = r.get("agg", {})
    folds = r.get("folds", [])

    trades = agg.get("trades") or max(
        (f["metrics"].get("trades", 0) for f in folds),
        default=0
    )

    pf = agg.get("profit_factor", 0.0)
    winrate = agg.get("winrate", 0.0)
    max_dd = abs(agg.get("max_drawdown_r", 999))
    score_worst = agg.get("score_worst", -999)

    return (
        trades >= GATES["min_trades"] and
        pf >= GATES["min_pf"] and
        winrate >= GATES["min_winrate"] and
        max_dd <= GATES["max_dd_r"] and
        score_worst > GATES["min_score_worst"]
    )


def promotion_score(scores: List[float]) -> float:
    if not scores:
        return -1e9
    return (
        statistics.median(scores)
        - 0.50 * statistics.pstdev(scores)
        + 0.25 * min(scores)
    )


def parse_filename(path: str) -> Tuple[str, int]:
    """
    robust_2021-01_2021-12_seed1337.json
    -> ("2021-01_2021-12", 1337)
    """
    name = os.path.basename(path)
    parts = name.replace(".json", "").split("_seed")
    return parts[0].replace("robust_", ""), int(parts[1])


# ===============================
# MAIN LOGIC
# ===============================

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(ROBUST_DIR, "robust_*_seed*.json"))
    if not files:
        print("[POST] No robust files found.")
        return

    # window -> param_key -> seed -> record
    bucket: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

    for fp in files:
        window, seed = parse_filename(fp)
        if seed not in SEEDS:
            continue

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        for r in data:
            key = json.dumps(r["params"], sort_keys=True)
            bucket.setdefault(window, {}).setdefault(key, {})[seed] = r

    promoted: Dict[str, Dict[str, Any]] = {}

    for window, params_map in bucket.items():
        for key, seed_map in params_map.items():
            passed_seeds = []
            robust_scores = []

            for seed, rec in seed_map.items():
                if _passes_hard_gates(rec):
                    passed_seeds.append(seed)
                    agg = rec.get("agg", {}) or {}
                    robust_scores.append(float(agg.get("robust_score", rec.get("robust_score", -1e9))))


            if len(passed_seeds) >= MIN_SEED_PASSES:
                promoted.setdefault(key, {
                    "params": json.loads(key),
                    "windows": {},
                    "scores": []
                })
                promoted[key]["windows"][window] = {
                    "seeds": passed_seeds,
                    "scores": robust_scores,
                }
                promoted[key]["scores"].extend(robust_scores)

    # persistencia temporal
    final = []
    for p in promoted.values():
        if len(p["windows"]) >= MIN_WINDOWS_PASSES:
            p["promotion_score"] = promotion_score(p["scores"])
            final.append(p)

    final.sort(key=lambda x: x["promotion_score"], reverse=True)
    final = final[:TOP_K_PROMOTED]

    out_path = os.path.join(OUT_DIR, "faseA_promoted.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("=========================================")
    print(f"[POST] Promoted candidates: {len(final)}")
    print(f"[POST] Saved -> {out_path}")
    print("=========================================")

    if not final:
        print("[POST] No promotion. Pipeline should continue Fase A.")
    else:
        print("[POST] Promotion SUCCESS. Ready for Fase B.")


if __name__ == "__main__":
    main()

