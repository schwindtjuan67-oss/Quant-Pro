#!/usr/bin/env python3
# analysis/analysis_post_robust.py

from __future__ import annotations

import os
import sys
import json
import glob
import csv
import statistics
from typing import Dict, List, Any, Tuple, Optional

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.opt_space import phase_keys

# -------------------------------------------------
# CONFIG (GENERIC)
# -------------------------------------------------

ROBUST_DIR = "results/robust"
OUT_DIR = "results/promotions"

RULES_PATH = os.getenv(
    "PIPELINE_RULES",
    os.path.join("configs", "promotion_rules_A.json"),
)

WHY_REJECTED_CSV = os.path.join(
    OUT_DIR,
    f"why_rejected_{os.path.basename(RULES_PATH).replace('.json','')}.csv"
)

SEEDS: Optional[set] = None  # None = aceptar cualquier seed

# -------------------------------------------------
# LOAD RULES
# -------------------------------------------------

with open(RULES_PATH, "r", encoding="utf-8") as f:
    RULES = json.load(f)

FILTERS = RULES.get("filters", {})
PROMOTION = RULES.get("promotion", {})
PHASE = str(RULES.get("phase", "A")).upper()

MIN_SEED_PASSES = int(PROMOTION.get("min_seeds_passed", 1))
MIN_WINDOWS_PASSES = int(PROMOTION.get("min_windows_passed", 1))
TOP_K_PROMOTED = int(PROMOTION.get("top_k", 50))

# -------------------------------------------------
# METRIC NORMALIZATION (CANONICAL)
# -------------------------------------------------

def normalize_metrics(rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Fuente canónica: folds.
    Todas las métricas de decisión salen de acá.
    """
    folds = rec.get("folds", []) or []
    has_fold_metrics = any(
        isinstance(f, dict) and isinstance(f.get("metrics"), dict)
        for f in folds
    )

    def _vals(key: str, default: float = 0.0) -> List[float]:
        return [
            float(f.get("metrics", {}).get(key, default))
            for f in folds
            if isinstance(f, dict)
        ]

    if has_fold_metrics:
        trades = _vals("trades", 0)
        pf = _vals("profit_factor", 0.0)
        wr = _vals("winrate", 0.0)
        dd = [abs(v) for v in _vals("max_drawdown_r", 999)]
        exp = _vals("expectancy", 0.0)
        rs = _vals("robust_score", -1e9)
    else:
        agg = rec.get("agg", {}) or {}

        def _agg_float(key: str, default: float) -> float:
            val = agg.get(key, default)
            return float(val) if val is not None else float(default)

        trades_val = _agg_float("trades", 0)
        pf_val = _agg_float("profit_factor", 0.0)
        wr_val = _agg_float("winrate", 0.0)
        dd_val = abs(_agg_float("max_drawdown_r", 999))
        exp_val = _agg_float("expectancy", 0.0)
        rs_val = _agg_float("robust_score", -1e9)

        trades = [trades_val]
        pf = [pf_val]
        wr = [wr_val]
        dd = [dd_val]
        exp = [exp_val]
        rs = [rs_val]

    return {
        "trades_min": min(trades) if trades else 0,
        "trades_mean": statistics.mean(trades) if trades else 0,

        "pf_min": min(pf) if pf else 0.0,
        "pf_mean": statistics.mean(pf) if pf else 0.0,

        "winrate_min": min(wr) if wr else 0.0,
        "winrate_mean": statistics.mean(wr) if wr else 0.0,

        "dd_max": max(dd) if dd else 999,

        "expectancy_mean": statistics.mean(exp) if exp else 0.0,

        "robust_score_min": min(rs) if rs else -1e9,
        "robust_score_mean": statistics.mean(rs) if rs else -1e9,
    }

# -------------------------------------------------
# FILTER EVAL
# -------------------------------------------------

def _passes_filters_verbose(
    metrics: Dict[str, float]
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    for metric, cfg in FILTERS.items():
        if PHASE == "A" and metric not in {"trades_min", "dd_max"}:
            continue

        val = metrics.get(metric)
        if val is None:
            reasons.append(f"{metric}:missing")
            continue

        if "min" in cfg and val < cfg["min"]:
            reasons.append(f"{metric}<{cfg['min']}")
        if "max" in cfg and val > cfg["max"]:
            reasons.append(f"{metric}>{cfg['max']}")

    return (len(reasons) == 0), reasons

def promotion_score(scores: List[float]) -> float:
    if not scores:
        return -1e9
    if PHASE == "A":
        return statistics.median(scores)
    return (
        statistics.median(scores)
        - 0.50 * statistics.pstdev(scores)
        + 0.25 * min(scores)
    )

def phase_a_score(metrics: Dict[str, float]) -> float:
    return metrics["trades_mean"] - 0.01 * metrics["dd_max"]

def parse_filename(path: str) -> Tuple[str, int]:
    """
    robust_2021-01_2021-12_seed1337.json
    -> ("2021-01_2021-12", 1337)
    """
    name = os.path.basename(path)
    parts = name.replace(".json", "").split("_seed")
    return parts[0].replace("robust_", ""), int(parts[1])

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(ROBUST_DIR, "robust_*_seed*.json"))
    if not files:
        print("[POST] No robust files found.")
        return

    frozen_keys = phase_keys(PHASE)

    # window -> param_key -> seed -> record
    bucket: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

    skipped_seed = 0
    skipped_phase = 0

    # ---------------------------
    # LOAD + BUCKET
    # ---------------------------

    for fp in files:
        window, seed = parse_filename(fp)

        if SEEDS is not None and seed not in SEEDS:
            skipped_seed += 1
            continue

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        for r in data:
            if not isinstance(r, dict):
                continue

            meta = r.get("meta", {}) or {}
            ph = str(meta.get("pipeline_phase", "")).strip().upper()
            if ph != PHASE:
                skipped_phase += 1
                continue

            params = r.get("params")
            if not isinstance(params, dict) or not params:
                continue

            key = json.dumps(params, sort_keys=True)
            bucket.setdefault(window, {}).setdefault(key, {})[seed] = r

    # ---------------------------
    # EVALUATION
    # ---------------------------

    promoted: Dict[str, Dict[str, Any]] = {}
    rejected_rows: List[List[Any]] = []

    for window, params_map in bucket.items():
        for key, seed_map in params_map.items():
            passed_seeds: List[int] = []
            scores: List[float] = []

            for seed, rec in seed_map.items():
                metrics = normalize_metrics(rec)
                is_degenerate = (
                    metrics["trades_min"] == 0
                    and metrics["pf_mean"] == 0.0
                    and metrics["winrate_mean"] == 0.0
                    and metrics["dd_max"] == 999
                    and metrics["expectancy_mean"] == 0.0
                    and metrics["robust_score_mean"] <= -1e9
                )
                if is_degenerate:
                    rejected_rows.append([
                        window,
                        seed,
                        key,
                        "metrics_degenerate",
                        metrics["trades_min"],
                        metrics["pf_mean"],
                        metrics["winrate_mean"],
                        metrics["dd_max"],
                        metrics["expectancy_mean"],
                        metrics["robust_score_mean"],
                    ])
                    continue

                ok, reasons = _passes_filters_verbose(metrics)

                if ok:
                    passed_seeds.append(seed)
                    score = (
                        phase_a_score(metrics)
                        if PHASE == "A"
                        else metrics["robust_score_mean"]
                    )
                    scores.append(score)
                else:
                    rejected_rows.append([
                        window,
                        seed,
                        key,
                        ";".join(reasons),
                        metrics["trades_min"],
                        metrics["pf_mean"],
                        metrics["winrate_mean"],
                        metrics["dd_max"],
                        metrics["expectancy_mean"],
                        metrics["robust_score_mean"],
                    ])

            if len(passed_seeds) >= MIN_SEED_PASSES:
                promoted.setdefault(key, {
                    "params": json.loads(key),
                    "windows": {},
                    "scores": [],
                    f"phase{PHASE}_frozen_keys": frozen_keys,
                    "phase": PHASE,
                })
                promoted[key]["windows"][window] = {
                    "seeds": sorted(set(passed_seeds)),
                    "scores": scores,
                }
                promoted[key]["scores"].extend(scores)

    # ---------------------------
    # WINDOW GATE + SCORE
    # ---------------------------

    final: List[Dict[str, Any]] = []
    for p in promoted.values():
        if len(p["windows"]) >= MIN_WINDOWS_PASSES:
            p["promotion_score"] = promotion_score(p["scores"])
            final.append(p)

    final.sort(key=lambda x: float(x.get("promotion_score", -1e9)), reverse=True)
    final = final[:TOP_K_PROMOTED]

    # ---------------------------
    # WRITE OUTPUTS
    # ---------------------------

    out_json = os.path.join(
        OUT_DIR,
        f"fase{PHASE}_promoted.json"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    # 👉 FIX CRÍTICO: writer SIEMPRE dentro del with
    with open(WHY_REJECTED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window",
            "seed",
            "params_key",
            "failed_rules",
            "trades",
            "profit_factor",
            "winrate",
            "max_drawdown_r",
            "expectancy",
            "robust_score",
        ])
        if rejected_rows:
            writer.writerows(rejected_rows)

    # ---------------------------
    # LOG
    # ---------------------------

    print("=========================================")
    print(f"[POST] Phase {PHASE} promoted: {len(final)}")
    print(f"[POST] Saved -> {out_json}")
    print(f"[POST] Rejection audit -> {WHY_REJECTED_CSV}")
    if skipped_seed:
        print(f"[POST][INFO] Skipped {skipped_seed} files by SEEDS")
    if skipped_phase:
        print(f"[POST][WARN] Skipped {skipped_phase} records from other phases")
    print("=========================================")

    if not final:
        print(f"[POST] No promotion. Pipeline should continue Phase {PHASE}.")
    else:
        print(f"[POST] Promotion SUCCESS. Ready for next phase.")

# -------------------------------------------------

if __name__ == "__main__":
    main()


