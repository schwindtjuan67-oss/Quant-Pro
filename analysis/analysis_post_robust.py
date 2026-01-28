#!/usr/bin/env python3
# analysis/analysis_post_robust.py
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

_PHASE_KEYS = {
    "A": {
        "ema_fast",
        "ema_slow",
        "delta_threshold",
        "delta_rolling_sec",
    },
    "B": {
        "atr_len",
        "sl_atr_mult",
        "tp_atr_mult",
        "rr_min",
        "cooldown_sec",
        "max_trades_day",
        "use_time_filter",
        "hour_start",
        "hour_end",
    },
}


def _normalize_phase(phase: Optional[str] = None) -> str:
    p = (phase or os.getenv("PIPELINE_PHASE", "") or "").strip().upper()
    if not p:
        return "FULL"
    if p in ("C", "EVAL"):
        return "C"
    if p in ("A", "B"):
        return p
    return "FULL"


def phase_keys(phase: Optional[str] = None) -> List[str]:
    p = _normalize_phase(phase)
    if p in ("FULL", "C"):
        return []
    return sorted(list(_PHASE_KEYS.get(p, set())))


def _load_rules(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    if not isinstance(rules, dict):
        raise ValueError(f"Rules file must be an object: {path}")
    return rules


def iter_robust_records(data: Any, source_path: str) -> List[Dict[str, Any]]:
    basename = os.path.basename(source_path)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]

    if isinstance(data, dict):
        required = {"params", "passed", "robust_score", "fail_reason"}
        if not required.issubset(data.keys()):
            raise ValueError(f"Unknown robust schema in {basename}")

        params = data.get("params")
        passed = data.get("passed")
        scores = data.get("robust_score")
        fail_reason = data.get("fail_reason")
        agg = data.get("agg")
        folds = data.get("folds")
        meta = data.get("meta", {})

        if not all(isinstance(x, list) for x in (params, passed, scores, fail_reason)):
            raise ValueError(
                f"Robust schema must include list params/passed/robust_score/fail_reason: {basename}"
            )

        if not (len(params) == len(passed) == len(scores) == len(fail_reason)):
            raise ValueError(
                f"Robust schema length mismatch in {basename}: "
                f"params={len(params)} passed={len(passed)} robust_score={len(scores)} "
                f"fail_reason={len(fail_reason)}"
            )

        if len(params) == 0:
            raise ValueError(f"Robust schema empty arrays in {basename}")

        records: List[Dict[str, Any]] = []
        for i, param in enumerate(params):
            score_val = scores[i]
            try:
                score_val = float(score_val)
            except (TypeError, ValueError):
                score_val = score_val

            record_agg: Dict[str, Any] = {}
            if isinstance(agg, list) and len(agg) == len(params) and isinstance(agg[i], dict):
                record_agg = agg[i]

            record_folds: List[Any] = []
            if isinstance(folds, list) and len(folds) == len(params) and isinstance(folds[i], list):
                record_folds = folds[i]

            record: Dict[str, Any] = {
                "params": param,
                "passed": bool(passed[i]),
                "robust_score": score_val,
                "fail_reason": fail_reason[i],
                "agg": record_agg,
                "folds": record_folds,
                "meta": meta,
            }
            records.append(record)
        return records

    return []


def parse_filename(path: str) -> Optional[Tuple[str, int]]:
    name = os.path.basename(path)
    match = re.match(r"^robust_(?P<window>.+)_seed(?P<seed>\d+)(?P<suffix>.*)\.json$", name)
    if not match:
        return None
    return match.group("window"), int(match.group("seed"))


def normalize_metrics(rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Fuente canónica: folds.
    Todas las métricas de decisión salen de acá.
    """
    folds = rec.get("folds", []) or []

    def _vals(key: str, default: float = 0.0) -> List[float]:
        return [
            float((f.get("metrics", {}) or {}).get(key, default))
            for f in folds
            if isinstance(f, dict)
        ]

    trades = _vals("trades", 0.0)
    pf = _vals("profit_factor", 0.0)
    wr = _vals("winrate", 0.0)
    dd = [abs(v) for v in _vals("max_drawdown_r", 999.0)]
    exp = _vals("expectancy", 0.0)
    rs = _vals("robust_score", -1e9)

    return {
        "trades_min": min(trades) if trades else 0.0,
        "trades_mean": statistics.mean(trades) if trades else 0.0,
        "pf_min": min(pf) if pf else 0.0,
        "pf_mean": statistics.mean(pf) if pf else 0.0,
        "winrate_min": min(wr) if wr else 0.0,
        "winrate_mean": statistics.mean(wr) if wr else 0.0,
        "dd_max": max(dd) if dd else 999.0,
        "expectancy_mean": statistics.mean(exp) if exp else 0.0,
        "robust_score_min": min(rs) if rs else -1e9,
        "robust_score_mean": statistics.mean(rs) if rs else -1e9,
    }


def _passes_filters_verbose(
    metrics: Dict[str, float],
    filters: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    thresholds = (filters or {}).get("thresholds", {}) or {}

    for metric, cfg in thresholds.items():
        val = metrics.get(metric)
        if val is None:
            reasons.append(f"{metric}:missing")
            continue
        if "min" in cfg and val < cfg["min"]:
            reasons.append(f"{metric}<{cfg['min']}")
        if "max" in cfg and val > cfg["max"]:
            reasons.append(f"{metric}>{cfg['max']}")

    if filters.get("reject_degenerate"):
        if metrics.get("trades_min", 0.0) <= 0:
            reasons.append("trades_min<=0")

    return (len(reasons) == 0), reasons


def promotion_score(scores: List[float]) -> float:
    if not scores:
        return -1e9
    return (
        statistics.median(scores)
        - 0.50 * statistics.pstdev(scores)
        + 0.25 * min(scores)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Post-analysis robust promotion")
    ap.add_argument("--root", default=".", help="Repo root for results/ and configs/")
    ap.add_argument("--rules", default=None, help="Override promotion rules JSON")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    root = os.path.abspath(args.root)

    robust_dir = os.path.join(root, "results", "robust")
    out_dir = os.path.join(root, "results", "promotions")
    os.makedirs(out_dir, exist_ok=True)

    rules_path = args.rules
    if not rules_path:
        primary_rules = os.path.join(root, "configs", "promotion_rules_A.json")
        fallback_rules = os.path.join(os.path.dirname(__file__), "promotion_rules.json")
        if os.path.exists(primary_rules):
            rules_path = primary_rules
        elif os.getenv("PIPELINE_RULES"):
            rules_path = os.getenv("PIPELINE_RULES")
        elif os.path.exists(fallback_rules):
            rules_path = fallback_rules
        else:
            raise ValueError(
                "Rules file not found. Expected configs/promotion_rules_A.json or analysis/promotion_rules.json"
            )

    rules = _load_rules(rules_path)
    phase = str(rules.get("phase", "A")).upper()
    filters = rules.get("filters", {}) or {}
    promotion = rules.get("promotion", {}) or {}

    min_seed_passes = int(promotion.get("min_seeds_passed", 1))
    min_windows_passes = int(promotion.get("min_windows_passed", 1))
    top_k_promoted = int(promotion.get("top_k", 50))

    why_rejected_csv = os.path.join(
        out_dir,
        f"why_rejected_{os.path.basename(rules_path).replace('.json', '')}.csv",
    )
    write_rejections = bool((rules.get("audit", {}) or {}).get("write_why_rejected_csv", True))

    files = glob.glob(os.path.join(robust_dir, "robust_*_seed*.json"))

    frozen_keys = phase_keys(phase)
    bucket: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    skipped_phase = 0

    for fp in files:
        parsed = parse_filename(fp)
        if not parsed:
            print(f"[POST][WARN] skipping unrecognized robust filename: {os.path.basename(fp)}")
            continue
        window, seed = parsed

        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed reading robust file {os.path.basename(fp)}: {exc}") from exc

        records = iter_robust_records(data, fp)
        for r in records:
            if not isinstance(r, dict):
                continue

            meta = r.get("meta", {}) or {}
            ph = str(meta.get("pipeline_phase", "")).strip().upper()
            if ph and ph != phase:
                skipped_phase += 1
                continue

            params = r.get("params")
            if not isinstance(params, dict) or not params:
                continue

            key = json.dumps(params, sort_keys=True)
            bucket.setdefault(window, {}).setdefault(key, {})[seed] = r

    promoted: Dict[str, Dict[str, Any]] = {}
    rejected_rows: List[List[Any]] = []

    for window, params_map in bucket.items():
        for key, seed_map in params_map.items():
            passed_seeds: List[int] = []
            scores: List[float] = []

            for seed, rec in seed_map.items():
                metrics = normalize_metrics(rec)
                ok, reasons = _passes_filters_verbose(metrics, filters)

                if ok:
                    passed_seeds.append(seed)
                    scores.append(metrics["robust_score_mean"])
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

            if len(passed_seeds) >= min_seed_passes:
                promoted.setdefault(key, {
                    "params": json.loads(key),
                    "windows": {},
                    "scores": [],
                    "passed_seeds": [],
                    f"phase{phase}_frozen_keys": frozen_keys,
                    "phase": phase,
                })
                promoted[key]["windows"][window] = {
                    "passed_seeds": sorted(set(passed_seeds)),
                    "scores": scores,
                }
                promoted[key]["scores"].extend(scores)
                promoted[key]["passed_seeds"].extend(passed_seeds)

    final: List[Dict[str, Any]] = []
    for p in promoted.values():
        if len(p["windows"]) >= min_windows_passes:
            p["passed_seeds"] = sorted(set(p.get("passed_seeds", [])))
            p["promotion_score"] = promotion_score(p["scores"])
            final.append(p)

    final.sort(key=lambda x: float(x.get("promotion_score", -1e9)), reverse=True)
    final = final[:top_k_promoted]

    out_json = os.path.join(out_dir, f"fase{phase}_promoted.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    if write_rejections:
        with open(why_rejected_csv, "w", newline="", encoding="utf-8") as f:
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

    print("=========================================")
    print(f"[POST] Phase {phase} promoted: {len(final)}")
    print(f"[POST] Saved -> {out_json}")
    if write_rejections:
        print(f"[POST] Rejection audit -> {why_rejected_csv}")
    if skipped_phase:
        print(f"[POST][WARN] Skipped {skipped_phase} records from other phases")
    print("=========================================")

    if not final:
        print(f"[POST] No promotion. Pipeline should continue Phase {phase}.")
    else:
        print("[POST] Promotion SUCCESS. Ready for next phase.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[POST][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
