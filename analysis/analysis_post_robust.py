#!/usr/bin/env python3
# analysis/analysis_post_robust.py
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Defaults (repo-relative)
# -----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_ROBUST_DIR = os.path.join(ROOT, "results", "robust")
DEFAULT_PROMO_DIR = os.path.join(ROOT, "results", "promotions")
DEFAULT_PROMO_FILE = os.path.join(DEFAULT_PROMO_DIR, "faseA_promoted.json")
DEFAULT_SUMMARY_FILE = os.path.join(DEFAULT_ROBUST_DIR, "faseA_summary.json")
DEFAULT_RULES_FILE = os.path.join(HERE, "promotion_rules.json")


# -----------------------------
# Utils
# -----------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_params_key(params: Dict[str, Any]) -> str:
    try:
        return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(params)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / float(len(xs))


def _std(xs: List[float]) -> float:
    # sample std (ddof=1) if possible; else 0
    n = len(xs)
    if n <= 1:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / float(n - 1)
    return var ** 0.5


# robust_2019-01_2020-12_seed1337.json
# robust_2019-01_2020-12_seed1337_20251231_235959.json
_RE_RUN = re.compile(r"robust_(?P<window>\d{4}-\d{2}_\d{4}-\d{2})_seed(?P<seed>\d+)", re.IGNORECASE)


def _parse_window_seed(filename: str) -> Tuple[Optional[str], Optional[int]]:
    m = _RE_RUN.search(filename)
    if not m:
        return None, None
    return m.group("window"), _safe_int(m.group("seed"), 0)


def _list_robust_json_files(robust_dir: str) -> List[str]:
    if not os.path.isdir(robust_dir):
        return []
    out: List[str] = []
    for name in os.listdir(robust_dir):
        if not name.lower().endswith(".json"):
            continue
        # avoid re-reading our outputs
        if name.lower().startswith("fasea_summary") or name.lower().startswith("fasea_promoted"):
            continue
        if name.lower().endswith("_summary.json") or name.lower().endswith("_promoted.json"):
            continue
        if name.lower().startswith("robust_"):
            out.append(os.path.join(robust_dir, name))
    return sorted(out)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass
class RunEvidence:
    source_file: str
    window: Optional[str]
    seed: Optional[int]
    passed: bool
    fail_reason: str
    robust_score: float
    score_worst: float
    score_mean: float
    trades_total: int
    equity_total_r: float
    dd_max_r: float
    pf_mean: float
    winrate_mean: float
    sortino_mean: float
    expectancy_mean: float


def _extract_run_evidence(source_file: str, item: Dict[str, Any]) -> RunEvidence:
    window, seed = _parse_window_seed(os.path.basename(source_file))

    passed = bool(item.get("passed", False))
    fail_reason = str(item.get("fail_reason", "") or "")

    agg = item.get("agg", {}) or {}
    robust_score = _safe_float(agg.get("robust_score", item.get("robust_score", -1e9)), -1e9)
    score_worst = _safe_float(agg.get("score_worst", -1e9), -1e9)
    score_mean = _safe_float(agg.get("score_mean", 0.0), 0.0)

    folds = item.get("folds", []) or []

    trades_list: List[int] = []
    equity_list: List[float] = []
    dd_list: List[float] = []
    pf_list: List[float] = []
    wr_list: List[float] = []
    sort_list: List[float] = []
    exp_list: List[float] = []

    for fr in folds:
        m = (fr or {}).get("metrics", {}) or {}
        trades_list.append(_safe_int(m.get("trades", 0), 0))
        equity_list.append(_safe_float(m.get("equity_r", 0.0), 0.0))
        dd_list.append(abs(_safe_float(m.get("max_drawdown_r", 0.0), 0.0)))
        pf_list.append(_safe_float(m.get("profit_factor", 0.0), 0.0))
        wr_list.append(_safe_float(m.get("winrate", 0.0), 0.0))
        sort_list.append(_safe_float(m.get("sortino", 0.0), 0.0))
        exp_list.append(_safe_float(m.get("expectancy", 0.0), 0.0))

    trades_total = sum(trades_list)
    equity_total_r = sum(equity_list)
    dd_max_r = max(dd_list) if dd_list else 0.0
    pf_mean = _mean(pf_list)
    winrate_mean = _mean(wr_list)
    sortino_mean = _mean(sort_list)
    expectancy_mean = _mean(exp_list)

    return RunEvidence(
        source_file=os.path.basename(source_file),
        window=window,
        seed=seed,
        passed=passed,
        fail_reason=fail_reason,
        robust_score=robust_score,
        score_worst=score_worst,
        score_mean=score_mean,
        trades_total=trades_total,
        equity_total_r=equity_total_r,
        dd_max_r=dd_max_r,
        pf_mean=pf_mean,
        winrate_mean=winrate_mean,
        sortino_mean=sortino_mean,
        expectancy_mean=expectancy_mean,
    )


def _default_rules() -> Dict[str, Any]:
    # NOTE: tus folds test suelen ser cortos (min_test=2000 velas).
    # Por eso el umbral de trades acá es moderado y lo podés endurecer en B.
    return {
        "phase": "A_to_B",
        "top_n": 20,

        # Coverage: robustness multi-seed y multi-ventana
        "min_runs": 3,
        "min_unique_seeds": 2,
        "min_unique_windows": 2,

        # Consistencia
        "min_pass_rate": 0.60,
        "min_robust_mean": 0.0,

        # Actividad (scalper): en A el test total ~ 5 folds * 2000 velas ~= ~7 días.
        "min_trades_total_mean": 120,
        "min_trades_total_min": 60,

        # Riesgo/edge (en R, según grid_metrics.py)
        "max_dd_max_mean": 20.0,
        "min_pf_mean": 1.05,
        "min_winrate_mean": 0.35,
        "min_expectancy_mean": 0.0,
    }


def _load_rules(path: str) -> Dict[str, Any]:
    if path and os.path.isfile(path):
        try:
            obj = _load_json(path)
            if isinstance(obj, dict):
                base = _default_rules()
                base.update(obj)
                return base
        except Exception:
            pass
    return _default_rules()


def _passes_rules(agg: Dict[str, Any], rules: Dict[str, Any]) -> Tuple[bool, str]:
    # Coverage
    if agg["runs"] < int(rules.get("min_runs", 1)):
        return False, "min_runs"
    if agg["unique_seeds"] < int(rules.get("min_unique_seeds", 1)):
        return False, "min_unique_seeds"
    if agg["unique_windows"] < int(rules.get("min_unique_windows", 1)):
        return False, "min_unique_windows"

    # Consistencia
    if agg["pass_rate"] < float(rules.get("min_pass_rate", 0.0)):
        return False, "min_pass_rate"
    if agg["robust_mean"] < float(rules.get("min_robust_mean", -1e18)):
        return False, "min_robust_mean"

    # Actividad
    if agg["trades_total_mean"] < int(rules.get("min_trades_total_mean", 0)):
        return False, "min_trades_total_mean"
    if agg["trades_total_min"] < int(rules.get("min_trades_total_min", 0)):
        return False, "min_trades_total_min"

    # Edge / riesgo
    if agg["dd_max_mean"] > float(rules.get("max_dd_max_mean", 1e18)):
        return False, "max_dd_max_mean"
    if agg["pf_mean"] < float(rules.get("min_pf_mean", 0.0)):
        return False, "min_pf_mean"
    if agg["winrate_mean"] < float(rules.get("min_winrate_mean", 0.0)):
        return False, "min_winrate_mean"
    if agg["expectancy_mean"] < float(rules.get("min_expectancy_mean", -1e18)):
        return False, "min_expectancy_mean"

    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser("analysis_post_robust")
    ap.add_argument("--robust-dir", default=DEFAULT_ROBUST_DIR, help="folder con robust_*.json")
    ap.add_argument("--promo-out", default=DEFAULT_PROMO_FILE, help="output promotions json")
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY_FILE, help="output summary json")
    ap.add_argument("--rules", default=DEFAULT_RULES_FILE, help="promotion_rules.json (opcional)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    robust_dir = args.robust_dir
    promo_out = args.promo_out
    summary_out = args.summary_out
    rules_path = args.rules

    files = _list_robust_json_files(robust_dir)
    if not files:
        print(f"[POST] No robust files found in: {robust_dir}")
        # No input => no promotions
        if os.path.isfile(promo_out):
            try:
                os.remove(promo_out)
            except Exception:
                pass
        return

    rules = _load_rules(rules_path)

    # -----------------------------
    # Load + group
    # -----------------------------
    groups: Dict[str, Dict[str, Any]] = {}  # key -> {params, runs:[RunEvidence]}

    bad_files = 0
    total_items = 0

    for fp in files:
        try:
            data = _load_json(fp)
        except Exception as e:
            bad_files += 1
            if args.verbose:
                print(f"[POST][WARN] Failed reading {fp}: {e!r}")
            continue

        if not isinstance(data, list):
            bad_files += 1
            if args.verbose:
                print(f"[POST][WARN] Unexpected JSON (not list): {fp}")
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            params = item.get("params", {}) or {}
            if not isinstance(params, dict):
                continue

            key = _stable_params_key(params)
            ev = _extract_run_evidence(fp, item)

            g = groups.get(key)
            if g is None:
                g = {"params": params, "runs": []}
                groups[key] = g
            g["runs"].append(ev)
            total_items += 1

    # -----------------------------
    # Aggregate groups
    # -----------------------------
    rows: List[Dict[str, Any]] = []
    for key, g in groups.items():
        runs: List[RunEvidence] = g["runs"]

        runs_n = len(runs)
        pass_n = sum(1 for r in runs if r.passed)
        pass_rate = (pass_n / runs_n) if runs_n else 0.0

        uniq_seeds = sorted({r.seed for r in runs if r.seed is not None})
        uniq_windows = sorted({r.window for r in runs if r.window})

        robust_scores = [r.robust_score for r in runs]
        score_worsts = [r.score_worst for r in runs]
        trades_totals = [r.trades_total for r in runs]
        dd_maxs = [r.dd_max_r for r in runs]
        pf_means = [r.pf_mean for r in runs]
        wr_means = [r.winrate_mean for r in runs]
        sort_means = [r.sortino_mean for r in runs]
        exp_means = [r.expectancy_mean for r in runs]

        agg = {
            "key": key,
            "runs": runs_n,
            "pass_runs": pass_n,
            "pass_rate": float(pass_rate),
            "unique_seeds": len(uniq_seeds),
            "unique_windows": len(uniq_windows),
            "robust_mean": float(_mean(robust_scores)),
            "robust_std": float(_std(robust_scores)),
            "robust_min": float(min(robust_scores)) if robust_scores else -1e9,
            "score_worst_mean": float(_mean(score_worsts)),
            "trades_total_mean": float(_mean([float(x) for x in trades_totals])),
            "trades_total_min": int(min(trades_totals)) if trades_totals else 0,
            "dd_max_mean": float(_mean(dd_maxs)),
            "pf_mean": float(_mean(pf_means)),
            "winrate_mean": float(_mean(wr_means)),
            "sortino_mean": float(_mean(sort_means)),
            "expectancy_mean": float(_mean(exp_means)),
            "seeds": uniq_seeds,
            "windows": uniq_windows,
        }

        ok, reason = _passes_rules(agg, rules)

        rows.append({
            "params": g["params"],
            "agg": agg,
            "promote_ok": bool(ok),
            "promote_reason": reason,
            "evidence": [r.__dict__ for r in runs],
        })

    # -----------------------------
    # Sort + write summary
    # -----------------------------
    rows.sort(
        key=lambda r: (
            r["agg"]["robust_mean"],
            r["agg"]["robust_min"],
            r["agg"]["pass_rate"],
            r["agg"]["trades_total_mean"],
            -r["agg"]["dd_max_mean"],
        ),
        reverse=True,
    )

    summary = {
        "meta": {
            "generated_at_utc": _now_utc_iso(),
            "robust_dir": os.path.abspath(robust_dir),
            "files_used": [os.path.basename(x) for x in files],
            "bad_files": bad_files,
            "items_read": total_items,
            "unique_param_keys": len(groups),
            "rules_path": os.path.abspath(rules_path) if rules_path else None,
            "rules": rules,
        },
        "rows": rows,
    }
    _save_json(summary_out, summary)

    # -----------------------------
    # Promotion output (ONLY if any)
    # -----------------------------
    promoted = [r for r in rows if r.get("promote_ok")]
    top_n = int(rules.get("top_n", 20) or 20)
    promoted = promoted[:max(1, top_n)] if promoted else []

    if not promoted:
        # if no promotion, remove the file if it existed
        if os.path.isfile(promo_out):
            try:
                os.remove(promo_out)
            except Exception:
                pass
        print(f"[POST] Summary saved: {summary_out}")
        print("[POST] No promotion (faseA_promoted.json NOT created).")
        return

    payload = {
        "meta": {
            "generated_at_utc": _now_utc_iso(),
            "phase": str(rules.get("phase", "A_to_B")),
            "rules": rules,
            "summary_file": os.path.basename(summary_out),
        },
        "promoted": [
            {
                "params": r["params"],
                "agg": r["agg"],
                "evidence": r["evidence"],
            }
            for r in promoted
        ],
    }
    _save_json(promo_out, payload)

    print(f"[POST] Summary saved: {summary_out}")
    print(f"[POST] PROMOTION SUCCESS: {promo_out}")
    if args.verbose:
        print(f"[POST] Promoted candidates: {len(promoted)}")


if __name__ == "__main__":
    main()

