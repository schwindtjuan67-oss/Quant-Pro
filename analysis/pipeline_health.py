from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - py<3.9 fallback
    ZoneInfo = None

from analysis.opt_space import default_param_space


EXPECTED_FOLD_METRICS = (
    "trades",
    "profit_factor",
    "winrate",
    "max_drawdown_r",
    "expectancy",
    "equity_r",
    "sortino",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _local_now() -> datetime:
    if ZoneInfo is None:
        return datetime.now()
    try:
        return datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))
    except Exception:
        return datetime.now()


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _space_keys() -> List[str]:
    return sorted(list(default_param_space().keys()))


def validate_robust_json(path: str) -> Dict[str, Any]:
    errors: List[str] = []
    meta: Dict[str, Any] = {}
    counts = {"samples": 0, "passed_count": 0}
    sanity = {
        "arrays_consistent": False,
        "folds_metrics_ok": False,
    }

    if not path or not os.path.exists(path):
        return {
            "ok": False,
            "errors": [f"missing:{path}"],
            "meta": meta,
            "counts": counts,
            "sanity": sanity,
        }

    try:
        payload = _read_json(path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"read_error:{exc!r}"],
            "meta": meta,
            "counts": counts,
            "sanity": sanity,
        }

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "errors": ["payload_not_dict"],
            "meta": meta,
            "counts": counts,
            "sanity": sanity,
        }

    meta_raw = payload.get("meta")
    if isinstance(meta_raw, dict):
        meta = {
            "gates": meta_raw.get("gates"),
            "space_keys": meta_raw.get("space_keys"),
            "score_fallback": meta_raw.get("score_fallback"),
        }

    params_list = _safe_list(payload.get("params"))
    passed_list = _safe_list(payload.get("passed"))
    fail_reason_list = _safe_list(payload.get("fail_reason"))
    robust_score_list = _safe_list(payload.get("robust_score"))
    folds_list = _safe_list(payload.get("folds"))

    lengths = {
        "params": len(params_list),
        "passed": len(passed_list),
        "fail_reason": len(fail_reason_list),
        "robust_score": len(robust_score_list),
        "folds": len(folds_list),
    }
    if len(set(lengths.values())) == 1:
        sanity["arrays_consistent"] = True
    else:
        errors.append(f"length_mismatch:{lengths}")

    counts["samples"] = lengths["params"]
    counts["passed_count"] = sum(1 for x in passed_list if bool(x))

    folds_ok = True
    for idx, folds in enumerate(folds_list):
        if not isinstance(folds, list):
            folds_ok = False
            errors.append(f"folds_not_list:{idx}")
            continue
        for fold in folds:
            if not isinstance(fold, dict):
                folds_ok = False
                errors.append(f"fold_not_dict:{idx}")
                continue
            metrics = fold.get("metrics") or {}
            if not isinstance(metrics, dict):
                folds_ok = False
                errors.append(f"metrics_not_dict:{idx}")
                continue
            missing = [k for k in EXPECTED_FOLD_METRICS if k not in metrics]
            if missing:
                folds_ok = False
                errors.append(f"metrics_missing:{idx}:{','.join(missing)}")

    sanity["folds_metrics_ok"] = folds_ok

    ok = len(errors) == 0 and sanity["arrays_consistent"]
    return {
        "ok": ok,
        "errors": errors,
        "meta": meta,
        "counts": counts,
        "sanity": sanity,
    }


def _extract_sources(entry: Dict[str, Any]) -> Dict[str, Any]:
    source: Dict[str, Any] = {}
    windows = entry.get("windows")
    if isinstance(windows, dict):
        source_windows: Dict[str, Any] = {}
        for window, wobj in windows.items():
            if isinstance(wobj, dict):
                seeds = wobj.get("seeds")
                if isinstance(seeds, list):
                    source_windows[str(window)] = sorted({int(s) for s in seeds if str(s).isdigit()})
        if source_windows:
            source["windows"] = source_windows
    if "source" in entry:
        source["source"] = entry.get("source")
    if "robust_path" in entry:
        source["robust_path"] = entry.get("robust_path")
    return source


def validate_faseA_promoted(path: str) -> Dict[str, Any]:
    errors: List[str] = []
    items: List[Dict[str, Any]] = []
    count = 0
    if not path or not os.path.exists(path):
        return {"ok": False, "errors": [f"missing:{path}"], "count": 0, "items": []}
    try:
        payload = _read_json(path)
    except Exception as exc:
        return {"ok": False, "errors": [f"read_error:{exc!r}"], "count": 0, "items": []}
    if isinstance(payload, dict):
        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            payload = candidates
        else:
            return {"ok": False, "errors": ["payload_not_list"], "count": 0, "items": []}
    if not isinstance(payload, list):
        return {"ok": False, "errors": ["payload_not_list"], "count": 0, "items": []}

    space_keys = set(_space_keys())
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            errors.append(f"item_not_dict:{idx}")
            continue
        params = entry.get("params")
        if not isinstance(params, dict):
            errors.append(f"params_not_dict:{idx}")
            continue
        if not (space_keys & set(params.keys())):
            errors.append(f"params_missing_space_keys:{idx}")
        items.append({
            "idx": idx,
            "params_keys": sorted(list(params.keys())),
            "source": _extract_sources(entry),
        })
        count += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "count": count,
        "items": items,
    }


def validate_faseB_promoted(path: str) -> Dict[str, Any]:
    errors: List[str] = []
    count = 0
    items: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return {"ok": False, "errors": [f"missing:{path}"], "count": 0, "items": []}
    try:
        payload = _read_json(path)
    except Exception as exc:
        return {"ok": False, "errors": [f"read_error:{exc!r}"], "count": 0, "items": []}
    if not isinstance(payload, list):
        return {"ok": False, "errors": ["payload_not_list"], "count": 0, "items": []}

    space_keys = set(_space_keys())
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            errors.append(f"item_not_dict:{idx}")
            continue
        params = entry.get("params_best")
        if not isinstance(params, dict):
            params = entry.get("params_frozen")
        if not isinstance(params, dict):
            errors.append(f"params_not_dict:{idx}")
            continue
        if not (space_keys & set(params.keys())):
            errors.append(f"params_missing_space_keys:{idx}")
        items.append({
            "idx": idx,
            "params_keys": sorted(list(params.keys())),
            "source": entry.get("source"),
        })
        count += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "count": count,
        "items": items,
    }


def detect_latest_files(root: str) -> Dict[str, Any]:
    root = os.path.abspath(root)
    robust_dir = os.path.join(root, "results", "robust")
    promo_dir = os.path.join(root, "results", "promotions")
    stagec_trades_dir = os.path.join(root, "results", "pipeline_stageC_trades")

    robust_candidates: List[str] = []
    if os.path.isdir(robust_dir):
        for name in os.listdir(robust_dir):
            if not name.lower().endswith(".json"):
                continue
            if name.lower().startswith("robust_"):
                robust_candidates.append(os.path.join(robust_dir, name))
        if not robust_candidates:
            smoke = os.path.join(robust_dir, "smoke.json")
            if os.path.exists(smoke):
                robust_candidates.append(smoke)

    def _latest(paths: List[str]) -> Tuple[Optional[str], Optional[float]]:
        if not paths:
            return None, None
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            return None, None
        latest_path = max(paths, key=lambda p: os.path.getmtime(p))
        return latest_path, os.path.getmtime(latest_path)

    robust_path, robust_mtime = _latest(robust_candidates)

    faseA_path = os.path.join(promo_dir, "faseA_promoted.json")
    faseB_path = os.path.join(promo_dir, "faseB_promoted.json")
    faseC_path = os.path.join(promo_dir, "faseC_promoted.json")
    faseC_report = os.path.join(promo_dir, "faseC_report.csv")

    def _mtime(path: str) -> Optional[float]:
        return os.path.getmtime(path) if os.path.exists(path) else None

    return {
        "robust_json": robust_path,
        "robust_json_mtime": robust_mtime,
        "faseA_promoted": faseA_path if os.path.exists(faseA_path) else None,
        "faseA_promoted_mtime": _mtime(faseA_path),
        "faseB_promoted": faseB_path if os.path.exists(faseB_path) else None,
        "faseB_promoted_mtime": _mtime(faseB_path),
        "faseC_promoted": faseC_path if os.path.exists(faseC_path) else None,
        "faseC_promoted_mtime": _mtime(faseC_path),
        "faseC_report": faseC_report if os.path.exists(faseC_report) else None,
        "faseC_report_mtime": _mtime(faseC_report),
        "stageC_trades_dir": stagec_trades_dir if os.path.isdir(stagec_trades_dir) else None,
    }


def _top_fail_reasons(fail_reason_list: List[Any], top_n: int = 3) -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for item in fail_reason_list:
        if isinstance(item, list):
            for sub in item:
                if sub:
                    counter[str(sub)] += 1
        elif item:
            counter[str(item)] += 1
    return counter.most_common(top_n)


def _stagec_summary(paths: Dict[str, Any]) -> Dict[str, Any]:
    fasec_promoted = paths.get("faseC_promoted")
    report_csv = paths.get("faseC_report")
    trades_dir = paths.get("stageC_trades_dir")

    if not fasec_promoted and not report_csv and not trades_dir:
        return {"stageC_ok": None, "pass_count": None, "windows_ok_avg": None}

    pass_count: Optional[int] = None
    windows_ok_avg: Optional[float] = None
    if fasec_promoted and os.path.exists(fasec_promoted):
        try:
            payload = _read_json(fasec_promoted)
            if isinstance(payload, list):
                pass_count = len(payload)
        except Exception:
            pass_count = None

    if report_csv and os.path.exists(report_csv):
        try:
            rows: List[Dict[str, Any]] = []
            with open(report_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            if rows:
                vals: List[int] = []
                for row in rows:
                    try:
                        vals.append(int(float(row.get("passed_windows", 0))))
                    except Exception:
                        vals.append(0)
                windows_ok_avg = sum(vals) / len(vals)
                if pass_count is None:
                    pass_count = sum(1 for row in rows if str(row.get("pass_C")).lower() in ("true", "1"))
        except Exception:
            windows_ok_avg = None

    stagec_ok: Optional[bool]
    if pass_count is None:
        stagec_ok = None
    else:
        stagec_ok = pass_count > 0

    return {
        "stageC_ok": stagec_ok,
        "pass_count": pass_count,
        "windows_ok_avg": windows_ok_avg,
    }


def build_health_report(root: str, stale_seconds: int = 300) -> Dict[str, Any]:
    paths = detect_latest_files(root)

    robust_path = paths.get("robust_json")
    robust_validation = validate_robust_json(robust_path) if robust_path else None
    fasea_path = paths.get("faseA_promoted")
    fasea_validation = validate_faseA_promoted(fasea_path) if fasea_path else None
    faseb_path = paths.get("faseB_promoted")
    faseb_validation = validate_faseB_promoted(faseb_path) if faseb_path else None

    latest_robust_ok = bool(robust_validation and robust_validation.get("ok"))
    latest_robust_passed_count = robust_validation.get("counts", {}).get("passed_count") if robust_validation else 0
    best_robust_score = None
    top_fail_reason: List[Tuple[str, int]] = []
    if robust_path and os.path.exists(robust_path):
        try:
            payload = _read_json(robust_path)
            rs_list = _safe_list(payload.get("robust_score"))
            if rs_list:
                best_robust_score = max(rs_list)
            top_fail_reason = _top_fail_reasons(_safe_list(payload.get("fail_reason")))
        except Exception:
            pass

    fasea_ok = bool(fasea_validation and fasea_validation.get("ok"))
    fasea_count = int(fasea_validation.get("count", 0)) if fasea_validation else 0
    no_promotion_a = bool(fasea_ok and fasea_count == 0)

    faseb_ok: Optional[bool] = None
    faseb_count: Optional[int] = None
    if faseb_validation is not None:
        faseb_ok = bool(faseb_validation.get("ok"))
        faseb_count = int(faseb_validation.get("count", 0))

    stagec_summary = _stagec_summary(paths)

    contract_ok = False
    contract_reason = "CONTRACT_FAIL"
    if no_promotion_a:
        contract_ok = True
        contract_reason = "NO_PROMOTION_CONTINUE_A"
    elif latest_robust_ok and latest_robust_passed_count > 0 and fasea_ok:
        contract_ok = True
        contract_reason = "CONTRACT_OK"

    a_ran = robust_path is not None
    b_ran = faseb_path is not None

    liveness_ok = True
    if a_ran:
        samples = robust_validation.get("counts", {}).get("samples", 0) if robust_validation else 0
        if not latest_robust_ok or samples <= 0:
            liveness_ok = False
    if b_ran and (faseb_count or 0) <= 0:
        liveness_ok = False

    warnings: List[str] = []
    warning_fasea_stale = False
    warning_fasea_stale_delta: Optional[int] = None
    robust_mtime = paths.get("robust_json_mtime")
    fasea_mtime = paths.get("faseA_promoted_mtime")
    if robust_mtime and fasea_mtime:
        delta_sec = robust_mtime - fasea_mtime
        if delta_sec > stale_seconds:
            warning_fasea_stale = True
            warning_fasea_stale_delta = int(delta_sec)
            warnings.append(
                f"faseA_promoted_stale:delta_sec={warning_fasea_stale_delta}"
                f":threshold_sec={int(stale_seconds)}"
            )

    report = {
        "ts_utc": _utc_now().isoformat(),
        "ts_local": _local_now().isoformat(),
        "paths": paths,
        "A": {
            "latest_robust_ok": latest_robust_ok,
            "latest_robust_passed_count": latest_robust_passed_count,
            "best_robust_score": best_robust_score,
            "top_fail_reason": top_fail_reason,
            "validation": robust_validation,
        },
        "Promo": {
            "faseA_ok": fasea_ok,
            "faseA_count": fasea_count,
            "validation": fasea_validation,
        },
        "B": {
            "faseB_ok": faseb_ok,
            "faseB_count": faseb_count,
            "validation": faseb_validation,
        },
        "C": stagec_summary,
        "contract_ok": contract_ok,
        "contract_reason": contract_reason,
        "liveness_ok": liveness_ok,
        "no_promotion_a": no_promotion_a,
        "warnings": warnings,
        "warning_faseA_stale": warning_fasea_stale,
        "warning_faseA_stale_delta_seconds": warning_fasea_stale_delta,
        "stale_seconds": int(stale_seconds),
    }

    return report


def _load_state(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {
            "fail_streak_contract": 0,
            "fail_streak_liveness": 0,
            "last_health_ts": None,
        }
    try:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ValueError("state_not_dict")
        return {
            "fail_streak_contract": int(payload.get("fail_streak_contract", 0)),
            "fail_streak_liveness": int(payload.get("fail_streak_liveness", 0)),
            "last_health_ts": payload.get("last_health_ts"),
        }
    except Exception:
        return {
            "fail_streak_contract": 0,
            "fail_streak_liveness": 0,
            "last_health_ts": None,
        }


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _write_stop_file(path: str, report: Dict[str, Any], reason: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        f"STOP AUTOLOOP {datetime.now().isoformat()}",
        f"reason={reason}",
        f"latest_robust={report.get('paths', {}).get('robust_json')}",
        f"A_passed={report.get('A', {}).get('latest_robust_passed_count')}",
        f"faseA_count={report.get('Promo', {}).get('faseA_count')}",
        f"faseB_count={report.get('B', {}).get('faseB_count')}",
        f"contract_ok={report.get('contract_ok')}",
        f"liveness_ok={report.get('liveness_ok')}",
        "",
        "paths:",
        json.dumps(report.get("paths", {}), ensure_ascii=False, indent=2),
        "",
        "errors:",
    ]
    errors: List[str] = []
    for section in ("A", "Promo", "B"):
        validation = (report.get(section) or {}).get("validation") or {}
        errs = validation.get("errors") or []
        for err in errs:
            errors.append(f"{section}:{err}")
    if errors:
        lines.extend(errors)
    else:
        lines.append("none")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _update_state(
    state_path: str,
    report: Dict[str, Any],
    stop_on_contract_fail: bool,
    stop_on_liveness_fail: bool,
) -> Tuple[Dict[str, Any], bool, str]:
    state = _load_state(state_path)
    contract_ok = bool(report.get("contract_ok"))
    liveness_ok = bool(report.get("liveness_ok"))

    if contract_ok:
        state["fail_streak_contract"] = 0
    else:
        state["fail_streak_contract"] = int(state.get("fail_streak_contract", 0)) + 1

    if liveness_ok:
        state["fail_streak_liveness"] = 0
    else:
        state["fail_streak_liveness"] = int(state.get("fail_streak_liveness", 0)) + 1

    state["last_health_ts"] = report.get("ts_utc")

    stop = False
    reason = ""
    if not contract_ok and stop_on_contract_fail:
        stop = True
        reason = "CONTRACT_FAIL"
    elif not liveness_ok and stop_on_liveness_fail:
        stop = True
        reason = "LIVENESS_FAIL"

    _write_json(state_path, state)

    return state, stop, reason


def _log_line(path: Optional[str], line: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser("pipeline_health")
    ap.add_argument("--root", required=True, help="repo root")
    ap.add_argument("--log", default=None, help="log file (optional)")
    ap.add_argument("--out", default=None, help="health json output")
    ap.add_argument("--stop-file", default=None, help="stop file path")
    ap.add_argument("--state-file", default=None, help="health state json")
    ap.add_argument(
        "--stop-on-contract-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--stop-on-liveness-fail",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--stale-seconds", type=int, default=300)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_path = args.out or os.path.join(root, "results", "health", "health_latest.json")
    state_path = args.state_file or os.path.join(root, "results", "health", "health_state.json")

    report = build_health_report(root, stale_seconds=int(args.stale_seconds))
    state, stop_requested, reason = _update_state(
        state_path,
        report,
        stop_on_contract_fail=bool(args.stop_on_contract_fail),
        stop_on_liveness_fail=bool(args.stop_on_liveness_fail),
    )

    contract_ok = bool(report.get("contract_ok"))
    liveness_ok = bool(report.get("liveness_ok"))
    no_promotion_a = bool(report.get("no_promotion_a"))
    if not contract_ok:
        exit_code = 2
        stop_reason = "CONTRACT_FAIL"
        reason = "CONTRACT_FAIL"
    elif not liveness_ok:
        exit_code = 3
        stop_reason = "LIVENESS_FAIL"
        reason = "LIVENESS_FAIL"
    else:
        exit_code = 0
        stop_reason = ""
        reason = "NO_PROMOTION_CONTINUE_A" if no_promotion_a else ""

    if stop_requested:
        if args.stop_file:
            _write_stop_file(args.stop_file, report, stop_reason or reason)
        else:
            warn_line = "[HEALTH][WARN] stop requested but --stop-file not provided"
            print(warn_line, flush=True)
            _log_line(args.log, warn_line)

    report["state"] = state
    report["stop"] = stop_requested
    report["stop_reason"] = stop_reason
    report["reason"] = reason
    report["exit_code"] = exit_code

    _write_json(out_path, report)

    line = (
        "[HEALTH] "
        f"contract_ok={report.get('contract_ok')} "
        f"liveness_ok={report.get('liveness_ok')} "
        f"A_passed={report.get('A', {}).get('latest_robust_passed_count')} "
        f"faseA_count={report.get('Promo', {}).get('faseA_count')} "
        f"faseB_count={report.get('B', {}).get('faseB_count')} "
        f"stop={stop_requested} reason=\"{reason}\" exit_code={report.get('exit_code')}"
    )
    print(line, flush=True)
    _log_line(args.log, line)

    if no_promotion_a:
        info_line = "[HEALTH] No promotion in Phase A -> continue Phase A (no stop)"
        print(info_line, flush=True)
        _log_line(args.log, info_line)

    warnings = report.get("warnings") or []
    for warning in warnings:
        warn_line = f"[HEALTH][WARN] {warning}"
        print(warn_line, flush=True)
        _log_line(args.log, warn_line)

    sys.exit(int(report.get("exit_code", 0)))


if __name__ == "__main__":
    main()
