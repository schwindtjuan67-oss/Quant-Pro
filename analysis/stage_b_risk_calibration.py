#!/usr/bin/env python3
# analysis/stage_b_risk_calibration.py
"""Stage B: calibración de risk/ejecución (research-only, sin Shadow).

Lee candidatos promovidos por Fase A (results/promotions/faseA_promoted.json),
congela los parámetros de señal/edge (phaseA_frozen_keys) y samplea SOLO los
parámetros de risk/ejecución (Phase B). Evalúa en TrainB + Holdout temporal
(tail) por ventana y exporta results/promotions/faseB_promoted.json.

Notas importantes:
  - Este script NO requiere que exista Shadow.
  - Usa el backtest in-memory vía backtest.run_backtest (mismo entrypoint que robust_optimizer).
  - Es compatible tanto si tu opt_space ya filtra por PIPELINE_PHASE como si NO:
    acá forzamos el subset de keys para B igualmente.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# Imports del repo
# -------------------------
from analysis.opt_space import default_param_space, sample_params
from analysis.grid_metrics import compute_metrics_from_trades, compute_score
from analysis.robust_optimizer import window_to_dates, load_candles_from_path, _real_backtest_fn


# -------------------------
# Defaults / Keys por fase
# -------------------------

# Keys típicas de "señal" en tu space. Si tu patch A-to-B cambió esto,
# no pasa nada: el script también usa phaseA_frozen_keys si viene en el JSON.
A_KEYS_FALLBACK = {
    "ema_fast",
    "ema_slow",
    "delta_threshold",
    "delta_rolling_sec",
}


def _months_in_window(window: str) -> int:
    """Cantidad de meses (aprox) inclusivos en una window YYYY-MM_YYYY-MM."""
    start, end = window.strip().split("_")
    y1, m1 = map(int, start.split("-"))
    y2, m2 = map(int, end.split("-"))
    return (y2 - y1) * 12 + (m2 - m1) + 1


def _default_holdout_days(window: str) -> int:
    months = _months_in_window(window)
    # scalper 1m: holdout suficientemente largo para captar colas
    if months >= 18:
        return 90
    return 60


def _parse_holdout_map(s: str) -> Dict[str, int]:
    """Parsea un mapa de holdout por window.

    Formato:
        "2019-01_2020-12:90,2021-01_2021-12:60"
    """
    out: Dict[str, int] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        w, d = part.split(":", 1)
        try:
            out[w.strip()] = int(d.strip())
        except Exception:
            pass
    return out


def _ms(v: int) -> str:
    try:
        return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return str(v)


def _split_train_holdout_by_tail_days(candles: List[Dict[str, Any]], holdout_days: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not candles:
        return [], []
    # timestamps normalizados en robust_optimizer como timestamp_ms
    max_ts = int(candles[-1].get("timestamp_ms", 0))
    cut_ts = max_ts - int(holdout_days) * 86400 * 1000
    train: List[Dict[str, Any]] = []
    hold: List[Dict[str, Any]] = []
    for c in candles:
        ts = int(c.get("timestamp_ms", 0))
        if ts >= cut_ts:
            hold.append(c)
        else:
            train.append(c)
    return train, hold


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class WindowBResult:
    window: str
    holdout_days: int
    train_metrics: Dict[str, float]
    hold_metrics: Dict[str, float]
    train_score: float
    hold_score: float
    stability: Dict[str, float]
    passed: bool
    reason: str
    best_params: Dict[str, Any]


def _passes_holdout_gates(m: Dict[str, float], gates: Dict[str, float]) -> Tuple[bool, str]:
    # trades
    if m.get("trades", 0) < gates.get("min_trades", 150):
        return False, f"trades<{gates.get('min_trades',150)}"
    if m.get("profit_factor", 0.0) < gates.get("min_pf", 1.20):
        return False, f"pf<{gates.get('min_pf',1.20)}"
    if m.get("winrate", 0.0) < gates.get("min_winrate", 0.35):
        return False, f"winrate<{gates.get('min_winrate',0.35)}"
    if abs(m.get("max_drawdown_r", 0.0)) > gates.get("max_dd_r", 12.0):
        return False, f"dd_r>{gates.get('max_dd_r',12.0)}"
    if m.get("expectancy", 0.0) <= gates.get("min_expectancy", 0.0):
        return False, "expectancy<=0"
    return True, "ok"


def _passes_stability(train_m: Dict[str, float], hold_m: Dict[str, float], stab: Dict[str, float]) -> Tuple[bool, str]:
    # ratios mínimos
    exp_ok = hold_m.get("expectancy", 0.0) >= stab.get("exp_ratio_min", 0.70) * max(1e-9, train_m.get("expectancy", 0.0))
    pf_ok = hold_m.get("profit_factor", 0.0) >= stab.get("pf_ratio_min", 0.85) * max(1e-9, train_m.get("profit_factor", 0.0))
    dd_ok = abs(hold_m.get("max_drawdown_r", 0.0)) <= stab.get("dd_ratio_max", 1.25) * max(1e-9, abs(train_m.get("max_drawdown_r", 0.0)))
    if not exp_ok:
        return False, "stability_expectancy"
    if not pf_ok:
        return False, "stability_pf"
    if not dd_ok:
        return False, "stability_dd"
    return True, "ok"


def _subset_space(full: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: full[k] for k in keys if k in full}


def calibrate_risk_for_candidate(
    *,
    base_params: Dict[str, Any],
    frozen_keys: List[str],
    data_path: str,
    base_cfg: Dict[str, Any],
    windows: List[str],
    samples: int,
    rng: np.random.Generator,
    gates_holdout: Dict[str, float],
    stability_cfg: Dict[str, float],
    holdout_map: Dict[str, int],
    interval: str = "1m",
    warmup: int = 500,
) -> Tuple[List[WindowBResult], Dict[str, Any]]:
    """Devuelve resultados por window y el mejor params global (merge frozen+risk) si aplica."""

    full_space = default_param_space()

    # Definimos B-keys como: todo lo del space menos frozen_keys (señal).
    frozen_set = set(frozen_keys)
    b_keys = [k for k in full_space.keys() if k not in frozen_set]
    space_b = _subset_space(full_space, b_keys)

    results: List[WindowBResult] = []

    # para elegir el mejor global, usamos suma de hold_score de windows pasadas
    global_best_params: Optional[Dict[str, Any]] = None
    global_best_sum_hold: float = -1e18

    for window in windows:
        from_date, to_date = window_to_dates(window)
        print(f"[B] WINDOW={window} dates {from_date} -> {to_date}")
        candles = load_candles_from_path(data_path, date_from=from_date, date_to=to_date, debug=False)
        if not candles:
            results.append(WindowBResult(
                window=window,
                holdout_days=holdout_map.get(window, _default_holdout_days(window)),
                train_metrics={},
                hold_metrics={},
                train_score=-1e9,
                hold_score=-1e9,
                stability={},
                passed=False,
                reason="no_candles",
                best_params=dict(base_params),
            ))
            continue

        holdout_days = holdout_map.get(window, _default_holdout_days(window))
        train_c, hold_c = _split_train_holdout_by_tail_days(candles, holdout_days)

        if not train_c or not hold_c:
            results.append(WindowBResult(
                window=window,
                holdout_days=holdout_days,
                train_metrics={},
                hold_metrics={},
                train_score=-1e9,
                hold_score=-1e9,
                stability={},
                passed=False,
                reason="split_empty",
                best_params=dict(base_params),
            ))
            continue

        print(f"[B] candles train={len(train_c)} holdout={len(hold_c)} holdout_days={holdout_days}")
        try:
            print(f"[B] train ts: {_ms(int(train_c[0]['timestamp_ms']))} -> {_ms(int(train_c[-1]['timestamp_ms']))}")
            print(f"[B] hold  ts: {_ms(int(hold_c[0]['timestamp_ms']))} -> {_ms(int(hold_c[-1]['timestamp_ms']))}")
        except Exception:
            pass

        best_train_score = -1e18
        best_params = None
        best_train_metrics: Dict[str, float] = {}

        # Evalúa risk samples sobre TRAIN
        for i in range(int(samples)):
            # samplea risk params (space_b) — aunque el opt_space ya filtre, acá garantizamos
            risk_p = sample_params(space_b, rng, hard_constraints=True) if space_b else {}
            merged = dict(base_params)
            merged.update(risk_p)
            # hard freeze
            for k in frozen_set:
                if k in base_params:
                    merged[k] = base_params[k]

            # backtest train
            trades = _real_backtest_fn(train_c, merged, base_cfg=base_cfg, symbol=base_cfg.get("symbol"), interval=interval, warmup=warmup)
            m = compute_metrics_from_trades(trades)
            s = compute_score(m)

            # gate mínimo en train para no elegir basura
            if m.get("trades", 0) < max(50, int(gates_holdout.get("min_trades", 150) // 2)):
                continue
            if m.get("profit_factor", 0.0) < max(1.05, gates_holdout.get("min_pf", 1.20) - 0.10):
                continue
            if abs(m.get("max_drawdown_r", 0.0)) > max(20.0, gates_holdout.get("max_dd_r", 12.0) * 1.8):
                continue

            if s > best_train_score:
                best_train_score = s
                best_params = merged
                best_train_metrics = m

        if best_params is None:
            results.append(WindowBResult(
                window=window,
                holdout_days=holdout_days,
                train_metrics={},
                hold_metrics={},
                train_score=-1e9,
                hold_score=-1e9,
                stability={},
                passed=False,
                reason="no_train_candidate",
                best_params=dict(base_params),
            ))
            continue

        # Evalúa HOLDOUT con el mejor params elegido en train
        hold_trades = _real_backtest_fn(hold_c, best_params, base_cfg=base_cfg, symbol=base_cfg.get("symbol"), interval=interval, warmup=warmup)
        hold_m = compute_metrics_from_trades(hold_trades)
        hold_s = compute_score(hold_m)

        ok_h, why_h = _passes_holdout_gates(hold_m, gates_holdout)
        ok_st, why_st = _passes_stability(best_train_metrics, hold_m, stability_cfg)
        passed = bool(ok_h and ok_st)
        reason = "ok" if passed else (why_h if not ok_h else why_st)

        stability = {
            "exp_ratio": _safe_float(hold_m.get("expectancy"), 0.0) / max(1e-9, _safe_float(best_train_metrics.get("expectancy"), 0.0)),
            "pf_ratio": _safe_float(hold_m.get("profit_factor"), 0.0) / max(1e-9, _safe_float(best_train_metrics.get("profit_factor"), 0.0)),
            "dd_ratio": abs(_safe_float(hold_m.get("max_drawdown_r"), 0.0)) / max(1e-9, abs(_safe_float(best_train_metrics.get("max_drawdown_r"), 0.0))),
        }

        results.append(WindowBResult(
            window=window,
            holdout_days=holdout_days,
            train_metrics=best_train_metrics,
            hold_metrics=hold_m,
            train_score=float(best_train_score),
            hold_score=float(hold_s),
            stability=stability,
            passed=passed,
            reason=reason,
            best_params=best_params,
        ))

        # ranking global
        sum_hold = sum(r.hold_score for r in results if r.passed)
        if sum_hold > global_best_sum_hold:
            global_best_sum_hold = sum_hold
            global_best_params = best_params

    return results, (global_best_params or dict(base_params))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default="datasets/SOLUSDT/1m", help="Path a dataset (carpeta con CSVs)")
    ap.add_argument("--base-config", default="configs/pipeline_research_backtest.json", help="Config base del backtest")
    ap.add_argument("--fasea", default="results/promotions/faseA_promoted.json", help="Input promoted JSON de Fase A")
    ap.add_argument("--out", default="results/promotions/faseB_promoted.json", help="Output JSON Fase B")
    ap.add_argument("--report-csv", default="results/promotions/faseB_report.csv", help="Reporte CSV para inspección")

    ap.add_argument("--samples", type=int, default=int(os.getenv("STAGEB_SAMPLES", "200")), help="Samples random de risk por window")
    ap.add_argument("--seed", type=int, default=int(os.getenv("STAGEB_SEED", "1337")), help="Seed RNG")
    ap.add_argument("--warmup", type=int, default=int(os.getenv("STAGEB_WARMUP", "500")), help="Warmup candles")
    ap.add_argument("--interval", default=os.getenv("STAGEB_INTERVAL", "1m"), help="Interval string")

    ap.add_argument("--min-windows-ok", type=int, default=int(os.getenv("STAGEB_MIN_WINDOWS_OK", "2")), help="Mínimo de windows que deben pasar para promover")

    ap.add_argument(
        "--holdout-map",
        default=os.getenv("STAGEB_HOLDOUT_MAP", ""),
        help="Override holdout días por window: '2019-01_2020-12:90,2021-01_2021-12:60'",
    )

    # Gates HOLDOUT
    ap.add_argument("--min-trades-holdout", type=int, default=int(os.getenv("STAGEB_MIN_TRADES", "150")))
    ap.add_argument("--min-pf-holdout", type=float, default=float(os.getenv("STAGEB_MIN_PF", "1.20")))
    ap.add_argument("--min-winrate-holdout", type=float, default=float(os.getenv("STAGEB_MIN_WINRATE", "0.35")))
    ap.add_argument("--max-dd-r-holdout", type=float, default=float(os.getenv("STAGEB_MAX_DD_R", "12.0")))
    ap.add_argument("--min-expectancy-holdout", type=float, default=float(os.getenv("STAGEB_MIN_EXPECTANCY", "0.0")))

    # Stability
    ap.add_argument("--exp-ratio-min", type=float, default=float(os.getenv("STAGEB_EXP_RATIO_MIN", "0.70")))
    ap.add_argument("--pf-ratio-min", type=float, default=float(os.getenv("STAGEB_PF_RATIO_MIN", "0.85")))
    ap.add_argument("--dd-ratio-max", type=float, default=float(os.getenv("STAGEB_DD_RATIO_MAX", "1.25")))

    args = ap.parse_args()

    # cargar base cfg
    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    # cargar fase A
    if not os.path.exists(args.fasea):
        raise FileNotFoundError(f"No existe {args.fasea}")
    with open(args.fasea, "r", encoding="utf-8") as f:
        fasea = json.load(f)
    if not isinstance(fasea, list):
        raise ValueError("faseA_promoted.json debe ser una lista")

    holdout_map = _parse_holdout_map(args.holdout_map)

    gates_holdout = {
        "min_trades": float(args.min_trades_holdout),
        "min_pf": float(args.min_pf_holdout),
        "min_winrate": float(args.min_winrate_holdout),
        "max_dd_r": float(args.max_dd_r_holdout),
        "min_expectancy": float(args.min_expectancy_holdout),
    }
    stability_cfg = {
        "exp_ratio_min": float(args.exp_ratio_min),
        "pf_ratio_min": float(args.pf_ratio_min),
        "dd_ratio_max": float(args.dd_ratio_max),
    }

    rng = np.random.default_rng(int(args.seed))

    out_rows: List[Dict[str, Any]] = []
    promoted_b: List[Dict[str, Any]] = []

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report_csv) or ".", exist_ok=True)

    print("=" * 60)
    print("[STAGE B] Risk calibration")
    print(f"data={args.data}")
    print(f"base_config={args.base_config}")
    print(f"faseA={args.fasea} candidates={len(fasea)}")
    print(f"samples/window={args.samples}")
    print("=" * 60)

    for idx, cand in enumerate(fasea, 1):
        params = cand.get("params") or {}
        if not isinstance(params, dict) or not params:
            continue

        windows = list((cand.get("windows") or {}).keys())
        if not windows:
            # fallback: si no trae windows, usar las del pipeline (si existe)
            windows = ["2019-01_2020-12", "2021-01_2021-12", "2022-01_2023-06"]

        frozen_keys = cand.get("phaseA_frozen_keys") or cand.get("phaseA_frozen") or None
        if isinstance(frozen_keys, list) and frozen_keys:
            frozen_keys = [str(x) for x in frozen_keys]
        else:
            frozen_keys = sorted([k for k in params.keys() if k in A_KEYS_FALLBACK])

        print("\n" + "-" * 60)
        print(f"[CAND] {idx}/{len(fasea)} promotion_score={cand.get('promotion_score')} frozen_keys={frozen_keys}")

        # calibrate
        win_results, best_global = calibrate_risk_for_candidate(
            base_params=params,
            frozen_keys=frozen_keys,
            data_path=args.data,
            base_cfg=base_cfg,
            windows=windows,
            samples=args.samples,
            rng=rng,
            gates_holdout=gates_holdout,
            stability_cfg=stability_cfg,
            holdout_map=holdout_map,
            interval=args.interval,
            warmup=args.warmup,
        )

        n_ok = sum(1 for r in win_results if r.passed)
        pass_b = n_ok >= int(args.min_windows_ok)

        # record
        entry = {
            "phase": "B",
            "phaseA_promotion_score": cand.get("promotion_score"),
            "phaseA_frozen_keys": frozen_keys,
            "params_frozen": {k: params.get(k) for k in frozen_keys},
            "params_best": best_global,
            "windows": {
                r.window: {
                    "passed": r.passed,
                    "reason": r.reason,
                    "holdout_days": r.holdout_days,
                    "train_metrics": r.train_metrics,
                    "hold_metrics": r.hold_metrics,
                    "train_score": r.train_score,
                    "hold_score": r.hold_score,
                    "stability": r.stability,
                }
                for r in win_results
            },
            "passed_windows": n_ok,
            "pass_B": pass_b,
        }

        # ranking por suma hold_score de windows pasadas
        entry["rank_score"] = float(sum(r.hold_score for r in win_results if r.passed))

        # CSV row simplificado
        out_rows.append({
            "rank_score": entry["rank_score"],
            "pass_B": pass_b,
            "passed_windows": n_ok,
            "promotion_score_A": cand.get("promotion_score"),
            "frozen_keys": ",".join(frozen_keys),
        })

        # Para debug, agregamos pocas columnas por window
        for r in win_results:
            out_rows[-1][f"{r.window}_pass"] = r.passed
            out_rows[-1][f"{r.window}_hold_pf"] = r.hold_metrics.get("profit_factor", 0.0) if r.hold_metrics else 0.0
            out_rows[-1][f"{r.window}_hold_exp"] = r.hold_metrics.get("expectancy", 0.0) if r.hold_metrics else 0.0
            out_rows[-1][f"{r.window}_hold_dd"] = r.hold_metrics.get("max_drawdown_r", 0.0) if r.hold_metrics else 0.0
            out_rows[-1][f"{r.window}_hold_trades"] = r.hold_metrics.get("trades", 0) if r.hold_metrics else 0

        if pass_b:
            promoted_b.append(entry)

        print(f"[CAND] pass_B={pass_b} passed_windows={n_ok}/{len(win_results)} rank_score={entry['rank_score']:.4f}")

    # orden
    promoted_b.sort(key=lambda x: x.get("rank_score", -1e18), reverse=True)

    # write JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(promoted_b, f, indent=2, ensure_ascii=False)

    # write CSV (sin pandas, para evitar deps)
    # headers: unión de keys
    headers: List[str] = []
    for r in out_rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)

    import csv
    with open(args.report_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in sorted(out_rows, key=lambda x: float(x.get("rank_score", -1e18)), reverse=True):
            w.writerow(r)

    print("=" * 60)
    print(f"[STAGE B] promoted: {len(promoted_b)}")
    print(f"[STAGE B] wrote: {args.out}")
    print(f"[STAGE B] wrote: {args.report_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
