#!/usr/bin/env python3
# analysis/stage_c_pipeline_eval.py

"""Stage C: supervivencia real (research-only, sin Shadow).

Objetivo
--------
- Lee candidatos promovidos por Fase B (results/promotions/faseB_promoted.json).
- Corre backtest offline por window usando el runner real.
- Loggea a CSV schema v4 (TradeLogger) con meta_json.params y meta_json.context.window.
- Aplica los *mismos filtros reales* del evaluator del repo (analysis/analyze_pipeline.py):
  trades mínimos, expectancy, sortino, profit factor, max drawdown (en equity_after),
  winrate y worst-5%.

Salida
------
- results/promotions/faseC_promoted.json: lista de candidatos que pasan C.
- results/promotions/faseC_report.csv: reporte de TODOS los candidatos evaluados.

Notas
-----
- Por default se desactiva el flush a parquet del logger para evitar depender de pyarrow;
  se usa sólo el CSV (PIPELINE_WRITE_CSV=1).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------
# IMPORTANT: Forzar modo PIPELINE + CSV (antes de instanciar loggers)
# ------------------------------------------------------------
os.environ.setdefault("RUN_MODE", "PIPELINE")
os.environ.setdefault("PIPELINE_WRITE_CSV", "1")

try:
    import Live.logger_pro as logger_pro

    # Forzar modo pipeline en el módulo (por si ya se importó con otro RUN_MODE)
    logger_pro.RUN_MODE = "PIPELINE"
    logger_pro.PIPELINE_WRITE_CSV = True

    # Desactivar parquet por defecto (no rompe el CSV)
    if os.getenv("PIPELINE_DISABLE_PARQUET", "1").lower() in ("1", "true", "yes"):
        def _flush_parquet_noop(self):
            self._parquet_buffer.clear()
            return
        def _close_noop(self):
            self._parquet_buffer.clear()
            return
        try:
            logger_pro.TradeLogger._flush_parquet = _flush_parquet_noop  # type: ignore
            logger_pro.TradeLogger.close = _close_noop  # type: ignore
        except Exception:
            pass
except Exception:
    logger_pro = None  # type: ignore

# Repo imports (después de setear env)
from analysis.robust_optimizer import window_to_dates, load_candles_from_path, _real_backtest_fn


# ------------------------------------------------------------
# Thresholds (idénticos a analyze_pipeline.py)
# ------------------------------------------------------------
MIN_TRADES = int(os.getenv("PIPELINE_MIN_TRADES", "300"))
MIN_R_OBS = int(os.getenv("PIPELINE_MIN_R_OBS", "200"))

TH_EXPECTANCY = float(os.getenv("PIPELINE_TH_EXPECTANCY", "0.05"))
TH_SORTINO = float(os.getenv("PIPELINE_TH_SORTINO", "1.50"))
TH_PF = float(os.getenv("PIPELINE_TH_PF", "1.30"))
TH_DD = float(os.getenv("PIPELINE_TH_DD", "-0.20"))
TH_WINRATE = float(os.getenv("PIPELINE_TH_WINRATE", "0.40"))
TH_WORST5 = float(os.getenv("PIPELINE_TH_WORST5", "-1.50"))

# Stage C gate
STAGEC_MIN_WINDOWS_OK = int(os.getenv("STAGEC_MIN_WINDOWS_OK", os.getenv("PIPELINE_C_MIN_WINDOWS", "2")))
STAGEC_REQUIRE_GLOBAL_PASS = os.getenv("STAGEC_REQUIRE_GLOBAL_PASS", "1").lower() in ("1", "true", "yes")

EPS = 1e-12


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_json_loads(x: Any) -> Dict[str, Any]:
    if not isinstance(x, str):
        return {}
    s = x.strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"_": obj}
    except Exception:
        return {}


def _stable_params_key(params: Dict[str, Any]) -> str:
    try:
        return json.dumps(params or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(params)


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _coerce_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _hash_params(p: Dict[str, Any]) -> str:
    s = _stable_params_key(p)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


@dataclass
class EvalMetrics:
    trades: int
    expectancy_r: float
    sortino: float
    profit_factor: float
    max_dd: float
    winrate: float
    worst_5pct_r: float


def sortino(r: np.ndarray) -> float:
    if r.size == 0:
        return float("nan")
    downside = r[r < 0]
    if downside.size == 0:
        return float("inf")
    dstd = np.std(downside, ddof=1) if downside.size > 1 else 0.0
    if dstd <= 0:
        return float("inf") if r.mean() > 0 else float("-inf")
    return float(r.mean() / dstd)


def max_drawdown(equity_after: List[float]) -> float:
    """MDD en % (negativo), idéntico a analyze_pipeline.py."""
    if not equity_after:
        return float("nan")
    eq = np.array(equity_after, dtype=float)
    eq = eq[np.isfinite(eq)]
    if eq.size == 0:
        return float("nan")
    roll = np.maximum.accumulate(eq)
    roll = np.where(roll == 0, np.nan, roll)
    dd = (eq - roll) / roll
    dd = dd[np.isfinite(dd)]
    if dd.size == 0:
        return float("nan")
    return float(np.min(dd))


def profit_factor(pnl_net: List[float]) -> float:
    if not pnl_net:
        return float("nan")
    x = np.array(pnl_net, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    pos = float(x[x > 0].sum())
    neg = float(x[x < 0].sum())
    if abs(neg) < EPS:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / abs(neg))


def compute_metrics(rows: List[Dict[str, Any]]) -> Optional[EvalMetrics]:
    """rows: lista de EXIT rows ya parseadas."""
    if not rows:
        return None

    r_list: List[float] = []
    pnl_net_list: List[float] = []
    eq_after_list: List[float] = []

    for row in rows:
        pr = row.get("pnl_r")
        if pr is not None:
            r_list.append(float(pr))
        pn = row.get("pnl_net_est")
        if pn is not None:
            pnl_net_list.append(float(pn))
        ea = row.get("equity_after")
        if ea is not None:
            eq_after_list.append(float(ea))

    trades = len(r_list)
    if trades == 0:
        return None

    r = np.array(r_list, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None

    expectancy = float(r.mean())
    s = sortino(r)
    pf = profit_factor(pnl_net_list)
    dd = max_drawdown(eq_after_list)
    winrate = float((r > 0).mean())
    worst5 = float(np.quantile(r, 0.05)) if r.size > 0 else float("nan")

    return EvalMetrics(
        trades=int(trades),
        expectancy_r=expectancy,
        sortino=float(s),
        profit_factor=float(pf),
        max_dd=float(dd),
        winrate=winrate,
        worst_5pct_r=worst5,
    )


def passes_thresholds(m: EvalMetrics) -> Tuple[bool, str]:
    if m.trades < MIN_TRADES:
        return False, f"trades<{MIN_TRADES}"
    if m.trades < MIN_R_OBS:
        return False, f"r_obs<{MIN_R_OBS}"

    if not (m.expectancy_r > TH_EXPECTANCY):
        return False, f"expectancy<=TH ({m.expectancy_r:.4f}<= {TH_EXPECTANCY})"
    if not (m.sortino >= TH_SORTINO):
        return False, f"sortino<TH ({m.sortino:.3f}< {TH_SORTINO})"
    if not (m.profit_factor >= TH_PF):
        return False, f"pf<TH ({m.profit_factor:.3f}< {TH_PF})"
    if not (np.isfinite(m.max_dd) and m.max_dd > TH_DD):
        return False, f"dd<=TH ({m.max_dd:.3f}<= {TH_DD})"
    if not (m.winrate >= TH_WINRATE):
        return False, f"winrate<TH ({m.winrate:.3f}< {TH_WINRATE})"
    if not (m.worst_5pct_r > TH_WORST5):
        return False, f"worst5<=TH ({m.worst_5pct_r:.3f}<= {TH_WORST5})"

    return True, "ok"


def load_exit_rows_with_context(csv_path: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Devuelve (entry_params_by_trade, exit_rows)."""
    entry_params: Dict[str, Dict[str, Any]] = {}
    exits: List[Dict[str, Any]] = []

    if not os.path.exists(csv_path):
        return entry_params, exits

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ttype = (row.get("type") or "").upper().strip()
            tid = row.get("trade_id") or ""
            meta = _safe_json_loads(row.get("meta_json"))

            params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
            ctx = meta.get("context") if isinstance(meta.get("context"), dict) else {}
            window = ctx.get("window") or ""

            if ttype == "ENTRY":
                if tid and isinstance(params, dict) and params:
                    entry_params[tid] = params
                continue

            if ttype != "EXIT":
                continue

            pr = _coerce_float(row.get("pnl_r"))
            pn = _coerce_float(row.get("pnl_net_est"))
            ea = _coerce_float(row.get("equity_after"))

            if pr is None:
                continue

            exits.append({
                "trade_id": tid,
                "pnl_r": pr,
                "pnl_net_est": pn,
                "equity_after": ea,
                "meta": meta,
                "params": params,
                "window": window,
            })

    for r in exits:
        if (not isinstance(r.get("params"), dict)) or not r.get("params"):
            tid = r.get("trade_id")
            if tid in entry_params:
                r["params"] = entry_params[tid]

    return entry_params, exits


def run_candidate_backtests(
    *,
    candidate_params: Dict[str, Any],
    windows: List[str],
    data_path: str,
    base_cfg: Dict[str, Any],
    interval: str,
    warmup: int,
    trades_csv: str,
    run_id: str,
) -> None:
    os.makedirs(os.path.dirname(trades_csv) or ".", exist_ok=True)
    try:
        os.remove(trades_csv)
    except Exception:
        pass

    os.environ["RUN_MODE"] = "PIPELINE"
    os.environ["PIPELINE_WRITE_CSV"] = "1"
    os.environ["PIPELINE_TRADES_PATH"] = trades_csv
    os.environ["PIPELINE_RUN_ID"] = run_id

    try:
        import Live.logger_pro as lp
        lp.RUN_MODE = "PIPELINE"
        lp.PIPELINE_WRITE_CSV = True
        if os.getenv("PIPELINE_DISABLE_PARQUET", "1").lower() in ("1", "true", "yes"):
            def _flush_parquet_noop(self):
                self._parquet_buffer.clear()
                return
            def _close_noop(self):
                self._parquet_buffer.clear()
                return
            lp.TradeLogger._flush_parquet = _flush_parquet_noop  # type: ignore
            lp.TradeLogger.close = _close_noop  # type: ignore
    except Exception:
        pass

    sym = (base_cfg.get("symbol") or (base_cfg.get("symbols") or ["SOLUSDT"])[0])

    for w in windows:
        date_from, date_to = window_to_dates(w)
        os.environ["PIPELINE_WINDOW"] = w
        os.environ["PIPELINE_SEED"] = ""

        print("=" * 72)
        print(f"[STAGE C][RUN] window={w}  date_from={date_from} date_to={date_to}")
        print("=" * 72)

        candles = load_candles_from_path(
            data_path,
            date_from=date_from,
            date_to=date_to,
            debug=False,
        )

        if not candles:
            print(f"[STAGE C][WARN] window={w} candles=0 -> skip")
            continue

        _real_backtest_fn(
            candles,
            candidate_params,
            base_cfg=base_cfg,
            symbol=sym,
            interval=interval,
            warmup=warmup,
        )


def evaluate_candidate(
    *,
    trades_csv: str,
    windows: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _, exits = load_exit_rows_with_context(trades_csv)

    by_win: Dict[str, List[Dict[str, Any]]] = {w: [] for w in windows}
    unknown: List[Dict[str, Any]] = []

    for r in exits:
        w = (r.get("window") or "").strip()
        if w in by_win:
            by_win[w].append(r)
        else:
            unknown.append(r)

    per_window: Dict[str, Any] = {}
    passed_windows = 0

    for w in windows:
        rows = by_win.get(w) or []
        m = compute_metrics(rows)
        if m is None:
            per_window[w] = {"passed": False, "reason": "no_trades", "metrics": {}}
            continue

        ok, reason = passes_thresholds(m)
        if ok:
            passed_windows += 1

        per_window[w] = {
            "passed": ok,
            "reason": reason,
            "metrics": {
                "trades": m.trades,
                "expectancy_r": m.expectancy_r,
                "sortino": m.sortino,
                "profit_factor": m.profit_factor,
                "max_dd": m.max_dd,
                "winrate": m.winrate,
                "worst_5pct_r": m.worst_5pct_r,
            },
        }

    all_rows: List[Dict[str, Any]] = []
    for w in windows:
        all_rows.extend(by_win.get(w) or [])
    all_rows.extend(unknown)

    g = compute_metrics(all_rows)
    if g is None:
        global_ok, global_reason = False, "no_trades"
        global_metrics = {}
    else:
        global_ok, global_reason = passes_thresholds(g)
        global_metrics = {
            "trades": g.trades,
            "expectancy_r": g.expectancy_r,
            "sortino": g.sortino,
            "profit_factor": g.profit_factor,
            "max_dd": g.max_dd,
            "winrate": g.winrate,
            "worst_5pct_r": g.worst_5pct_r,
        }

    global_eval = {
        "passed_windows": int(passed_windows),
        "min_windows_ok": int(STAGEC_MIN_WINDOWS_OK),
        "pass_global": bool(global_ok),
        "global_reason": str(global_reason),
        "global_metrics": global_metrics,
    }

    return global_eval, per_window


def main() -> int:
    # ✅ PATCH REAL: declarar global ANTES de usar esas variables dentro de la función
    global STAGEC_MIN_WINDOWS_OK, STAGEC_REQUIRE_GLOBAL_PASS

    ap = argparse.ArgumentParser("Stage C: pipeline eval (research-only)")
    ap.add_argument("--faseb", default="results/promotions/faseB_promoted.json", help="Input JSON (Fase B promoted)")
    ap.add_argument("--data", required=True, help="Dataset path (ej: datasets/SOLUSDT/1m)")
    ap.add_argument("--base-config", required=True, help="Config base JSON (ej: configs/pipeline_research_backtest.json)")
    ap.add_argument("--out", default="results/promotions/faseC_promoted.json")
    ap.add_argument("--report-csv", default="results/promotions/faseC_report.csv")
    ap.add_argument("--trades-dir", default="results/pipeline_stageC_trades")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--warmup", type=int, default=int(os.getenv("PIPELINE_WARMUP", "500")))
    ap.add_argument("--min-windows-ok", type=int, default=STAGEC_MIN_WINDOWS_OK)
    ap.add_argument("--require-global-pass", type=int, default=1 if STAGEC_REQUIRE_GLOBAL_PASS else 0)
    args = ap.parse_args()

    STAGEC_MIN_WINDOWS_OK = int(args.min_windows_ok)
    STAGEC_REQUIRE_GLOBAL_PASS = bool(int(args.require_global_pass))

    base_cfg = _read_json(args.base_config)
    cands = _read_json(args.faseb)
    if not isinstance(cands, list):
        raise SystemExit(f"Fase B file must be a list, got: {type(cands)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report_csv) or ".", exist_ok=True)
    os.makedirs(args.trades_dir, exist_ok=True)

    promoted_c: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    print("=" * 72)
    print(f"[STAGE C] candidates_in={len(cands)}")
    print(f"[STAGE C] thresholds: MIN_TRADES={MIN_TRADES} MIN_R_OBS={MIN_R_OBS} TH_EXP={TH_EXPECTANCY} TH_SORT={TH_SORTINO} TH_PF={TH_PF} TH_DD={TH_DD} TH_WR={TH_WINRATE} TH_W5={TH_WORST5}")
    print(f"[STAGE C] gate: min_windows_ok={STAGEC_MIN_WINDOWS_OK} require_global_pass={STAGEC_REQUIRE_GLOBAL_PASS}")
    print("=" * 72)

    for i, cand in enumerate(cands, start=1):
        params = cand.get("params_best") or cand.get("params")
        if not isinstance(params, dict) or not params:
            print(f"[STAGE C][SKIP] idx={i} missing params_best")
            continue

        windows = list((cand.get("windows") or {}).keys())
        if not windows:
            windows = ["2019-01_2020-12", "2021-01_2021-12", "2022-01_2023-06"]

        run_id = f"C_{i:03d}_{_hash_params(params)}_{int(time.time())}"
        trades_csv = os.path.join(args.trades_dir, f"trades_{run_id}.csv")

        print("\n" + "#" * 72)
        print(f"[CAND] {i}/{len(cands)} run_id={run_id} windows={len(windows)}")
        print(f"[CAND] params_key={_stable_params_key(params)[:140]}")
        print("#" * 72)

        try:
            run_candidate_backtests(
                candidate_params=params,
                windows=windows,
                data_path=args.data,
                base_cfg=base_cfg,
                interval=args.interval,
                warmup=int(args.warmup),
                trades_csv=trades_csv,
                run_id=run_id,
            )
        except Exception as e:
            print(f"[STAGE C][ERROR] backtest failed: {e!r}")
            report_rows.append({
                "idx": i,
                "run_id": run_id,
                "pass_C": False,
                "reason": f"backtest_error:{e!r}",
                "passed_windows": 0,
                "global_pass": False,
                "global_reason": "backtest_error",
                "trades_csv": trades_csv,
                "ts": _utc_now(),
            })
            continue

        global_eval, per_win = evaluate_candidate(trades_csv=trades_csv, windows=windows)

        passed_windows = int(global_eval.get("passed_windows", 0))
        global_pass = bool(global_eval.get("pass_global", False))

        pass_c = passed_windows >= STAGEC_MIN_WINDOWS_OK
        reason_c = "" if pass_c else f"passed_windows<{STAGEC_MIN_WINDOWS_OK}"

        if STAGEC_REQUIRE_GLOBAL_PASS:
            if not global_pass:
                pass_c = False
                reason_c = f"global_fail:{global_eval.get('global_reason')}"

        if pass_c:
            reason_c = "ok"

        entry = {
            "phase": "C",
            "source": "faseB_promoted",
            "run_id": run_id,
            "evaluated_at_utc": _utc_now(),
            "params_best": params,
            "faseB": {
                "rank_score": cand.get("rank_score"),
                "phaseA_promotion_score": cand.get("phaseA_promotion_score"),
                "phaseA_frozen_keys": cand.get("phaseA_frozen_keys"),
            },
            "gate": {
                "min_windows_ok": STAGEC_MIN_WINDOWS_OK,
                "require_global_pass": STAGEC_REQUIRE_GLOBAL_PASS,
            },
            "global": global_eval,
            "windows": per_win,
            "pass_C": bool(pass_c),
            "reason": str(reason_c),
            "trades_csv": trades_csv,
        }

        gm = global_eval.get("global_metrics") or {}
        report_rows.append({
            "idx": i,
            "run_id": run_id,
            "pass_C": pass_c,
            "reason": reason_c,
            "passed_windows": passed_windows,
            "global_pass": global_pass,
            "global_reason": global_eval.get("global_reason"),
            "g_trades": gm.get("trades"),
            "g_expectancy": gm.get("expectancy_r"),
            "g_sortino": gm.get("sortino"),
            "g_pf": gm.get("profit_factor"),
            "g_dd": gm.get("max_dd"),
            "g_winrate": gm.get("winrate"),
            "g_worst5": gm.get("worst_5pct_r"),
            "trades_csv": trades_csv,
            "ts": entry["evaluated_at_utc"],
        })

        for w in windows:
            wobj = per_win.get(w) or {}
            report_rows[-1][f"{w}_pass"] = wobj.get("passed")
            m = wobj.get("metrics") or {}
            report_rows[-1][f"{w}_trades"] = m.get("trades")
            report_rows[-1][f"{w}_pf"] = m.get("profit_factor")
            report_rows[-1][f"{w}_exp"] = m.get("expectancy_r")
            report_rows[-1][f"{w}_dd"] = m.get("max_dd")

        if pass_c:
            promoted_c.append(entry)

        print(f"[CAND] pass_C={pass_c} passed_windows={passed_windows}/{len(windows)} global_pass={global_pass} reason={reason_c}")

    def _rank_key(x: Dict[str, Any]) -> float:
        gm = (x.get("global") or {}).get("global_metrics") or {}
        try:
            return float(gm.get("expectancy_r", 0.0)) * float(gm.get("trades", 0.0))
        except Exception:
            return -1e18

    promoted_c.sort(key=_rank_key, reverse=True)

    _write_json(args.out, promoted_c)

    headers: List[str] = []
    for r in report_rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)

    with open(args.report_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    print("=" * 72)
    print(f"[STAGE C] promoted: {len(promoted_c)}")
    print(f"[STAGE C] wrote: {args.out}")
    print(f"[STAGE C] wrote: {args.report_csv}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

