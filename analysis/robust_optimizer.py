#!/usr/bin/env python3
# robust_optimizer.py
from __future__ import annotations


import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import calendar
import csv
import json
import pickle
import multiprocessing as mp
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError  # ✅ FIX: timeout correcto

# Reutilizamos helpers existentes (pero NO dependemos exclusivamente del iterador)
from backtest.run_backtest import _date_range_to_ms, _iter_candles_from_csvs, _list_csv_files

from analysis.feature_cache import FeatureCache, set_active_cache
from analysis.grid_metrics import compute_metrics_from_trades, compute_score
from analysis.opt_space import param_space_for_phase, normalize_phase, sample_params

BacktestFn = Callable[[Any, Dict[str, Any]], List[Dict[str, Any]]]


@dataclass
class FoldResult:
    fold_id: int
    metrics: Dict[str, float]
    score: float


@dataclass
class EvalResult:
    params: Dict[str, Any]
    fold_results: List[FoldResult]
    agg: Dict[str, float]
    robust_score: float
    passed: bool
    fail_reason: List[str] = field(default_factory=list)
    exception: str = ""
    traceback: str = ""


FAIL_COUNTS = Counter()


# ============================================================
# Window -> dates (reproducible)
# ============================================================

def window_to_dates(window: str) -> Tuple[str, str]:
    """
    '2020-01_2021-12' -> ('2020-01-01', '2021-12-31')
    """
    try:
        start, end = str(window).strip().split("_")
        y1, m1 = map(int, start.split("-"))
        y2, m2 = map(int, end.split("-"))
        from_date = f"{y1:04d}-{m1:02d}-01"
        last_day = calendar.monthrange(y2, m2)[1]
        to_date = f"{y2:04d}-{m2:02d}-{last_day:02d}"
        return from_date, to_date
    except Exception as e:
        raise ValueError(f"[ROBUST] Invalid --window format: {window!r}. Expected YYYY-MM_YYYY-MM") from e


# ============================================================
# Helpers de diagnóstico + normalización timestamps
# ============================================================

def _ms_to_iso(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return str(ms)


def _coerce_ts_to_ms(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        x = int(float(v))
    except Exception:
        return None
    # Heurística: si parece segundos -> ms
    if x < 10_000_000_000:
        x *= 1000
    return x


def _pick_ts_ms(c: Dict[str, Any]) -> Optional[int]:
    for k in ("timestamp_ms", "ts_ms", "ts", "timestamp", "open_time"):
        if k in c and c[k] not in (None, ""):
            return _coerce_ts_to_ms(c[k])
    return None


def _format_elapsed_hms(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _emit_heartbeat(
    done_params: int,
    total_params: int,
    passed_params: int,
    failed_params: int,
    started_at: float,
    last_emit_at: float,
    window_label: str,
    seed: int,
) -> float:
    now = time.monotonic()
    if (now - last_emit_at) < 30.0:
        return last_emit_at
    elapsed = now - started_at
    rate = (done_params / elapsed) if elapsed > 0 else 0.0
    msg = (
        f"[ROBUST][HB] done={done_params}/{total_params} "
        f"passed={passed_params} failed={failed_params} "
        f"elapsed={_format_elapsed_hms(elapsed)} "
        f"rate={rate:.2f} params/s window={window_label} seed={seed}"
    )
    top_fails = ", ".join(f"{k}={v}" for k, v in FAIL_COUNTS.most_common(5))
    msg = f"{msg} top_fails: {top_fails}"
    print(msg, flush=True)
    return now


def _normalize_fail_reasons(reasons: Any) -> List[str]:
    if isinstance(reasons, list):
        return [str(r) for r in reasons if str(r)]
    if isinstance(reasons, str):
        return [reasons] if reasons else []
    if reasons is None:
        return []
    return [str(reasons)]


def _record_fail_reasons(reasons: List[str]) -> None:
    for reason in reasons:
        FAIL_COUNTS[reason] += 1


def _format_fail_metrics_from_folds(folds: List[FoldResult]) -> List[Dict[str, Any]]:
    return [{"fold_id": fr.fold_id, "metrics": fr.metrics} for fr in folds]


def _format_fail_metrics_from_payload_folds(folds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"fold_id": fr.get("fold_id"), "metrics": fr.get("metrics")} for fr in folds]


# ============================================================
# Fallback: parser CSV simple para tu formato:
# timestamp,open,high,low,close,volume
# ============================================================

def _iter_candles_from_simple_csv(
    files: List[str],
    start_ms: int,
    end_ms: int,
    debug: bool = False,
):
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue

            fields = [x.strip() for x in reader.fieldnames]

            # timestamp column
            ts_col = None
            for cand in ("timestamp_ms", "timestamp", "ts", "open_time"):
                if cand in fields:
                    ts_col = cand
                    break
            if ts_col is None:
                ts_col = fields[0]

            def pick_col(*names: str) -> Optional[str]:
                for n in names:
                    if n in fields:
                        return n
                return None

            open_col = pick_col("open", "o")
            high_col = pick_col("high", "h")
            low_col = pick_col("low", "l")
            close_col = pick_col("close", "c")
            vol_col = pick_col("volume", "vol", "v")

            for row in reader:
                ts = _coerce_ts_to_ms(row.get(ts_col))
                if ts is None:
                    continue
                # end_ms inclusive (compat)
                if ts < start_ms or ts > end_ms:
                    continue

                def ffloat(x):
                    try:
                        return float(x)
                    except Exception:
                        return 0.0

                yield {
                    "timestamp_ms": ts,
                    "open": ffloat(row.get(open_col)) if open_col else 0.0,
                    "high": ffloat(row.get(high_col)) if high_col else 0.0,
                    "low": ffloat(row.get(low_col)) if low_col else 0.0,
                    "close": ffloat(row.get(close_col)) if close_col else 0.0,
                    "volume": ffloat(row.get(vol_col)) if vol_col else 0.0,
                }


# ============================================================
# Dataset loader (CSV) + fallback (pkl)
# ============================================================

def load_candles_from_path(
    data_path: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    files = _list_csv_files(data_path)
    if files:
        start_ms, end_ms = _date_range_to_ms(date_from, date_to)

        if debug:
            print(f"[ROBUST][DEBUG] csv_files={len(files)}")
            print(f"[ROBUST][DEBUG] start_ms={start_ms} ({_ms_to_iso(start_ms)})")
            print(f"[ROBUST][DEBUG] end_ms  ={end_ms} ({_ms_to_iso(end_ms)})")
            for fp in files[:5]:
                print(f"[ROBUST][DEBUG] file: {fp}")

        candles: List[Dict[str, Any]] = []

        # 1) loader oficial
        try:
            for c in _iter_candles_from_csvs(files, start_ms=start_ms, end_ms=end_ms, quiet=True):
                ts = _pick_ts_ms(c)
                if ts is not None:
                    c["timestamp_ms"] = ts
                candles.append(c)
        except Exception as e:
            if debug:
                print(f"[ROBUST][DEBUG] official loader exception: {e!r}")

        # 2) fallback simple si no matcheó rango / headers raros
        if not candles:
            if debug:
                print("[ROBUST][DEBUG] official loader loaded 0 (NO MATCH IN RANGE). Trying simple CSV parser...")
            for c in _iter_candles_from_simple_csv(files, start_ms=start_ms, end_ms=end_ms, debug=debug):
                candles.append(c)

        if debug:
            if candles:
                ts0 = _pick_ts_ms(candles[0])
                ts1 = _pick_ts_ms(candles[-1])
                print(f"[ROBUST][DEBUG] candles_loaded={len(candles)}")
                if ts0 is not None and ts1 is not None:
                    print(f"[ROBUST][DEBUG] first_ts={ts0} ({_ms_to_iso(ts0)})")
                    print(f"[ROBUST][DEBUG] last_ts ={ts1} ({_ms_to_iso(ts1)})")
            else:
                print("[ROBUST][DEBUG] candles_loaded=0 (NO MATCH IN RANGE)")

            # ✅ EXTRA (pedido): conteo por mes SOLO si hay velas
            if candles:
                from collections import Counter
                ym = Counter()
                for c in candles:
                    ts = _pick_ts_ms(c)
                    if ts:
                        d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                        ym[f"{d.year}-{d.month:02d}"] += 1

                print("[ROBUST][DEBUG] candles per month:")
                for k in sorted(ym):
                    print(f"  {k}: {ym[k]}")

        return candles

    # fallback pkl/pickle (opcional)
    if os.path.isdir(data_path):
        pkl_files = sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".pkl") or f.endswith(".pickle")
        )
        if pkl_files:
            data: List[Dict[str, Any]] = []
            for fp in pkl_files:
                with open(fp, "rb") as f:
                    part = pickle.load(f)
                    if isinstance(part, list):
                        data.extend(part)
            return data

    raise RuntimeError(f"[ROBUST] No CSVs (ni PKL) encontrados en: {data_path}")


# ============================================================
# Walk-forward splits
# ============================================================

def make_walk_forward_splits(
    n: int,
    n_folds: int = 5,
    min_train: int = 10_000,
    min_test: int = 2_000,
) -> List[Tuple[slice, slice]]:
    if n < (min_train + min_test):
        train_end = max(1, int(n * 0.7))
        return [(slice(0, train_end), slice(train_end, n))]

    max_train_end = n - min_test
    train_ends = np.linspace(min_train, max_train_end, n_folds, dtype=int).tolist()

    splits: List[Tuple[slice, slice]] = []
    for te in train_ends:
        tr = slice(0, te)
        ts = slice(te, min(n, te + min_test))
        if (ts.stop - ts.start) >= min_test and (tr.stop - tr.start) >= min_train:
            splits.append((tr, ts))

    if not splits:
        train_end = max(1, int(n * 0.7))
        splits = [(slice(0, train_end), slice(train_end, n))]

    return splits


# ============================================================
# Gates + robust aggregation
# ============================================================

def _normalize_max_drawdown_r(metrics: Dict[str, Any]) -> None:
    if not isinstance(metrics, dict):
        return
    if "max_drawdown_r" not in metrics:
        return
    try:
        val = float(metrics.get("max_drawdown_r"))
    except Exception:
        return
    if val > 1.0:
        metrics["max_drawdown_r"] = val / 100.0


def _normalize_max_dd_r_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if out > 1.0:
        out = out / 100.0
    return out


def _passes_gates(metrics: Dict[str, float], gates: Dict[str, Any]) -> Tuple[bool, str]:
    t = metrics.get("n", metrics.get("trades", 0))
    if t < gates.get("min_trades", 30):
        return False, f"min_trades ({t})"

    max_dd = abs(metrics.get("max_drawdown_r", 0.0))
    if max_dd > gates.get("max_dd_r", 0.2):
        return False, f"max_dd_r ({max_dd:.2f})"

    pf = metrics.get("profit_factor", 0.0)
    if pf < gates["min_pf"]:
        return False, f"min_pf ({pf:.2f})"

    winrate = metrics.get("winrate", 0.0)
    if winrate < gates["min_winrate"]:
        return False, f"min_winrate ({winrate:.2f})"

    return True, ""


def check_filters(metrics: Dict[str, float], filters: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if isinstance(metrics, dict) and "metrics" in metrics and "n" in metrics and isinstance(metrics["metrics"], dict):
        inner = dict(metrics["metrics"])
        inner["trades"] = metrics.get("n", 0)
        metrics = inner

    _normalize_max_drawdown_r(metrics)

    reasons: List[str] = []
    used_keys = ("trades", "max_drawdown_r", "profit_factor", "winrate")
    for key in used_keys:
        if key not in metrics or metrics.get(key) is None:
            reasons.append(f"missing:{key}")
            continue
        try:
            val = float(metrics.get(key))
        except Exception:
            reasons.append(f"nan:{key}")
            continue
        if np.isnan(val):
            reasons.append(f"nan:{key}")
        elif np.isinf(val):
            reasons.append(f"inf:{key}")

    ok, reason = _passes_gates(metrics, filters)
    if not ok and reason:
        reasons.append(reason)

    if reasons:
        return False, reasons
    return True, []


def aggregate_fold_scores(
    fold_scores: List[float],
    lam_std: float = 0.75,
    beta_worst: float = 0.35,
) -> float:
    if not fold_scores:
        return -1e9
    mean_s = float(np.mean(fold_scores))
    std_s = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    worst_s = float(np.min(fold_scores))
    return mean_s - lam_std * std_s + beta_worst * worst_s


def _normalize_score_fallback(mode: Optional[str]) -> str:
    if mode in ("equity", "expectancy", "none"):
        return str(mode)
    return "none"


def _resolve_fold_score(metrics: Dict[str, float], ok: bool, mode: str) -> float:
    base_score = compute_score(metrics) if ok else -1e9
    mode = _normalize_score_fallback(mode)
    if base_score == -1e9 and mode == "equity":
        return float(metrics.get("equity_r", base_score))
    if base_score == -1e9 and mode == "expectancy":
        return float(metrics.get("expectancy", base_score))
    return base_score


def evaluate_params_walk_forward(
    data: Any,
    params: Dict[str, Any],
    backtest_fn: BacktestFn,
    splits: List[Tuple[slice, slice]],
    gates: Dict[str, Any],
    test_slices: Optional[List[Any]] = None,
    feature_caches: Optional[List[FeatureCache]] = None,
    score_fallback: str = "none",
    sample_idx: Optional[int] = None,
) -> EvalResult:
    fold_results: List[FoldResult] = []
    passed = True
    fail_reason: List[str] = []

    for i, (_tr_slice, te_slice) in enumerate(splits):
        test_data = test_slices[i] if test_slices is not None else data[te_slice]
        if feature_caches is not None and i < len(feature_caches):
            set_active_cache(feature_caches[i])
        _set_diag_context(sample_idx, i)
        trades_test = backtest_fn(test_data, params)

        m = compute_metrics_from_trades(trades_test)
        _normalize_max_drawdown_r(m)
        ok, reasons = check_filters(m, gates)
        s = _resolve_fold_score(m, ok, score_fallback)

        fold_results.append(FoldResult(fold_id=i, metrics=m, score=s))
        if not ok:
            passed = False
            for reason in reasons:
                fail_reason.append(f"fold {i}: {reason}")

    fold_scores = [fr.score for fr in fold_results]
    robust = aggregate_fold_scores(
        fold_scores,
        lam_std=gates.get("lam_std", 0.75),
        beta_worst=gates.get("beta_worst", 0.35),
    )

    agg = {
        "folds": len(fold_results),
        "score_mean": float(np.mean(fold_scores)) if fold_scores else -1e9,
        "score_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        "score_worst": float(np.min(fold_scores)) if fold_scores else -1e9,
        "robust_score": float(robust),
    }

    return EvalResult(
        params=params,
        fold_results=fold_results,
        agg=agg,
        robust_score=float(robust),
        passed=passed,
        fail_reason=fail_reason,
    )


# ============================================================
# Backtest REAL (integración) + dummy (compat)
# ============================================================

def _dummy_backtest_fn(_data_slice, _params):
    return []


def _real_backtest_fn(
    data_slice: Any,
    params: Dict[str, Any],
    *,
    base_cfg: Dict[str, Any],
    symbol: Optional[str] = None,
    interval: str = "1m",
    warmup: int = 500,
) -> List[Dict[str, Any]]:
    """
    Ejecuta el backtest REAL sobre un slice de velas, usando config base.
    """
    try:
        import backtest.run_backtest as rb  # type: ignore
    except Exception as e:
        raise RuntimeError(f"[ROBUST] No pude importar backtest.run_backtest: {e!r}")

    candidates = [
        "run_backtest_on_candles",
        "backtest_on_candles",
        "run_on_candles",
        "run_backtest_candles",
    ]

    fn = None
    for name in candidates:
        cand = getattr(rb, name, None)
        if callable(cand):
            fn = cand
            break

    if fn is None:
        raise RuntimeError(
            "[ROBUST] No encontré un entrypoint in-memory en backtest.run_backtest.\n"
            "Busqué: " + ", ".join(candidates) + "\n"
            "Solución: exportá una función tipo run_backtest_on_candles(candles=..., strategy_params=..., base_config=...)"
        )

    res = None
    last_err: Optional[Exception] = None

    # Probar firmas comunes sin romper
    strategy_kwargs = dict(params or {})
    mapped_kwargs = _map_params_to_hybrid_kwargs(params)
    for key, value in mapped_kwargs.items():
        strategy_kwargs.setdefault(key, value)
    cfg_payload = json.loads(json.dumps(base_cfg)) if isinstance(base_cfg, dict) else {}
    cfg_payload.setdefault("params", {})
    cfg_payload["params"].update(params or {})
    cfg_payload["strategy"] = {
        "name": "hybrid_scalper_pro",
        "kwargs": strategy_kwargs,
    }
    cfg_payload["strategy_kwargs"] = strategy_kwargs
    cfg_payload["params"] = dict(cfg_payload.get("params") or {})
    if _diagnostics_enabled():
        sample_idx = _DIAG_CONTEXT.get("sample_idx")
        fold_idx = _DIAG_CONTEXT.get("fold_idx")
        print(
            "[ROBUST][DIAG] "
            f"sample={sample_idx} fold={fold_idx} "
            f"strategy={cfg_payload['strategy']['name']} "
            f"kwargs={_diag_dump(cfg_payload['strategy']['kwargs'])}",
            flush=True,
        )

    _emit_verbose_diagnostics(
        params=params,
        base_cfg=cfg_payload,
        symbol=symbol,
        interval=interval,
    )
    for call in (
        lambda: fn(candles=data_slice, strategy_params=params, base_config=cfg_payload, symbol=symbol, interval=interval, warmup_candles=warmup),
        lambda: fn(candles=data_slice, params=params, base_config=cfg_payload, symbol=symbol, interval=interval, warmup_candles=warmup),
        lambda: fn(data=data_slice, strategy_params=params, base_config=cfg_payload, symbol=symbol, interval=interval, warmup_candles=warmup),
        lambda: fn(data=data_slice, params=params, base_config=cfg_payload, symbol=symbol, interval=interval, warmup_candles=warmup),
        lambda: fn(data_slice, params, base_config=cfg_payload, symbol=symbol, interval=interval, warmup_candles=warmup),
    ):
        try:
            res = call()
            last_err = None
            break
        except TypeError as e:
            last_err = e
        except Exception:
            raise  # error interno real, que suba

    if last_err is not None and res is None:
        raise RuntimeError(f"[ROBUST] No pude llamar al entrypoint '{fn.__name__}' con firmas conocidas: {last_err!r}")

    # Normalización de retorno
    trades: List[Dict[str, Any]] = []
    if isinstance(res, list):
        trades = res
    elif isinstance(res, dict):
        t = res.get("trades") or res.get("trades_list") or res.get("fills") or res.get("executions")
        if isinstance(t, list):
            trades = t
        else:
            trades = []
    elif isinstance(res, tuple) and res:
        for item in res:
            if isinstance(item, list):
                trades = item
                break

    if not isinstance(trades, list):
        return []
    out: List[Dict[str, Any]] = []
    for t in trades:
        if isinstance(t, dict):
            out.append(t)
    return out


# ============================================================
# ✅ Paralelización: Worker globals + init (Windows-safe)
# ============================================================

_WORKER_DATA: Optional[List[Dict[str, Any]]] = None
_WORKER_BASE_CFG: Optional[Dict[str, Any]] = None
_WORKER_SYMBOL: Optional[str] = None
_WORKER_INTERVAL: str = "1m"
_WORKER_WARMUP: int = 500
_WORKER_GATES: Dict[str, Any] = {}
_WORKER_SPLITS: List[Tuple[slice, slice]] = []
_WORKER_USE_DUMMY: bool = False
_WORKER_SCORE_FALLBACK: str = "none"

# ✅ DIAGNOSTICS: contexto por llamada (sample/fold) + estático (window/date range)
_DIAG_CONTEXT: Dict[str, Any] = {}
_DIAG_STATIC_CONTEXT: Dict[str, Any] = {}

# ✅ NUEVO: slice cache por worker (lista de test_slices ya materializados)
_WORKER_TEST_SLICES: Optional[List[Any]] = None  # Any para permitir list/slice/np/etc
_WORKER_FEATURE_CACHES: Optional[List[FeatureCache]] = None

_SPACE_KEYS = [
    "ema_fast",
    "ema_slow",
    "atr_len",
    "sl_atr_mult",
    "tp_atr_mult",
    "rr_min",
    "delta_threshold",
    "delta_rolling_sec",
    "cooldown_sec",
    "max_trades_day",
    "use_time_filter",
    "hour_start",
    "hour_end",
]


def _map_params_to_hybrid_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "ema_fast": "ema_fast",
        "ema_slow": "ema_slow",
        "atr_len": "atr_n",
        "sl_atr_mult": "atr_stop_mult",
        "tp_atr_mult": "atr_trail_mult",
        "max_trades_day": "risk_max_trades",
        "cooldown_sec": "cooldown_after_loss_sec",
    }
    out: Dict[str, Any] = {}
    for key, value in (params or {}).items():
        mapped_key = mapping.get(key)
        if mapped_key:
            out[mapped_key] = value
    return out


def _diagnostics_enabled() -> bool:
    return os.getenv("PIPELINE_VERBOSE_DIAGNOSTICS", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
        "on",
        "ON",
    )


def _set_diag_context(sample_idx: Optional[int], fold_idx: Optional[int]) -> None:
    if not _diagnostics_enabled():
        return
    _DIAG_CONTEXT.clear()
    if sample_idx is not None:
        _DIAG_CONTEXT["sample_idx"] = sample_idx
    if fold_idx is not None:
        _DIAG_CONTEXT["fold_idx"] = fold_idx


def _set_diag_static_context(window: Optional[str], from_date: Optional[str], to_date: Optional[str]) -> None:
    if not _diagnostics_enabled():
        return
    _DIAG_STATIC_CONTEXT.clear()
    if window:
        _DIAG_STATIC_CONTEXT["window"] = window
    if from_date:
        _DIAG_STATIC_CONTEXT["from_date"] = from_date
    if to_date:
        _DIAG_STATIC_CONTEXT["to_date"] = to_date


def _unwrap_sample_params(param_input: Any) -> Tuple[Optional[int], Dict[str, Any]]:
    if isinstance(param_input, tuple) and len(param_input) == 2:
        idx, params = param_input
        if isinstance(params, dict):
            try:
                return (int(idx), params)
            except Exception:
                return (None, params)
    if isinstance(param_input, dict):
        return (None, param_input)
    return (None, {})


def _diag_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def _extract_cfg_param_sources(base_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(base_cfg, dict):
        return {}
    strategy = base_cfg.get("strategy")
    return {
        "cfg.params": base_cfg.get("params"),
        "cfg.strategy.params": strategy.get("params") if isinstance(strategy, dict) else None,
        "cfg.strategy.kwargs": strategy.get("kwargs") if isinstance(strategy, dict) else None,
        "cfg.strategy_config": base_cfg.get("strategy_config"),
        "cfg.strategy_kwargs": base_cfg.get("strategy_kwargs"),
        "cfg.strategy_settings": base_cfg.get("strategy_settings"),
    }


def _pick_effective_params(params: Dict[str, Any], sources: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(params, dict) and any(k in params for k in _SPACE_KEYS):
        source = params
    else:
        source = None
        for key in (
            "cfg.strategy.kwargs",
            "cfg.strategy_kwargs",
            "cfg.params",
            "cfg.strategy.params",
            "cfg.strategy_config",
            "cfg.strategy_settings",
        ):
            candidate = sources.get(key)
            if isinstance(candidate, dict):
                source = candidate
                break
        if source is None:
            source = {}
    return {k: source.get(k) for k in _SPACE_KEYS}


def _emit_verbose_diagnostics(
    *,
    params: Dict[str, Any],
    base_cfg: Optional[Dict[str, Any]],
    symbol: Optional[str],
    interval: str,
) -> None:
    if not _diagnostics_enabled():
        return
    sample_idx = _DIAG_CONTEXT.get("sample_idx")
    fold_idx = _DIAG_CONTEXT.get("fold_idx")
    sources = _extract_cfg_param_sources(base_cfg)
    picked = _pick_effective_params(params, sources)
    print(
        f"[ROBUST][DIAG] sample={sample_idx} fold={fold_idx}",
        flush=True,
    )
    print(f"  params={_diag_dump(params)}", flush=True)
    for key, value in sources.items():
        print(f"  {key}={_diag_dump(value)}", flush=True)
    print(f"  picked_effective={_diag_dump(picked)}", flush=True)
    args_payload = {
        "symbol": symbol,
        "interval": interval,
        "window": _DIAG_STATIC_CONTEXT.get("window"),
        "from_date": _DIAG_STATIC_CONTEXT.get("from_date"),
        "to_date": _DIAG_STATIC_CONTEXT.get("to_date"),
    }
    print(f"  backtest_args={_diag_dump(args_payload)}", flush=True)


def _worker_init(
    data: List[Dict[str, Any]],
    base_cfg: Optional[Dict[str, Any]],
    symbol: Optional[str],
    interval: str,
    warmup: int,
    gates: Dict[str, Any],
    splits: List[Tuple[slice, slice]],
    use_dummy: bool,
    score_fallback: str,
    diag_context: Optional[Dict[str, Any]] = None,
):
    # Se ejecuta 1 vez por proceso
    global _WORKER_DATA, _WORKER_BASE_CFG, _WORKER_SYMBOL, _WORKER_INTERVAL, _WORKER_WARMUP, _WORKER_GATES, _WORKER_SPLITS, _WORKER_USE_DUMMY, _WORKER_TEST_SLICES, _WORKER_FEATURE_CACHES, _WORKER_SCORE_FALLBACK
    _WORKER_DATA = data
    _WORKER_BASE_CFG = base_cfg
    _WORKER_SYMBOL = symbol
    _WORKER_INTERVAL = interval
    _WORKER_WARMUP = int(warmup)
    _WORKER_GATES = gates or {}
    _WORKER_SPLITS = splits or []
    _WORKER_USE_DUMMY = bool(use_dummy)
    _WORKER_SCORE_FALLBACK = _normalize_score_fallback(score_fallback)
    if isinstance(diag_context, dict) and _diagnostics_enabled():
        _DIAG_STATIC_CONTEXT.clear()
        _DIAG_STATIC_CONTEXT.update(diag_context)

    # ✅ Slice cache (reduce overhead de slicing repetido por param)
    try:
        _WORKER_TEST_SLICES = []
        _WORKER_FEATURE_CACHES = []
        for (_tr_slice, te_slice) in _WORKER_SPLITS:
            raw_slice = _WORKER_DATA[te_slice]
            materialized = []
            for idx, c in enumerate(raw_slice):
                row = dict(c)
                row["_idx"] = idx
                materialized.append(row)
            _WORKER_TEST_SLICES.append(materialized)
            _WORKER_FEATURE_CACHES.append(FeatureCache.from_candles(materialized))
    except Exception:
        _WORKER_TEST_SLICES = None
        _WORKER_FEATURE_CACHES = None


def _worker_backtest_fn(data_slice: Any, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if _WORKER_USE_DUMMY:
        return _dummy_backtest_fn(data_slice, params)
    if _WORKER_BASE_CFG is None:
        # No debería pasar si main validó
        return []
    return _real_backtest_fn(
        data_slice,
        params,
        base_cfg=_WORKER_BASE_CFG,
        symbol=_WORKER_SYMBOL,
        interval=_WORKER_INTERVAL,
        warmup=_WORKER_WARMUP,
    )


# ============================================================
# ✅ NUEVO: eval usando slices cache (misma lógica, menos overhead)
# ============================================================

def _evaluate_params_on_cached_test_slices(
    param_input: Any,
) -> Dict[str, Any]:
    global _WORKER_TEST_SLICES, _WORKER_GATES, _WORKER_FEATURE_CACHES, _WORKER_SCORE_FALLBACK

    sample_idx, params = _unwrap_sample_params(param_input)
    if _WORKER_TEST_SLICES is None:
        # fallback a la vieja vía (no debería si init pudo cachear)
        return _worker_eval_one(param_input)

    try:
        fold_results: List[FoldResult] = []
        passed = True
        fail_reason: List[str] = []

        for i, test_data in enumerate(_WORKER_TEST_SLICES):
            if _WORKER_FEATURE_CACHES is not None and i < len(_WORKER_FEATURE_CACHES):
                set_active_cache(_WORKER_FEATURE_CACHES[i])
            _set_diag_context(sample_idx, i)
            trades_test = _worker_backtest_fn(test_data, params)

            m = compute_metrics_from_trades(trades_test)
            _normalize_max_drawdown_r(m)
            ok, reasons = check_filters(m, _WORKER_GATES)
            s = _resolve_fold_score(m, ok, _WORKER_SCORE_FALLBACK)

            fold_results.append(FoldResult(fold_id=i, metrics=m, score=s))
            if not ok:
                passed = False
                for reason in reasons:
                    fail_reason.append(f"fold {i}: {reason}")

        fold_scores = [fr.score for fr in fold_results]
        robust = aggregate_fold_scores(
            fold_scores,
            lam_std=_WORKER_GATES.get("lam_std", 0.75),
            beta_worst=_WORKER_GATES.get("beta_worst", 0.35),
        )

        agg = {
            "folds": len(fold_results),
            "score_mean": float(np.mean(fold_scores)) if fold_scores else -1e9,
            "score_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
            "score_worst": float(np.min(fold_scores)) if fold_scores else -1e9,
            "robust_score": float(robust),
        }

        payload = {
            "params": params,
            "passed": passed,
            "fail_reason": fail_reason,
            "agg": agg,
            "robust_score": float(robust),
            "folds": [
                {"fold_id": fr.fold_id, "score": fr.score, "metrics": fr.metrics}
                for fr in fold_results
            ],
        }
        return payload
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "params": params,
            "passed": False,
            "fail_reason": [f"EXCEPTION:{type(e).__name__}: {e}"],
            "exception": str(e),
            "traceback": tb,
            "agg": {},
            "robust_score": -1e9,
            "folds": [],
        }


def _worker_eval_one(param_input: Any) -> Dict[str, Any]:
    global _WORKER_DATA
    if _WORKER_DATA is None:
        raise RuntimeError("[ROBUST][WORKER] Missing worker data (init not called).")

    sample_idx, params = _unwrap_sample_params(param_input)
    try:
        er = evaluate_params_walk_forward(
            data=_WORKER_DATA,
            params=params,
            backtest_fn=_worker_backtest_fn,
            splits=_WORKER_SPLITS,
            gates=_WORKER_GATES,
            score_fallback=_WORKER_SCORE_FALLBACK,
            sample_idx=sample_idx,
        )
    except MemoryError:
        # 🔴 OOM explícito
        import traceback
        tb = traceback.format_exc()
        return {
            "params": params,
            "passed": False,
            "fail_reason": ["OOM"],
            "exception": "MemoryError",
            "traceback": tb,
            "agg": {},
            "robust_score": -1e9,
            "folds": [],
        }
    except Exception as e:
        # 🔴 cualquier otro crash del worker
        import traceback
        tb = traceback.format_exc()
        return {
            "params": params,
            "passed": False,
            "fail_reason": [f"EXCEPTION:{type(e).__name__}: {e}"],
            "exception": str(e),
            "traceback": tb,
            "agg": {},
            "robust_score": -1e9,
            "folds": [],
        }

    return {
        "params": er.params,
        "passed": er.passed,
        "fail_reason": er.fail_reason,
        "agg": er.agg,
        "robust_score": er.robust_score,
        "folds": [
            {"fold_id": fr.fold_id, "score": fr.score, "metrics": fr.metrics}
            for fr in er.fold_results
        ],
    }


# ============================================================
# ✅ FIX: dejar UNA sola definición de _worker_eval_batch
#    - usa cache (cuando existe)
#    - mantiene OOM/EXCEPTION por param (no revienta todo el batch)
# ============================================================

def _worker_eval_batch(params_batch: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in params_batch:
        try:
            out.append(_evaluate_params_on_cached_test_slices(p))
        except MemoryError:
            import traceback
            tb = traceback.format_exc()
            _sample_idx, params = _unwrap_sample_params(p)
            out.append({
                "params": params,
                "passed": False,
                "fail_reason": ["OOM"],
                "exception": "MemoryError",
                "traceback": tb,
                "agg": {},
                "robust_score": -1e9,
                "folds": [],
            })
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            _sample_idx, params = _unwrap_sample_params(p)
            out.append({
                "params": params,
                "passed": False,
                "fail_reason": [f"EXCEPTION:{type(e).__name__}: {e}"],
                "exception": str(e),
                "traceback": tb,
                "agg": {},
                "robust_score": -1e9,
                "folds": [],
            })
    return out


# ============================================================
# Search loop (secuencial + paralelo sin perder features)
# ============================================================

def run_robust_search(
    data: Any,
    backtest_fn: BacktestFn,
    n_samples: int = 200,
    seed: int = 1337,
    n_folds: int = 5,
    min_train: int = 10_000,
    min_test: int = 2_000,
    gates: Optional[Dict[str, Any]] = None,
    top_k: int = 20,
    window: Optional[str] = None,
    params_list: Optional[List[Dict[str, Any]]] = None,
    score_fallback: str = "none",
) -> Tuple[List[EvalResult], List[EvalResult]]:
    """
    Mantiene compat total.
    Si ROBUST_WORKERS>1 y no es un callable raro -> usa multiprocessing con ProcessPoolExecutor.
    """
    gates = gates or {
        "min_trades": 30,
        "min_r_obs": 200,
        "max_dd_r": 0.2,
        "min_pf": 1.05,
        "min_winrate": 0.35,
        "lam_std": 0.75,
        "beta_worst": 0.35,
    }
    normalized_max_dd = _normalize_max_dd_r_value(gates.get("max_dd_r"))
    if normalized_max_dd is not None:
        gates["max_dd_r"] = normalized_max_dd
    FAIL_COUNTS.clear()

    splits = make_walk_forward_splits(len(data), n_folds=n_folds, min_train=min_train, min_test=min_test)

    # ---------- generar params determinísticamente ----------
    if params_list is None:
        rng = np.random.default_rng(seed)
        phase = normalize_phase(None)
        space = param_space_for_phase(phase)
        seen = set()
        params_list = []
        for _i in range(n_samples):
            params = sample_params(space, rng=rng, hard_constraints=True)
            key = json.dumps(params, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            params_list.append(params)

    # ---------- decidir si paralelizar ----------
    workers_env = os.getenv("ROBUST_WORKERS", "").strip()
    try:
        workers = int(workers_env) if workers_env else 0
    except Exception:
        workers = 0

    # Si no seteaste workers -> por defecto usa cores-1 (mín 1) SOLO si pedís explícito ROBUST_PARALLEL=1
    parallel_flag = os.getenv("ROBUST_PARALLEL", "").strip()
    if workers <= 0 and parallel_flag in ("1", "true", "TRUE", "yes", "YES", "on", "ON"):
        try:
            cpu = os.cpu_count() or 2
        except Exception:
            cpu = 2
        workers = max(1, cpu - 1)

    # Si workers<=1 -> secuencial (idéntico a tu lógica actual)
    if workers <= 1:
        results: List[EvalResult] = []
        feature_caches: List[FeatureCache] = []
        test_slices: List[Any] = []
        for (_tr_slice, te_slice) in splits:
            raw_slice = data[te_slice]
            materialized = []
            for idx, c in enumerate(raw_slice):
                row = dict(c)
                row["_idx"] = idx
                materialized.append(row)
            test_slices.append(materialized)
            feature_caches.append(FeatureCache.from_candles(materialized))
        done_params = 0
        total_params = len(params_list)
        passed_params = 0
        failed_params = 0
        hb_started_at = time.monotonic()
        hb_last_emit = hb_started_at
        window_label = window or "unknown"
        failed_samples: List[Dict[str, Any]] = []
        for sample_idx, params in enumerate(params_list):
            try:
                er = evaluate_params_walk_forward(
                    data=data,
                    params=params,
                    backtest_fn=backtest_fn,
                    splits=splits,
                    gates=gates,
                    test_slices=test_slices,
                    feature_caches=feature_caches,
                    score_fallback=score_fallback,
                    sample_idx=sample_idx,
                )
                results.append(er)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                results.append(EvalResult(
                    params=params,
                    fold_results=[],
                    agg={},
                    robust_score=-1e9,
                    passed=False,
                    fail_reason=[f"EXCEPTION:{type(e).__name__}: {e}"],
                    exception=str(e),
                    traceback=tb,
                ))
            done_params += 1
            if results[-1].passed:
                passed_params += 1
            else:
                failed_params += 1
                reasons = _normalize_fail_reasons(results[-1].fail_reason)
                _record_fail_reasons(reasons)
                if len(failed_samples) < 20:
                    failed_samples.append({
                        "params": results[-1].params,
                        "metrics": _format_fail_metrics_from_folds(results[-1].fold_results),
                        "reasons": reasons,
                    })
            hb_last_emit = _emit_heartbeat(
                done_params,
                total_params,
                passed_params,
                failed_params,
                hb_started_at,
                hb_last_emit,
                window_label,
                int(seed),
            )

        results.sort(
            key=lambda r: (r.robust_score, r.agg.get("score_worst", -1e9), -r.agg.get("score_std", 1e9)),
            reverse=True,
        )
        debug_path = os.path.join("results", "debug", f"fails_{window_label}_seed{seed}.json")
        _atomic_write_json(debug_path, failed_samples)
        return results[:top_k], results

    # ---------- paralelo: usamos workers global-init (NO pierde features) ----------
    use_dummy = bool(os.getenv("ROBUST_USE_DUMMY", "0").strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON"))

    base_cfg = None
    symbol = None
    interval = "1m"
    warmup = 500

    # --- safety defaults ---
    param_timeout = float(os.getenv("ROBUST_PARAM_TIMEOUT", "0"))  # 0 = sin timeout
    batch_size = int(os.getenv("ROBUST_BATCH_SIZE", "1"))          # 1 = modo actual

    # ✅ NUEVO: batch size via env (si no está, 1 => comportamiento idéntico al anterior)
    batch_env = os.getenv("ROBUST_BATCH_SIZE", "").strip()
    try:
        batch_size = int(batch_env) if batch_env else 1
    except Exception:
        batch_size = 1
    if batch_size < 1:
        batch_size = 1

    results_payloads: List[Dict[str, Any]] = []
    failed_samples: List[Dict[str, Any]] = []

    # Context spawn = Windows safe; en Linux también funciona.
    ctx = mp.get_context("spawn")

    progress_every = int(os.getenv("ROBUST_PROGRESS_EVERY", "25") or "25")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(
            data,         # se picklea 1 vez por worker
            _WORKER_BASE_CFG,  # si no lo seteaste, queda None (main lo setea en el wrapper)
            _WORKER_SYMBOL,
            _WORKER_INTERVAL,
            _WORKER_WARMUP,
            gates,
            splits,
            use_dummy,
            score_fallback,
            dict(_DIAG_STATIC_CONTEXT),
        ),
    ) as ex:
        # ============================================================
        # ✅ FIX #2 + #3: flujo único (sin duplicar submit) y acumulación robusta
        # ============================================================

        fut_to_batch: Dict[Any, List[Dict[str, Any]]] = {}

        params_payloads = list(enumerate(params_list))
        if batch_size == 1:
            futs = [ex.submit(_evaluate_params_on_cached_test_slices, p) for p in params_payloads]
        else:
            batches: List[List[Any]] = []
            for i in range(0, len(params_payloads), batch_size):
                batches.append(params_payloads[i:i + batch_size])
            futs = [ex.submit(_worker_eval_batch, b) for b in batches]
            fut_to_batch = {fut: b for fut, b in zip(futs, batches)}

        done_params = 0
        total_params = len(params_list)
        passed_params = 0
        failed_params = 0
        hb_started_at = time.monotonic()
        hb_last_emit = hb_started_at
        window_label = window or "unknown"

        for fut in as_completed(futs):
            try:
                res = fut.result(timeout=param_timeout) if param_timeout > 0 else fut.result()

                # res puede ser Dict (batch_size==1) o List[Dict] (batch_size>1)
                if isinstance(res, list):
                    results_payloads.extend(res)
                    done_params += len(res)
                    passed_in_batch = sum(1 for item in res if item.get("passed"))
                    passed_params += passed_in_batch
                    failed_params += len(res) - passed_in_batch
                    for item in res:
                        if not item.get("passed"):
                            reasons = _normalize_fail_reasons(item.get("fail_reason"))
                            _record_fail_reasons(reasons)
                            if len(failed_samples) < 20:
                                failed_samples.append({
                                    "params": item.get("params"),
                                    "metrics": _format_fail_metrics_from_payload_folds(item.get("folds", [])),
                                    "reasons": reasons,
                                })
                else:
                    results_payloads.append(res)
                    done_params += 1
                    if res.get("passed"):
                        passed_params += 1
                    else:
                        failed_params += 1
                        reasons = _normalize_fail_reasons(res.get("fail_reason"))
                        _record_fail_reasons(reasons)
                        if len(failed_samples) < 20:
                            failed_samples.append({
                                "params": res.get("params"),
                                "metrics": _format_fail_metrics_from_payload_folds(res.get("folds", [])),
                                "reasons": reasons,
                            })

            except FuturesTimeoutError:
                # Timeout: si era batch, marcamos todos los params del batch
                if fut in fut_to_batch:
                    b = fut_to_batch[fut]
                    for p in b:
                        _sample_idx, params = _unwrap_sample_params(p)
                        payload = {
                            "params": params,
                            "passed": False,
                            "fail_reason": ["TIMEOUT"],
                            "agg": {},
                            "robust_score": -1e9,
                            "folds": [],
                        }
                        results_payloads.append(payload)
                        reasons = _normalize_fail_reasons(payload.get("fail_reason"))
                        _record_fail_reasons(reasons)
                        if len(failed_samples) < 20:
                            failed_samples.append({
                                "params": payload.get("params"),
                                "metrics": [],
                                "reasons": reasons,
                            })
                    done_params += len(b)
                    failed_params += len(b)
                else:
                    payload = {
                        "params": {},
                        "passed": False,
                        "fail_reason": ["TIMEOUT"],
                        "agg": {},
                        "robust_score": -1e9,
                        "folds": [],
                    }
                    results_payloads.append(payload)
                    reasons = _normalize_fail_reasons(payload.get("fail_reason"))
                    _record_fail_reasons(reasons)
                    if len(failed_samples) < 20:
                        failed_samples.append({
                            "params": payload.get("params"),
                            "metrics": [],
                            "reasons": reasons,
                        })
                    done_params += 1
                    failed_params += 1

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                # Exception: si era batch, marcamos todos los params del batch
                if fut in fut_to_batch:
                    b = fut_to_batch[fut]
                    for p in b:
                        _sample_idx, params = _unwrap_sample_params(p)
                        payload = {
                            "params": params,
                            "passed": False,
                            "fail_reason": [f"EXCEPTION:{type(e).__name__}: {e}"],
                            "exception": str(e),
                            "traceback": tb,
                            "agg": {},
                            "robust_score": -1e9,
                            "folds": [],
                        }
                        results_payloads.append(payload)
                        reasons = _normalize_fail_reasons(payload.get("fail_reason"))
                        _record_fail_reasons(reasons)
                        if len(failed_samples) < 20:
                            failed_samples.append({
                                "params": payload.get("params"),
                                "metrics": [],
                                "reasons": reasons,
                            })
                    done_params += len(b)
                    failed_params += len(b)
                else:
                    payload = {
                        "params": {},
                        "passed": False,
                        "fail_reason": [f"EXCEPTION:{type(e).__name__}: {e}"],
                        "exception": str(e),
                        "traceback": tb,
                        "agg": {},
                        "robust_score": -1e9,
                        "folds": [],
                    }
                    results_payloads.append(payload)
                    reasons = _normalize_fail_reasons(payload.get("fail_reason"))
                    _record_fail_reasons(reasons)
                    if len(failed_samples) < 20:
                        failed_samples.append({
                            "params": payload.get("params"),
                            "metrics": [],
                            "reasons": reasons,
                        })
                    done_params += 1
                    failed_params += 1

            if progress_every > 0 and (done_params % progress_every == 0):
                print(f"[ROBUST][PAR] done {done_params}/{total_params} (batch_size={batch_size})")
            hb_last_emit = _emit_heartbeat(
                done_params,
                total_params,
                passed_params,
                failed_params,
                hb_started_at,
                hb_last_emit,
                window_label,
                int(seed),
            )

    # Convertir payloads -> EvalResult (misma API)
    results: List[EvalResult] = []
    for pl in results_payloads:
        fold_results = []
        for fr in pl.get("folds", []):
            fold_results.append(FoldResult(
                fold_id=int(fr.get("fold_id", 0)),
                metrics=fr.get("metrics", {}) or {},
                score=float(fr.get("score", -1e9)),
            ))
        results.append(EvalResult(
            params=pl.get("params", {}) or {},
            fold_results=fold_results,
            agg=pl.get("agg", {}) or {},
            robust_score=float(pl.get("robust_score", -1e9)),
            passed=bool(pl.get("passed", False)),
            fail_reason=_normalize_fail_reasons(pl.get("fail_reason")),
            exception=str(pl.get("exception", "") or ""),
            traceback=str(pl.get("traceback", "") or ""),
        ))

    results.sort(
        key=lambda r: (r.robust_score, r.agg.get("score_worst", -1e9), -r.agg.get("score_std", 1e9)),
        reverse=True,
    )
    debug_path = os.path.join("results", "debug", f"fails_{window_label}_seed{seed}.json")
    _atomic_write_json(debug_path, failed_samples)
    return results[:top_k], results


def save_results_json(path: str, results: List[EvalResult], meta: Optional[Dict[str, Any]] = None) -> None:
    """Guarda resultados robust.

    Nota: mantenemos compat hacia atrás (dict con arrays), pero agregamos:
      - robust_score en el nivel raíz (para que post-analysis no dependa de agg)
      - meta con pipeline_phase + space_keys (para auditar qué se sampleó)
    """
    meta_payload = _normalize_meta(meta)
    payload: Dict[str, Any] = {
        "params": [],
        "passed": [],
        "fail_reason": [],
        "robust_score": [],
        "agg": [],
        "folds": [],
        "meta": meta_payload,
    }
    exceptions: List[str] = []
    tracebacks: List[str] = []
    for r in results:
        payload["params"].append(r.params)
        payload["passed"].append(r.passed)
        payload["fail_reason"].append(r.fail_reason)
        payload["robust_score"].append(r.robust_score)
        payload["agg"].append(r.agg)
        payload["folds"].append([
            {"fold_id": fr.fold_id, "score": fr.score, "metrics": fr.metrics}
            for fr in r.fold_results
        ])
        exceptions.append(r.exception or "")
        tracebacks.append(r.traceback or "")
    if any(exceptions):
        payload["exception"] = exceptions
    if any(tracebacks):
        payload["traceback"] = tracebacks
    _atomic_write_json(path, payload)


def _normalize_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    payload = dict(meta)
    space_keys = payload.get("space_keys")
    if isinstance(space_keys, list):
        payload["space_keys"] = sorted({str(k) for k in space_keys})
    return payload


def _parse_env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _parse_env_float(name: str) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _resolve_gates(
    base_cfg: Optional[Dict[str, Any]],
    cli_min_trades: Optional[int],
    cli_min_r_obs: Optional[int],
    cli_min_pf: Optional[float],
    cli_min_winrate: Optional[float],
    cli_max_dd_r: Optional[float],
) -> Dict[str, Any]:
    gates = {
        "min_trades": 30,
        "min_r_obs": 200,
        "max_dd_r": 0.2,
        "min_pf": 1.05,
        "min_winrate": 0.35,
        "lam_std": 0.75,
        "beta_worst": 0.35,
    }
    if isinstance(base_cfg, dict):
        base_gates = base_cfg.get("gates")
        if isinstance(base_gates, dict):
            gates.update(base_gates)

    env_min_trades = _parse_env_int("PIPELINE_MIN_TRADES")
    if env_min_trades is not None:
        gates["min_trades"] = env_min_trades
    env_min_r_obs = _parse_env_int("PIPELINE_MIN_R_OBS")
    if env_min_r_obs is not None:
        gates["min_r_obs"] = env_min_r_obs
    env_min_pf = _parse_env_float("PIPELINE_MIN_PF")
    if env_min_pf is not None:
        gates["min_pf"] = env_min_pf
    env_min_winrate = _parse_env_float("PIPELINE_MIN_WINRATE")
    if env_min_winrate is not None:
        gates["min_winrate"] = env_min_winrate

    if cli_min_trades is not None:
        gates["min_trades"] = int(cli_min_trades)
    if cli_min_r_obs is not None:
        gates["min_r_obs"] = int(cli_min_r_obs)
    if cli_min_pf is not None:
        gates["min_pf"] = float(cli_min_pf)
    if cli_min_winrate is not None:
        gates["min_winrate"] = float(cli_min_winrate)
    if cli_max_dd_r is not None:
        gates["max_dd_r"] = float(cli_max_dd_r)

    normalized_max_dd = _normalize_max_dd_r_value(gates.get("max_dd_r"))
    if normalized_max_dd is not None:
        gates["max_dd_r"] = normalized_max_dd

    return gates


def _atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dir_name = os.path.dirname(path) or "."
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dir_name, delete=False) as tmp:
            tmp_path = tmp.name
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ============================================================
# ✅ Wrapper paralelo “correcto” (sin romper features)
#    (main lo usa si workers>1)
# ============================================================

def run_robust_search_parallel(
    *,
    data: List[Dict[str, Any]],
    base_cfg: Dict[str, Any],
    symbol: Optional[str],
    interval: str,
    warmup: int,
    n_samples: int,
    seed: int,
    n_folds: int,
    min_train: int,
    min_test: int,
    gates: Optional[Dict[str, Any]],
    top_k: int,
    workers: int,
    use_dummy: bool = False,
    batch_size: int = 8,  # ✅ NUEVO: batch params
    window: Optional[str] = None,
    params_list: Optional[List[Dict[str, Any]]] = None,
    score_fallback: str = "none",
) -> Tuple[List[EvalResult], List[EvalResult]]:
    gates = gates or {
        "min_trades": 30,
        "min_r_obs": 200,
        "max_dd_r": 0.2,
        "min_pf": 1.05,
        "min_winrate": 0.35,
        "lam_std": 0.75,
        "beta_worst": 0.35,
    }
    normalized_max_dd = _normalize_max_dd_r_value(gates.get("max_dd_r"))
    if normalized_max_dd is not None:
        gates["max_dd_r"] = normalized_max_dd
    FAIL_COUNTS.clear()

    splits = make_walk_forward_splits(len(data), n_folds=n_folds, min_train=min_train, min_test=min_test)

    # generar params determinístico
    if params_list is None:
        rng = np.random.default_rng(seed)
        phase = normalize_phase(None)
        space = param_space_for_phase(phase)
        seen = set()
        params_list = []
        for _i in range(n_samples):
            params = sample_params(space, rng=rng, hard_constraints=True)
            key = json.dumps(params, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            params_list.append(params)

    # Windows-safe
    ctx = mp.get_context("spawn")
    progress_every = int(os.getenv("ROBUST_PROGRESS_EVERY", "25") or "25")

    # ✅ env override (si querés setearlo sin tocar CLI)
    env_bs = os.getenv("ROBUST_BATCH_SIZE", "").strip()
    if env_bs:
        try:
            batch_size = int(env_bs)
        except Exception:
            pass
    if batch_size < 1:
        batch_size = 1

    results_payloads: List[Dict[str, Any]] = []
    passed_params = 0
    failed_params = 0
    hb_started_at = time.monotonic()
    hb_last_emit = hb_started_at
    window_label = window or "unknown"
    failed_samples: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(
            data,
            base_cfg,
            symbol,
            interval,
            int(warmup),
            gates,
            splits,
            bool(use_dummy),
            score_fallback,
            dict(_DIAG_STATIC_CONTEXT),
        ),
    ) as ex:
        params_payloads = list(enumerate(params_list))
        if batch_size == 1:
            futs = [ex.submit(_evaluate_params_on_cached_test_slices, p) for p in params_payloads]
            done = 0
            for fut in as_completed(futs):
                done += 1
                res = fut.result()
                results_payloads.append(res)
                if res.get("passed"):
                    passed_params += 1
                else:
                    failed_params += 1
                    reasons = _normalize_fail_reasons(res.get("fail_reason"))
                    _record_fail_reasons(reasons)
                    if len(failed_samples) < 20:
                        failed_samples.append({
                            "params": res.get("params"),
                            "metrics": _format_fail_metrics_from_payload_folds(res.get("folds", [])),
                            "reasons": reasons,
                        })
                if progress_every > 0 and (done % progress_every == 0):
                    print(f"[ROBUST][PAR] done {done}/{len(futs)}")
                hb_last_emit = _emit_heartbeat(
                    done,
                    len(futs),
                    passed_params,
                    failed_params,
                    hb_started_at,
                    hb_last_emit,
                    window_label,
                    int(seed),
                )
        else:
            batches: List[List[Any]] = []
            for i in range(0, len(params_payloads), batch_size):
                batches.append(params_payloads[i:i + batch_size])

            futs = [ex.submit(_worker_eval_batch, b) for b in batches]
            done_params = 0
            total_params = len(params_list)

            for fut in as_completed(futs):
                batch_payloads = fut.result()
                for pl in batch_payloads:
                    results_payloads.append(pl)
                passed_in_batch = sum(1 for item in batch_payloads if item.get("passed"))
                done_params += len(batch_payloads)
                passed_params += passed_in_batch
                failed_params += len(batch_payloads) - passed_in_batch
                for pl in batch_payloads:
                    if not pl.get("passed"):
                        reasons = _normalize_fail_reasons(pl.get("fail_reason"))
                        _record_fail_reasons(reasons)
                        if len(failed_samples) < 20:
                            failed_samples.append({
                                "params": pl.get("params"),
                                "metrics": _format_fail_metrics_from_payload_folds(pl.get("folds", [])),
                                "reasons": reasons,
                            })
                if progress_every > 0 and (done_params % progress_every == 0):
                    print(f"[ROBUST][PAR] done {done_params}/{total_params} (batch_size={batch_size})")
                hb_last_emit = _emit_heartbeat(
                    done_params,
                    total_params,
                    passed_params,
                    failed_params,
                    hb_started_at,
                    hb_last_emit,
                    window_label,
                    int(seed),
                )

    # payloads -> EvalResult
    results: List[EvalResult] = []
    for pl in results_payloads:
        fold_results = []
        for fr in pl.get("folds", []):
            fold_results.append(FoldResult(
                fold_id=int(fr.get("fold_id", 0)),
                metrics=fr.get("metrics", {}) or {},
                score=float(fr.get("score", -1e9)),
            ))
        results.append(EvalResult(
            params=pl.get("params", {}) or {},
            fold_results=fold_results,
            agg=pl.get("agg", {}) or {},
            robust_score=float(pl.get("robust_score", -1e9)),
            passed=bool(pl.get("passed", False)),
            fail_reason=_normalize_fail_reasons(pl.get("fail_reason")),
            exception=str(pl.get("exception", "") or ""),
            traceback=str(pl.get("traceback", "") or ""),
        ))

    results.sort(
        key=lambda r: (r.robust_score, r.agg.get("score_worst", -1e9), -r.agg.get("score_std", 1e9)),
        reverse=True,
    )
    debug_path = os.path.join("results", "debug", f"fails_{window_label}_seed{seed}.json")
    _atomic_write_json(debug_path, failed_samples)
    return results[:top_k], results


def _load_survivor_params(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("survivors"), list):
        payload = payload["survivors"]

    params_list: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("params"), dict):
                params_list.append(item["params"])
            elif isinstance(item, dict):
                params_list.append(item)

    seen = set()
    uniq: List[Dict[str, Any]] = []
    for params in params_list:
        key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(params)
    return uniq


def _save_survivors(path: str, results: List[EvalResult], meta: Optional[Dict[str, Any]] = None) -> None:
    meta_payload = _normalize_meta(meta)
    payload = []
    for r in results:
        if not r.passed:
            continue
        payload.append({
            "params": r.params,
            "robust_score": r.robust_score,
            "agg": r.agg,
            "folds": [
                {"fold_id": fr.fold_id, "score": fr.score, "metrics": fr.metrics}
                for fr in r.fold_results
            ],
            "meta": meta_payload,
        })
    _atomic_write_json(path, payload)


# ============================================================
# CLI runner
# ============================================================

def main():
    ap = argparse.ArgumentParser("robust_optimizer")
    ap.add_argument("--data", required=True, help="dataset path (folder with CSVs)")
    ap.add_argument("--out", default=None, help="output json")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--folds", type=int, default=3, help="folds walk-forward (SEARCH default=3, REVALIDATE recomendado=5)")

    ap.add_argument("--window", default=None, help="YYYY-MM_YYYY-MM (ej: 2020-01_2021-12)")
    ap.add_argument("--from-date", default=None, help="YYYY-MM-DD (compat)")
    ap.add_argument("--to-date", default=None, help="YYYY-MM-DD (compat)")

    ap.add_argument("--debug-loader", action="store_true", help="imprime diagnóstico del loader")
    ap.add_argument("--min-candles", type=int, default=1, help="si carga menos de esto, falla (exit!=0)")

    # ✅ nuevo: permite forzar dummy para debugging, pero por defecto usamos REAL
    ap.add_argument("--use-dummy", action="store_true", help="usa dummy backtest (solo para debug)")

    # ✅ FIX: config base requerida para REAL in-memory
    ap.add_argument("--base-config", default=None, help="config base (json) para backtest REAL in-memory")
    ap.add_argument("--symbol", default=None, help="override symbol (ej: SOLUSDT)")
    ap.add_argument("--interval", default="1m", help="interval (default 1m)")
    ap.add_argument("--warmup", type=int, default=500, help="candles warmup")
    ap.add_argument("--min-train", type=int, default=10000, help="min candles for train split")
    ap.add_argument("--min-test", type=int, default=2000, help="min candles for test split")
    ap.add_argument("--min-trades", type=int, default=None, help="override gate min_trades")
    ap.add_argument("--min-r-obs", type=int, default=None, help="override gate min_r_obs")
    ap.add_argument("--min-pf", type=float, default=None, help="override gate min_pf")
    ap.add_argument("--min-winrate", type=float, default=None, help="override gate min_winrate")
    ap.add_argument("--max-dd-r", type=float, default=None, help="override gate max_dd_r (ratio 0..1)")

    # ✅ paralelo (opcionales)
    ap.add_argument("--workers", type=int, default=0, help="workers para paralelo (0=auto/env/disabled)")
    ap.add_argument("--parallel", action="store_true", help="habilita paralelo (si workers>1 o env ROBUST_WORKERS)")

    # ✅ NUEVO: batch-size explícito (sin obligarte a usar env)
    ap.add_argument("--batch-size", type=int, default=0, help="batch params por task (0=auto/env default 8)")
    ap.add_argument("--save-survivors", default=None, help="path JSON para guardar survivors (SEARCH)")
    ap.add_argument("--revalidate", default=None, help="path JSON con survivors para revalidar (solo evalúa esos params)")
    ap.add_argument("--score-fallback", choices=("none", "equity", "expectancy"), default="none")

    args = ap.parse_args()

    out_path = (args.out or "").strip()
    meta: Dict[str, Any] = {
        "symbol": args.symbol,
        "interval": args.interval,
        "window": args.window,
        "seed": int(args.seed),
        "samples": int(args.samples),
        "min_train": int(args.min_train),
        "min_test": int(args.min_test),
        "score_fallback": _normalize_score_fallback(args.score_fallback),
    }

    try:
        date_from = args.from_date
        date_to = args.to_date
        if args.window:
            date_from, date_to = window_to_dates(args.window)

        if not date_from or not date_to:
            raise SystemExit("[ROBUST] Missing date range. Provide --window OR (--from-date AND --to-date).")

        window_label = args.window or f"{date_from[:7]}_{date_to[:7]}"
        _set_diag_static_context(window_label, date_from, date_to)
        out_path = (args.out or "").strip()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        else:
            out_path = os.path.join("results", "robust", f"robust_{window_label}_seed{int(args.seed)}.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        print("[ROBUST] loading dataset...")
        print(f"[ROBUST] date_from={date_from} date_to={date_to} window={args.window!r}")

        data = load_candles_from_path(
            args.data,
            date_from=date_from,
            date_to=date_to,
            debug=bool(args.debug_loader),
        )
        print(f"[ROBUST] data size = {len(data)}")

        if len(data) < int(args.min_candles):
            print(f"[ROBUST][ERROR] Loaded {len(data)} candles (< min-candles={args.min_candles}).")
            print("[ROBUST][ERROR] Revisá: ruta --data, CSVs, timestamps, o si el rango window está cubierto.")
            raise SystemExit(2)

        if args.use_dummy:
            print("[ROBUST][WARN] Using _dummy_backtest_fn => resultados NO representan performance real todavía.")
            base_cfg = None
        else:
            if not args.base_config:
                raise SystemExit("[ROBUST] Missing --base-config (required for REAL in-memory backtest).")
            with open(args.base_config, "r", encoding="utf-8") as f:
                base_cfg = json.load(f)
            print("[ROBUST] Using REAL backtest function (in-memory).")

        gates = _resolve_gates(
            base_cfg,
            args.min_trades,
            args.min_r_obs,
            args.min_pf,
            args.min_winrate,
            args.max_dd_r,
        )

        # resolver workers
        workers = int(args.workers or 0)
        if workers <= 0:
            envw = os.getenv("ROBUST_WORKERS", "").strip()
            try:
                workers = int(envw) if envw else 0
            except Exception:
                workers = 0
        if workers <= 0 and (args.parallel or (os.getenv("ROBUST_PARALLEL", "").strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON"))):
            cpu = os.cpu_count() or 2
            workers = max(1, cpu - 1)

        # ✅ batch size
        batch_size = int(args.batch_size or 0)
        if batch_size <= 0:
            env_bs = os.getenv("ROBUST_BATCH_SIZE", "").strip()
            if env_bs:
                try:
                    batch_size = int(env_bs)
                except Exception:
                    batch_size = 0
        if batch_size <= 0:
            batch_size = 8  # default recomendado
        if batch_size < 1:
            batch_size = 1

        # backtest_fn (secuencial) mantiene compat
        if args.use_dummy:
            backtest_fn: BacktestFn = _dummy_backtest_fn
        else:
            assert base_cfg is not None

            def backtest_fn(data_slice: Any, params: Dict[str, Any]) -> List[Dict[str, Any]]:
                return _real_backtest_fn(
                    data_slice,
                    params,
                    base_cfg=base_cfg,
                    symbol=args.symbol,
                    interval=args.interval,
                    warmup=int(args.warmup),
                )

        folds = max(1, int(args.folds))
        revalidate_params: Optional[List[Dict[str, Any]]] = None
        if args.revalidate:
            revalidate_params = _load_survivor_params(args.revalidate)
            if not revalidate_params:
                raise SystemExit(f"[ROBUST] No params found in --revalidate={args.revalidate}")

        # ejecutar
        if (not args.use_dummy) and workers > 1:
            assert base_cfg is not None
            print(f"[ROBUST] Parallel enabled: workers={workers} (spawn-safe) batch_size={batch_size}")
            top_results, all_results = run_robust_search_parallel(
                data=data,
                base_cfg=base_cfg,
                symbol=args.symbol,
                interval=args.interval,
                warmup=int(args.warmup),
                n_samples=int(args.samples if not revalidate_params else len(revalidate_params)),
                seed=int(args.seed),
                n_folds=folds,
                min_train=int(args.min_train),
                min_test=int(args.min_test),
                gates=gates,
                top_k=len(revalidate_params) if revalidate_params else 20,
                workers=workers,
                use_dummy=bool(args.use_dummy),
                batch_size=int(batch_size),
                window=window_label,
                params_list=revalidate_params,
                score_fallback=args.score_fallback,
            )
        else:
            if workers > 1 and args.use_dummy:
                print("[ROBUST][WARN] Parallel + dummy: usando modo secuencial (dummy es demasiado rápido y el overhead domina).")
            top_results, all_results = run_robust_search(
                data=data,
                backtest_fn=backtest_fn,
                n_samples=int(args.samples if not revalidate_params else len(revalidate_params)),
                seed=args.seed,
                n_folds=folds,
                min_train=int(args.min_train),
                min_test=int(args.min_test),
                gates=gates,
                window=window_label,
                params_list=revalidate_params,
                score_fallback=args.score_fallback,
            )

        print(f"[ROBUST] finished, top={len(top_results)}")
        # meta audit: qué phase y qué keys se samplearon (crítico para congelar A→B)
        _phase = normalize_phase(None)
        _space = param_space_for_phase(_phase)
        meta.update({
            "pipeline_phase": _phase,
            "space_keys": sorted(list(_space.keys())),
            "strategy_kwargs_effective": sorted(_SPACE_KEYS),
            "workers": int(workers),
            "batch_size": int(batch_size),
            "folds": int(folds),
            "mode": "revalidate" if revalidate_params else "search",
            "gates": {
                "min_trades": int(gates.get("min_trades", 30)),
                "min_r_obs": int(gates.get("min_r_obs", 200)),
                "min_pf": float(gates.get("min_pf", 1.05)),
                "min_winrate": float(gates.get("min_winrate", 0.35)),
                "max_dd_r": float(gates.get("max_dd_r", 0.2)),
            },
        })
        meta = _normalize_meta(meta)
        if revalidate_params:
            passed_results = [r for r in all_results if r.passed]
            save_results_json(out_path, passed_results, meta=meta)
        else:
            save_results_json(out_path, top_results, meta=meta)
            if args.save_survivors:
                _save_survivors(args.save_survivors, all_results, meta=meta)
        print(f"[ROBUST] saved -> {out_path}")
    except Exception as e:
        import traceback
        if not out_path:
            out_path = os.path.join("results", "robust", f"robust_error_seed{int(args.seed)}.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        err_payload = {
            "passed": False,
            "fail_reason": [f"TOPLEVEL_EXCEPTION:{type(e).__name__}"],
            "exception": str(e),
            "traceback": traceback.format_exc(),
            "meta": _normalize_meta(meta),
        }
        _atomic_write_json(out_path, err_payload)
        raise


if __name__ == "__main__":
    # Manual tests (examples):
    #   python3 analysis/robust_optimizer.py --data datasets/BTCUSDT --window 2023-01_2023-03 \
    #     --base-config configs/pipeline_research_backtest.json --samples 2 --folds 2 --score-fallback equity
    #   PIPELINE_VERBOSE_DIAGNOSTICS=1 python3 analysis/robust_optimizer.py --data datasets/BTCUSDT --window 2023-01_2023-03 \
    #     --base-config configs/pipeline_research_backtest.json --samples 1 --folds 1 --score-fallback expectancy
    main()
