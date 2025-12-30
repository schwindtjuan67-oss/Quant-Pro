#!/usr/bin/env python3
# robust_optimizer.py
from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import pickle
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Reutilizamos helpers existentes (pero NO dependemos exclusivamente del iterador)
from backtest.run_backtest import _date_range_to_ms, _iter_candles_from_csvs, _list_csv_files

from analysis.grid_metrics import compute_metrics_from_trades, compute_score
from analysis.opt_space import default_param_space, sample_params

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
    fail_reason: str = ""


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

def _passes_gates(metrics: Dict[str, float], gates: Dict[str, Any]) -> Tuple[bool, str]:
    t = metrics.get("trades", 0)
    if t < gates.get("min_trades", 30):
        return False, f"min_trades ({t})"

    max_dd = abs(metrics.get("max_drawdown_r", 0.0))
    if max_dd > gates.get("max_dd_r", 20.0):
        return False, f"max_dd_r ({max_dd:.2f})"

    pf = metrics.get("profit_factor", 0.0)
    if pf < gates.get("min_pf", 1.05):
        return False, f"min_pf ({pf:.2f})"

    winrate = metrics.get("winrate", 0.0)
    if winrate < gates.get("min_winrate", 0.35):
        return False, f"min_winrate ({winrate:.2f})"

    return True, ""


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


def evaluate_params_walk_forward(
    data: Any,
    params: Dict[str, Any],
    backtest_fn: BacktestFn,
    splits: List[Tuple[slice, slice]],
    gates: Dict[str, Any],
) -> EvalResult:
    fold_results: List[FoldResult] = []
    passed = True
    fail_reason = ""

    for i, (_tr_slice, te_slice) in enumerate(splits):
        test_data = data[te_slice]
        trades_test = backtest_fn(test_data, params)

        m = compute_metrics_from_trades(trades_test)
        ok, reason = _passes_gates(m, gates)
        s = compute_score(m)

        fold_results.append(FoldResult(fold_id=i, metrics=m, score=s))
        if not ok:
            passed = False
            fail_reason = f"fold {i}: {reason}"

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
    for call in (
        lambda: fn(candles=data_slice, strategy_params=params, base_config=base_cfg, symbol=symbol, interval=interval, warmup_candles=warmup),
        lambda: fn(candles=data_slice, params=params, base_config=base_cfg, symbol=symbol, interval=interval, warmup_candles=warmup),
        lambda: fn(data=data_slice, strategy_params=params, base_config=base_cfg, symbol=symbol, interval=interval, warmup_candles=warmup),
        lambda: fn(data=data_slice, params=params, base_config=base_cfg, symbol=symbol, interval=interval, warmup_candles=warmup),
        lambda: fn(data_slice, params, base_config=base_cfg, symbol=symbol, interval=interval, warmup_candles=warmup),
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

# ✅ NUEVO: slice cache por worker (lista de test_slices ya materializados)
_WORKER_TEST_SLICES: Optional[List[Any]] = None  # Any para permitir list/slice/np/etc


def _worker_init(
    data: List[Dict[str, Any]],
    base_cfg: Optional[Dict[str, Any]],
    symbol: Optional[str],
    interval: str,
    warmup: int,
    gates: Dict[str, Any],
    splits: List[Tuple[slice, slice]],
    use_dummy: bool,
):
    # Se ejecuta 1 vez por proceso
    global _WORKER_DATA, _WORKER_BASE_CFG, _WORKER_SYMBOL, _WORKER_INTERVAL, _WORKER_WARMUP, _WORKER_GATES, _WORKER_SPLITS, _WORKER_USE_DUMMY, _WORKER_TEST_SLICES
    _WORKER_DATA = data
    _WORKER_BASE_CFG = base_cfg
    _WORKER_SYMBOL = symbol
    _WORKER_INTERVAL = interval
    _WORKER_WARMUP = int(warmup)
    _WORKER_GATES = gates or {}
    _WORKER_SPLITS = splits or []
    _WORKER_USE_DUMMY = bool(use_dummy)

    # ✅ Slice cache (reduce overhead de slicing repetido por param)
    try:
        _WORKER_TEST_SLICES = []
        for (_tr_slice, te_slice) in _WORKER_SPLITS:
            _WORKER_TEST_SLICES.append(_WORKER_DATA[te_slice])  # materializa 1 vez
    except Exception:
        _WORKER_TEST_SLICES = None


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

def _worker_eval_batch(params_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in params_batch:
        try:
            out.append(_worker_eval_one(p))
        except MemoryError:
            out.append({
                "params": p,
                "passed": False,
                "fail_reason": "OOM",
                "agg": {},
                "robust_score": -1e9,
                "folds": [],
            })
        except Exception as e:
            out.append({
                "params": p,
                "passed": False,
                "fail_reason": f"EXCEPTION:{type(e).__name__}",
                "agg": {},
                "robust_score": -1e9,
                "folds": [],
            })
    return out



def _evaluate_params_on_cached_test_slices(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    global _WORKER_TEST_SLICES, _WORKER_GATES

    if _WORKER_TEST_SLICES is None:
        # fallback a la vieja vía (no debería si init pudo cachear)
        return _worker_eval_one(params)

    fold_results: List[FoldResult] = []
    passed = True
    fail_reason = ""

    for i, test_data in enumerate(_WORKER_TEST_SLICES):
        trades_test = _worker_backtest_fn(test_data, params)

        m = compute_metrics_from_trades(trades_test)
        ok, reason = _passes_gates(m, _WORKER_GATES)
        s = compute_score(m)

        fold_results.append(FoldResult(fold_id=i, metrics=m, score=s))
        if not ok:
            passed = False
            fail_reason = f"fold {i}: {reason}"

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


def _worker_eval_one(params: Dict[str, Any]) -> Dict[str, Any]:
    global _WORKER_DATA
    if _WORKER_DATA is None:
        raise RuntimeError("[ROBUST][WORKER] Missing worker data (init not called).")

    try:
        er = evaluate_params_walk_forward(
            data=_WORKER_DATA,
            params=params,
            backtest_fn=_worker_backtest_fn,
            splits=_WORKER_SPLITS,
            gates=_WORKER_GATES,
        )
    except MemoryError:
        # 🔴 OOM explícito
        return {
            "params": params,
            "passed": False,
            "fail_reason": "OOM",
            "agg": {},
            "robust_score": -1e9,
            "folds": [],
        }
    except Exception as e:
        # 🔴 cualquier otro crash del worker
        return {
            "params": params,
            "passed": False,
            "fail_reason": f"EXCEPTION: {type(e).__name__}",
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
) -> List[EvalResult]:
    """
    Mantiene compat total.
    Si ROBUST_WORKERS>1 y no es un callable raro -> usa multiprocessing con ProcessPoolExecutor.
    """
    gates = gates or {
        "min_trades": 30,
        "max_dd_r": 20.0,
        "min_pf": 1.05,
        "min_winrate": 0.35,
        "lam_std": 0.75,
        "beta_worst": 0.35,
    }

    rng = np.random.default_rng(seed)
    space = default_param_space()
    splits = make_walk_forward_splits(len(data), n_folds=n_folds, min_train=min_train, min_test=min_test)

    # ---------- generar params determinísticamente ----------
    seen = set()
    params_list: List[Dict[str, Any]] = []
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
        for params in params_list:
            er = evaluate_params_walk_forward(
                data=data,
                params=params,
                backtest_fn=backtest_fn,
                splits=splits,
                gates=gates,
            )
            results.append(er)

        results.sort(
            key=lambda r: (r.robust_score, r.agg.get("score_worst", -1e9), -r.agg.get("score_std", 1e9)),
            reverse=True,
        )
        return results[:top_k]

    # ---------- paralelo: usamos workers global-init (NO pierde features) ----------
    # Importante: el backtest_fn pasado podría ser closure no-pickleable.
    # Para no romper, el modo paralelo NO usa ese callable directamente:
    # se apoya en _real_backtest_fn con base_cfg/symbol/interval/warmup en init del worker.
    # Si querés forzar uso del callable externo: desactivá paralelo.
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

    # Context spawn = Windows safe; en Linux también funciona.
    ctx = mp.get_context("spawn")

    progress_every = int(os.getenv("ROBUST_PROGRESS_EVERY", "25") or "25")

    # NOTA: acá NO sabemos base_cfg/symbol/interval/warmup: lo setea el main usando la función paralela dedicada.
    # Aun así, seguimos: el worker necesitará base_cfg seteado por main.
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
        ),
    ) as ex:
        # ✅ NUEVO: submit por batches
        if batch_size == 1:
            futs = [ex.submit(_evaluate_params_on_cached_test_slices, p) for p in params_list]
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    if param_timeout > 0:
                        res = fut.result(timeout=param_timeout)
                    else:
                        res = fut.result()
                except TimeoutError:
                    res = { 
                        "params": {},
                        "passed": False,
                        "fail_reason": "TIMEOUT",
                        "agg": {},
                        "robust_score": -1e9,
                        "folds": [],
                    }
                except Exception as e:
                    res = {
                        "params": {},
                        "passed": False,
                        "fail_reason": f"EXCEPTION: {type(e).__name__}",
                        "agg": {},
                        "robust_score": -1e9,
                        "folds": [],
                    }

                results_payloads.append(res)

                if progress_every > 0 and (done % progress_every == 0):
                    print(f"[ROBUST][PAR] done {done}/{len(params_list)}")

            batches: List[List[Dict[str, Any]]] = []


            for i in range(0, len(params_list), batch_size):
                batches.append(params_list[i:i + batch_size])

            futs = [ex.submit(_worker_eval_batch, b) for b in batches]
            done_params = 0
            total_params = len(params_list)
            for fut in as_completed(futs):
                batch_payloads = fut.result()
                # batch_payloads es List[Dict]
                for pl in batch_payloads:
                    results_payloads.append(pl)
                done_params += len(batch_payloads)
                if progress_every > 0 and (done_params % progress_every == 0):
                    print(f"[ROBUST][PAR] done {done_params}/{total_params} (batch_size={batch_size})")

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
            fail_reason=str(pl.get("fail_reason", "") or ""),
        ))

    results.sort(
        key=lambda r: (r.robust_score, r.agg.get("score_worst", -1e9), -r.agg.get("score_std", 1e9)),
        reverse=True,
    )
    return results[:top_k]


def save_results_json(path: str, results: List[EvalResult]) -> None:
    payload = []
    for r in results:
        payload.append({
            "params": r.params,
            "passed": r.passed,
            "fail_reason": r.fail_reason,
            "agg": r.agg,
            "folds": [
                {"fold_id": fr.fold_id, "score": fr.score, "metrics": fr.metrics}
                for fr in r.fold_results
            ],
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
) -> List[EvalResult]:
    gates = gates or {
        "min_trades": 30,
        "max_dd_r": 20.0,
        "min_pf": 1.05,
        "min_winrate": 0.35,
        "lam_std": 0.75,
        "beta_worst": 0.35,
    }

    rng = np.random.default_rng(seed)
    space = default_param_space()
    splits = make_walk_forward_splits(len(data), n_folds=n_folds, min_train=min_train, min_test=min_test)

    # generar params determinístico
    seen = set()
    params_list: List[Dict[str, Any]] = []
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
    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(data, base_cfg, symbol, interval, int(warmup), gates, splits, bool(use_dummy)),
    ) as ex:
        if batch_size == 1:
            futs = [ex.submit(_evaluate_params_on_cached_test_slices, p) for p in params_list]
            done = 0
            for fut in as_completed(futs):
                done += 1
                results_payloads.append(fut.result())
                if progress_every > 0 and (done % progress_every == 0):
                    print(f"[ROBUST][PAR] done {done}/{len(futs)}")
        else:
            batches: List[List[Dict[str, Any]]] = []
            for i in range(0, len(params_list), batch_size):
                batches.append(params_list[i:i + batch_size])

            futs = [ex.submit(_worker_eval_batch, b) for b in batches]
            done_params = 0
            total_params = len(params_list)

            for fut in as_completed(futs):
                batch_payloads = fut.result()
                for pl in batch_payloads:
                    results_payloads.append(pl)
                done_params += len(batch_payloads)
                if progress_every > 0 and (done_params % progress_every == 0):
                    print(f"[ROBUST][PAR] done {done_params}/{total_params} (batch_size={batch_size})")

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
            fail_reason=str(pl.get("fail_reason", "") or ""),
        ))

    results.sort(
        key=lambda r: (r.robust_score, r.agg.get("score_worst", -1e9), -r.agg.get("score_std", 1e9)),
        reverse=True,
    )
    return results[:top_k]


# ============================================================
# CLI runner
# ============================================================

def main():
    ap = argparse.ArgumentParser("robust_optimizer")
    ap.add_argument("--data", required=True, help="dataset path (folder with CSVs)")
    ap.add_argument("--out", required=True, help="output json")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)

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

    # ✅ paralelo (opcionales)
    ap.add_argument("--workers", type=int, default=0, help="workers para paralelo (0=auto/env/disabled)")
    ap.add_argument("--parallel", action="store_true", help="habilita paralelo (si workers>1 o env ROBUST_WORKERS)")

    # ✅ NUEVO: batch-size explícito (sin obligarte a usar env)
    ap.add_argument("--batch-size", type=int, default=0, help="batch params por task (0=auto/env default 8)")

    args = ap.parse_args()

    date_from = args.from_date
    date_to = args.to_date
    if args.window:
        date_from, date_to = window_to_dates(args.window)

    if not date_from or not date_to:
        raise SystemExit("[ROBUST] Missing date range. Provide --window OR (--from-date AND --to-date).")

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

    # ejecutar
    if (not args.use_dummy) and workers > 1:
        assert base_cfg is not None
        print(f"[ROBUST] Parallel enabled: workers={workers} (spawn-safe) batch_size={batch_size}")
        results = run_robust_search_parallel(
            data=data,
            base_cfg=base_cfg,
            symbol=args.symbol,
            interval=args.interval,
            warmup=int(args.warmup),
            n_samples=int(args.samples),
            seed=int(args.seed),
            n_folds=5,
            min_train=10_000,
            min_test=2_000,
            gates=None,
            top_k=20,
            workers=workers,
            use_dummy=bool(args.use_dummy),
            batch_size=int(batch_size),
        )
    else:
        if workers > 1 and args.use_dummy:
            print("[ROBUST][WARN] Parallel + dummy: usando modo secuencial (dummy es demasiado rápido y el overhead domina).")
        results = run_robust_search(
            data=data,
            backtest_fn=backtest_fn,
            n_samples=args.samples,
            seed=args.seed,
        )

    print(f"[ROBUST] finished, top={len(results)}")
    save_results_json(args.out, results)
    print(f"[ROBUST] saved -> {args.out}")


if __name__ == "__main__":
    main()








