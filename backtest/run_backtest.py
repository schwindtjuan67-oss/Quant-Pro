#!/usr/bin/env python3
# run_backtest.py
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List, Optional, Tuple

from analysis.grid_metrics import compute_metrics_from_trades, compute_score


# --- import robusto (funciona como script o como -m) ---
if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from backtest.backtest_runner import BacktestRunner
else:
    from .backtest_runner import BacktestRunner


# =====================================================
# Time helpers
# =====================================================
def _parse_date_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _date_range_to_ms(
    date_from: Optional[str], date_to: Optional[str]
) -> Tuple[Optional[int], Optional[int]]:
    start_ms = None
    end_ms = None

    if date_from:
        start_ms = int(_parse_date_ymd(date_from).timestamp() * 1000)

    if date_to:
        end_dt = _parse_date_ymd(date_to)
        end_ms = int(end_dt.timestamp() * 1000)

    return start_ms, end_ms


# =====================================================
# Candle parsing / validation
# =====================================================
REQUIRED_COLS = ("timestamp", "open", "high", "low", "close", "volume")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _safe_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _normalize_candle(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "timestamp": _safe_int(row["timestamp"]),
        "open": _safe_float(row["open"]),
        "high": _safe_float(row["high"]),
        "low": _safe_float(row["low"]),
        "close": _safe_float(row["close"]),
        "volume": _safe_float(row["volume"]),
    }


def _validate_candle(c: Dict[str, Any]) -> Optional[str]:
    ts = c["timestamp"]
    o, h, l, cl, v = c["open"], c["high"], c["low"], c["close"], c["volume"]

    if ts <= 0:
        return "timestamp<=0"
    if l > h:
        return "low>high"
    if not (l <= o <= h):
        return "open_outside_range"
    if not (l <= cl <= h):
        return "close_outside_range"
    if v < 0:
        return "volume<0"
    return None


def _coerce_ts_to_ms(v: Any) -> int:
    """
    Acepta timestamp en segundos o ms. Devuelve ms.
    """
    try:
        x = int(float(v))
    except Exception:
        return 0
    if x < 1_000_000_000_000:  # parece segundos
        x *= 1000
    return x


def _normalize_candles_any_ts(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Acepta candles con:
      - timestamp (seg o ms)
      - timestamp_ms (seg o ms)
    y devuelve candles con "timestamp" en ms (lo que el runner ya usa).
    """
    out: List[Dict[str, Any]] = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        if "timestamp" in c:
            ts = _coerce_ts_to_ms(c.get("timestamp"))
        elif "timestamp_ms" in c:
            ts = _coerce_ts_to_ms(c.get("timestamp_ms"))
        else:
            # best-effort: probar claves comunes
            ts = 0
            for k in ("ts", "open_time"):
                if k in c:
                    ts = _coerce_ts_to_ms(c.get(k))
                    break

        if ts <= 0:
            continue

        cc = dict(c)
        cc["timestamp"] = ts
        # normalizar nombres típicos si vienen con timestamp_ms
        if "timestamp_ms" in cc:
            cc.pop("timestamp_ms", None)

        # asegurar floats
        for k in ("open", "high", "low", "close", "volume"):
            if k in cc:
                cc[k] = _safe_float(cc.get(k))
        out.append(cc)

    # ordenar por timestamp para estabilidad
    out.sort(key=lambda x: int(x.get("timestamp", 0)))
    return out


# =====================================================
# CSV loader
# =====================================================
def _list_csv_files(data_path: str) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path]
    return sorted(glob.glob(os.path.join(data_path, "*.csv")))


def _iter_candles_from_csvs(
    files: List[str],
    *,
    start_ms: Optional[int],
    end_ms: Optional[int],
    strict: bool = False,
    max_bad_rows: int = 50,
    quiet: bool = False,
) -> Iterable[Dict[str, Any]]:
    bad = 0
    last_ts = None

    for fp in files:
        with open(fp, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise RuntimeError(f"CSV sin header: {fp}")

            missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
            if missing:
                raise RuntimeError(f"CSV {fp} no tiene columnas requeridas: {missing}")

            for row in reader:
                c = _normalize_candle(row)

                ts = int(c["timestamp"])
                if ts < 1_000_000_000_000:
                    ts *= 1000
                c["timestamp"] = ts

                if start_ms is not None and ts < start_ms:
                    continue
                if end_ms is not None and ts >= end_ms:
                    continue

                err = _validate_candle(c)
                if err:
                    bad += 1
                    if strict:
                        raise RuntimeError(f"Fila inválida ({err}) en {fp}")
                    if not quiet and bad <= max_bad_rows:
                        print(f"[WARN] bad row ({err}) {os.path.basename(fp)} ts={ts}")
                    continue

                if last_ts is not None and ts <= last_ts and not quiet:
                    print(f"[WARN] timestamps no crecientes last={last_ts} now={ts}")

                last_ts = ts
                yield c


# =====================================================
# IO helpers
# =====================================================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _copy_file(src: str, dst: str) -> None:
    _ensure_dir(os.path.dirname(dst) or ".")
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())


def _find_trades_csv(out_dir: str) -> Optional[str]:
    pats = [
        os.path.join(out_dir, "*trades*.csv"),
        os.path.join(out_dir, "*trade*.csv"),
        os.path.join(out_dir, "logs", "*trades*.csv"),
        os.path.join(out_dir, "logs", "*trade*.csv"),
        os.path.join(out_dir, "**", "*trades*.csv"),
        os.path.join(out_dir, "**", "*trade*.csv"),
    ]
    cands: List[str] = []
    for p in pats:
        cands.extend(glob.glob(p, recursive=True))

    # filtrar "bars"/"candles" falsos positivos
    cands = [c for c in cands if "bar" not in os.path.basename(c).lower()]
    if not cands:
        return None
    cands.sort(key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0, reverse=True)
    return cands[0]


def _load_trades_from_csv(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(dict(row))

    # coerción básica numérica
    for t in out:
        for k in ("pnl_r", "holding_time_sec", "timestamp_ms", "timestamp"):
            if k in t and t[k] not in (None, ""):
                try:
                    if k in ("timestamp_ms", "timestamp"):
                        t[k] = int(float(t[k]))
                    else:
                        t[k] = float(t[k])
                except Exception:
                    pass
    return out


def _extract_trades_list(result: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Intentamos obtener lista de trades desde el resultado del runner.
    Compat con distintos nombres.
    """
    for k in ("trades_list", "trades", "fills", "executions"):
        v = result.get(k)
        if isinstance(v, list) and (not v or isinstance(v[0], dict)):
            return v  # type: ignore[return-value]
    return None


def _apply_strategy_params(cfg: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mergea params sin asumir un único esquema de config:
      - cfg["strategy_params"]
      - cfg["strategy"]["params"]
    """
    cfg2 = json.loads(json.dumps(cfg))  # deep copy
    if isinstance(cfg2.get("strategy_params"), dict):
        params = cfg2["strategy_params"]
    elif isinstance(cfg2.get("strategy"), dict) and isinstance(cfg2["strategy"].get("params"), dict):
        params = cfg2["strategy"]["params"]
    else:
        cfg2["strategy_params"] = {}
        params = cfg2["strategy_params"]

    for k, v in (patch or {}).items():
        params[k] = v

    return cfg2


def _merge_strategy_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if not isinstance(cfg, dict):
        return merged
    strategy = cfg.get("strategy")
    if isinstance(strategy, dict) and isinstance(strategy.get("kwargs"), dict):
        merged.update(strategy["kwargs"])
    if isinstance(cfg.get("strategy_kwargs"), dict):
        merged.update(cfg["strategy_kwargs"])
    if isinstance(cfg.get("params"), dict):
        merged.update(cfg["params"])
    if isinstance(strategy, dict) and isinstance(strategy.get("params"), dict):
        merged.update(strategy["params"])
    if isinstance(cfg.get("strategy_params"), dict):
        merged.update(cfg["strategy_params"])
    return merged


def _is_verbose_pipeline() -> bool:
    return os.getenv("PIPELINE_VERBOSE_DIAGNOSTICS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _stable_params_key(params: Dict[str, Any]) -> str:
    try:
        return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(params)


_SANITY_TRADE_COUNTS: Dict[str, int] = {}


def _sanity_check_trade_counts(params_key: str, trades: int) -> None:
    if not params_key:
        return
    if params_key in _SANITY_TRADE_COUNTS:
        return
    for other_key, other_trades in _SANITY_TRADE_COUNTS.items():
        if other_trades == trades:
            if _is_verbose_pipeline():
                print(
                    "[BACKTEST][SANITY] params_key changed but trades count stayed the same: "
                    f"{trades} (prev_key={other_key} new_key={params_key})"
                )
            break
    _SANITY_TRADE_COUNTS[params_key] = trades


def _adapter_override_diagnostics(params: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    try:
        from Live.hybrid_scalper_pro import HybridScalperPRO
    except Exception:
        return [], sorted(params.keys())

    try:
        import inspect

        accepted = {
            k
            for k in inspect.signature(HybridScalperPRO.__init__).parameters
            if k != "self"
        }
    except Exception:
        accepted = set()

    mapping = {
        "ema_fast": ("EMA_FAST",),
        "ema_slow": ("EMA_SLOW",),
        "atr_len": ("ATR_LEN", "ATR_N"),
        "sl_atr_mult": ("ATR_STOP_MULT", "RANGE_STOP_ATR_MULT"),
        "tp_atr_mult": ("ATR_TRAIL_MULT", "RANGE_TP_TO_VWAP_ATR"),
        "cooldown_sec": ("cooldown_after_loss_sec", "cooldown_after_win_sec", "reentry_block_sec"),
    }
    applied: List[str] = []
    skipped: List[str] = []
    for key in (params or {}).keys():
        applied_any = False
        if key in accepted or hasattr(HybridScalperPRO, key):
            applied_any = True
        if hasattr(HybridScalperPRO, key.upper()):
            applied_any = True
        for mapped in mapping.get(key, ()):
            if mapped in accepted or hasattr(HybridScalperPRO, mapped):
                applied_any = True
                break
        if applied_any:
            applied.append(key)
        else:
            skipped.append(key)
    return sorted(set(applied)), sorted(set(skipped))


# =====================================================
# ✅ In-memory entrypoint (para robust_optimizer / optimizadores)
# =====================================================
def run_backtest_on_candles(
    candles: List[Dict[str, Any]],
    strategy_params: Dict[str, Any],
    *,
    base_config: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    symbol: Optional[str] = None,
    interval: str = "1m",
    warmup_candles: int = 500,
    strict: bool = False,
    quiet: bool = True,
    compute_metrics: bool = False,
    return_result: bool = False,
) -> Any:
    """
    Entrypoint in-memory para optimizadores (robust/grid/etc).

    - candles: lista de dicts con timestamp/timestamp_ms (seg o ms).
    - strategy_params: dict con params a aplicar al config base.
    - base_config/config/config_path: proveen el config base (uno de ellos).
        * config: dict ya cargado
        * base_config: dict o path str (compat)
        * config_path: path str (si tu robust lo pasa con otro nombre)
    - Devuelve:
        * por defecto: List[Dict] de trades (lo que espera robust_optimizer)
        * si return_result=True: dict result del runner (con trades si están)
    """
    # Resolver config base
    cfg: Optional[Dict[str, Any]] = None
    if isinstance(config, dict):
        cfg = config
    elif isinstance(base_config, dict):
        cfg = base_config
    elif isinstance(base_config, str) and base_config.strip():
        cfg = _read_json(base_config)
    elif isinstance(config_path, str) and config_path.strip():
        cfg = _read_json(config_path)

    if cfg is None:
        raise RuntimeError(
            "[BACKTEST][INMEM] Missing base config. Provide config dict or base_config/config_path (json path)."
        )

    if _is_verbose_pipeline():
        print(
            "[BACKTEST][DIAG] received_strategy_params=",
            json.dumps(strategy_params or {}, ensure_ascii=False, sort_keys=True),
        )

    # Symbol
    sym = (symbol or cfg.get("symbol") or (cfg.get("symbols") or ["SOLUSDT"])[0]).upper()

    # Aplicar patch de params
    cfg2 = _apply_strategy_params(cfg, strategy_params)
    merged_kwargs = _merge_strategy_kwargs(cfg2)
    cfg2["strategy_kwargs"] = merged_kwargs
    cfg2["params"] = dict(merged_kwargs)

    strategy_cfg = cfg2.get("strategy")
    if not isinstance(strategy_cfg, dict):
        strategy_cfg = {}
    else:
        strategy_cfg = dict(strategy_cfg)
    name = str(strategy_cfg.get("name") or "").strip().lower()
    if name in ("hybrid_scalper_pro", "hybridscalperpro", "hybrid_scalper"):
        strategy_cfg.pop("name", None)
    strategy_cfg["kwargs"] = dict(merged_kwargs)
    cfg2["strategy"] = strategy_cfg

    if _is_verbose_pipeline():
        applied, skipped = _adapter_override_diagnostics(merged_kwargs)
        print("[BACKTEST][DIAG] merged_strategy_kwargs=", json.dumps(merged_kwargs, ensure_ascii=False, sort_keys=True))
        print("[BACKTEST][DIAG] adapter_overrides_applied=", json.dumps(applied, ensure_ascii=False))
        print("[BACKTEST][DIAG] adapter_overrides_skipped=", json.dumps(skipped, ensure_ascii=False))
        print("[BACKTEST][DIAG] params_key=", _stable_params_key(merged_kwargs))

    # Normalizar candles (timestamp -> ms)
    candles2 = _normalize_candles_any_ts(candles)

    runner = BacktestRunner(
        config=cfg2,
        candles=candles2,
        symbol=sym,
        interval=interval,
        warmup_candles=warmup_candles,
    )

    # ===============================
    # 🔑 PIPELINE FIX: inyectar params al logger
    # ===============================
    try:
        logger = getattr(runner, "logger", None)
        if logger and hasattr(logger, "set_pending_meta"):
            logger.set_pending_meta({
                "params": strategy_params,
            })
    except Exception:
        pass


    result = runner.run() or {}
    trades_list = _extract_trades_list(result)

    # Si el runner devolvió lista de trades, usamos eso.
    if trades_list is None:
        # Algunos runners devuelven "trades" como int y escriben CSV afuera;
        # en modo in-memory no asumimos filesystem => devolvemos [] para no romper.
        trades_list = []

    _sanity_check_trade_counts(_stable_params_key(merged_kwargs), len(trades_list))

    if compute_metrics:
        try:
            metrics = compute_metrics_from_trades(trades_list)
            score = compute_score(metrics)
            result["metrics"] = metrics
            result["score"] = score
        except Exception:
            # no hacemos hard-fail en métricas
            result["metrics"] = {}
            result["score"] = -1e9

    if return_result:
        return result

    return trades_list


# Aliases (compat con lo que tu robust estaba buscando)
backtest_on_candles = run_backtest_on_candles
run_on_candles = run_backtest_on_candles
run_backtest_candles = run_backtest_on_candles


# =====================================================
# Main (CLI)
# =====================================================
def main():
    ap = argparse.ArgumentParser("Quant Shadow Backtest Runner (1m CSVs)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max_candles", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--warmup", type=int, default=500, help="Velas de warmup")

    args = ap.parse_args()

    cfg = _read_json(args.config)
    symbol = (
        args.symbol
        or cfg.get("symbol")
        or (cfg.get("symbols") or ["SOLUSDT"])[0]
    ).upper()

    files = _list_csv_files(args.data)
    if not files:
        raise RuntimeError("No se encontraron CSVs")

    start_ms, end_ms = _date_range_to_ms(args.date_from, args.date_to)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or f"results/backtest_{symbol}_{ts_tag}"
    _ensure_dir(out_dir)

    _copy_file(args.config, os.path.join(out_dir, "config_used.json"))

    if not args.quiet:
        print(f"[BACKTEST] symbol={symbol} files={len(files)} warmup={args.warmup}")

    candles: List[Dict[str, Any]] = []
    for c in _iter_candles_from_csvs(
        files,
        start_ms=start_ms,
        end_ms=end_ms,
        strict=args.strict,
        quiet=args.quiet,
    ):
        candles.append(c)
        if args.max_candles and len(candles) >= args.max_candles:
            break

    runner = BacktestRunner(
        config=cfg,
        candles=candles,
        symbol=symbol,
        interval=args.interval,
        warmup_candles=args.warmup,
    )

    try:
        logger = getattr(runner, "logger", None)
        if logger and hasattr(logger, "set_pending_meta"):
            strategy_params = {}

            if isinstance(cfg.get("strategy_params"), dict):
                strategy_params = cfg["strategy_params"]
            elif isinstance(cfg.get("strategy"), dict) and isinstance(cfg["strategy"].get("params"), dict):
                strategy_params = cfg["strategy"]["params"]

        logger.set_pending_meta({
            "params": strategy_params,
        })
    except Exception:
        pass



    result = runner.run() or {}

    # -----------------------------
    # ✅ métricas DESPUÉS de correr
    # -----------------------------
    trades_list = _extract_trades_list(result)

    metrics: Dict[str, float] = {}
    score: float = -1e9

    if trades_list is not None:
        metrics = compute_metrics_from_trades(trades_list)
        score = compute_score(metrics)
        _write_json(os.path.join(out_dir, "metrics.json"), metrics)
        _write_json(os.path.join(out_dir, "score.json"), {"score": score})
    else:
        # fallback: buscar trades CSV en out_dir (si tu runner lo escribió)
        trades_csv = _find_trades_csv(out_dir)
        if trades_csv:
            try:
                trades_list2 = _load_trades_from_csv(trades_csv)
                metrics = compute_metrics_from_trades(trades_list2)
                score = compute_score(metrics)
                _write_json(os.path.join(out_dir, "metrics.json"), metrics)
                _write_json(os.path.join(out_dir, "score.json"), {"score": score})
                if not args.quiet:
                    print(f"[BACKTEST] metrics computed from trades_csv={trades_csv}")
            except Exception as e:
                if not args.quiet:
                    print(f"[BACKTEST] WARN: failed to compute metrics from CSV: {e}")

    summary = {
        "symbol": symbol,
        "interval": args.interval,
        "candles": len(candles),
        "warmup": args.warmup,
        "equity_r": result.get("equity_r"),
        "trades": result.get("trades") if not isinstance(result.get("trades"), list) else len(result.get("trades")),
        "score": score,
    }

    _write_json(os.path.join(out_dir, "summary.json"), summary)

    # prints esperados por tu grid_runner (parsea trades/equity + output_dir)
    equity_r_val = summary["equity_r"]
    try:
        equity_r_val = float(equity_r_val) if equity_r_val is not None else 0.0
    except Exception:
        equity_r_val = 0.0

    trades_n = summary["trades"]
    try:
        trades_n = int(trades_n) if trades_n is not None else 0
    except Exception:
        trades_n = 0

    print(f"[BACKTEST] ✅ trades={trades_n} equity_r={equity_r_val:.4f}")
    print(f"[BACKTEST] output_dir={out_dir}")

    # opcional: mostrar métricas si existen
    if metrics and not args.quiet:
        keys = list(metrics.keys())[:10]
        print("[BACKTEST] metrics (top10):")
        for k in keys:
            print(f"  {k}: {metrics.get(k)}")
        print(f"[BACKTEST] SCORE = {score}")


if __name__ == "__main__":
    main()

