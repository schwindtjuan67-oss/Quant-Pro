#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

from backtest.run_backtest import _normalize_candles_any_ts, run_backtest_on_candles


CONFIG_PATH = "configs/pipeline_research_backtest.json"
DATA_ROOT = "datasets"
MAX_CANDLES = 600


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_candles_from_csv(path: str, max_rows: int) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV sin header: {path}")

        required = ("timestamp", "open", "high", "low", "close", "volume")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"CSV {path} no tiene columnas requeridas: {missing}")

        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            yield {
                "timestamp": int(float(row["timestamp"])),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }


def _load_candles(symbol: str, interval: str, max_rows: int) -> List[Dict[str, Any]]:
    pattern = os.path.join(DATA_ROOT, symbol, interval, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No se encontraron CSVs en {pattern}")
    raw = list(_iter_candles_from_csv(files[0], max_rows))
    return _normalize_candles_any_ts(raw)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, (int, str)) or value is None:
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _signature_for_trades(trades: List[Dict[str, Any]]) -> str:
    normalized: List[List[Tuple[str, Any]]] = []
    for trade in trades:
        items = []
        for key in sorted(trade.keys()):
            items.append((key, _normalize_value(trade[key])))
        normalized.append(items)
    payload = json.dumps(normalized, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_first_trade_info(trades: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not trades:
        return "n/a", "n/a"
    trade = trades[0]
    time_keys = ("entry_time", "entry_timestamp", "timestamp", "timestamp_ms", "entry_ts")
    price_keys = ("entry_price", "price", "entry", "open", "close")
    entry_time = next((str(trade[k]) for k in time_keys if k in trade), "n/a")
    entry_price = next((str(trade[k]) for k in price_keys if k in trade), "n/a")
    return entry_time, entry_price


def _run_variant(
    candles: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    label: str,
    strategy_params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    warmup = min(200, max(1, len(candles) // 3))
    trades = run_backtest_on_candles(
        candles,
        strategy_params,
        config=cfg,
        warmup_candles=warmup,
    )
    signature = _signature_for_trades(trades)
    entry_time, entry_price = _extract_first_trade_info(trades)
    print(
        f"{label}: trades={len(trades)} first_entry_time={entry_time} "
        f"first_entry_price={entry_price} signature={signature}"
    )
    return trades, signature


def main() -> int:
    cfg = _read_json(CONFIG_PATH)
    symbol = (cfg.get("symbol") or (cfg.get("symbols") or ["SOLUSDT"])[0]).upper()
    interval = cfg.get("interval", "1m")
    candles = _load_candles(symbol, interval, MAX_CANDLES)

    params_fast = {"ema_fast": 6, "ema_slow": 30}
    params_slow = {"ema_fast": 20, "ema_slow": 74}

    _, sig_fast = _run_variant(candles, cfg, "FAST", params_fast)
    _, sig_slow = _run_variant(candles, cfg, "SLOW", params_slow)

    if sig_fast == sig_slow:
        print("!!! WARNING: SIGNATURES IDENTICAL. PARAMS MAY NOT BE APPLYING !!!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
