#!/usr/bin/env python3
# Live/logger_pro.py
from __future__ import annotations

import csv
import os
import json
from datetime import datetime, timezone, timedelta
import uuid
import atexit
from typing import Any, List

# ============================================================
# RUN MODE (PIPELINE | SHADOW | LIVE)
# ============================================================
RUN_MODE = os.getenv("RUN_MODE", "LIVE").upper().strip()

# =======================
# PIPELINE Parquet config
# =======================
PIPELINE_PARQUET_DIR = os.getenv(
    "PIPELINE_PARQUET_DIR",
    "results/pipeline_trades_parquet",
)
PIPELINE_PARQUET_FLUSH = int(os.getenv("PIPELINE_PARQUET_FLUSH", "500"))
PIPELINE_WRITE_CSV = os.getenv("PIPELINE_WRITE_CSV", "0").lower() in ("1", "true", "yes")
PIPELINE_DISABLE_TRADE_LOG = os.getenv("PIPELINE_DISABLE_TRADE_LOG", "0").lower() in ("1", "true", "yes")

# ---- TZ: Argentina
def _get_local_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/Argentina/Buenos_Aires")
    except Exception:
        return timezone(timedelta(hours=-3))

LOCAL_TZ = _get_local_tz()

def _ts_ms_now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def _dt_from_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _safe_json_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        # fallback conservador (no romper logger)
        try:
            return str(x)
        except Exception:
            return ""

def _merge_meta_json(meta_json_str: str, ctx: dict) -> str:
    """
    Garantiza:
      - params: dict
      - context: dict
    Migra "__ctx" -> "context"
    Mergea ctx final (ctx gana).
    """
    ctx = ctx or {}

    def _ensure(obj: dict) -> dict:
        if not isinstance(obj.get("params"), dict):
            obj["params"] = {}

        legacy_ctx = obj.pop("__ctx", None)
        if legacy_ctx and isinstance(legacy_ctx, dict):
            obj.setdefault("context", {})
            if isinstance(obj["context"], dict):
                obj["context"].update(legacy_ctx)

        if not isinstance(obj.get("context"), dict):
            obj["context"] = {}

        obj["context"].update(ctx)
        return obj

    if not meta_json_str:
        return _safe_json_str(_ensure({}))

    try:
        obj = json.loads(meta_json_str)
        if isinstance(obj, dict):
            return _safe_json_str(_ensure(obj))
        return _safe_json_str(_ensure({"meta_obj": obj}))
    except Exception:
        return _safe_json_str(_ensure({"meta_raw": meta_json_str}))

def _extract_symbol(meta_json_str: str):
    try:
        m = json.loads(meta_json_str) if isinstance(meta_json_str, str) and meta_json_str else {}
        ctx = m.get("context", {})
        if isinstance(ctx, dict):
            return ctx.get("symbol")
        return None
    except Exception:
        return None

class TradeLogger:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.run_mode = RUN_MODE
        self._pipeline_mode = (self.run_mode == "PIPELINE")
        self._trade_log_disabled = self._pipeline_mode and PIPELINE_DISABLE_TRADE_LOG
        self._pending_meta_json: str = ""
        self._parquet_buffer: List[dict] = []

        self._trades_header = [
            "trade_id","type","timestamp_ms","ts_utc_iso","ts_local_iso",
            "date_local","time_local","weekday_local","hour_local","minute_local",
            "side","qty","price","entry_price","exit_price","pnl","pnl_pct",
            "pnl_net_est","fee_est","fee_rate","reason","regime","holding_time_sec",
            "atr_entry","sl_initial","risk_usdt","risk_pct_equity","mfe","mae",
            "equity_before","equity_after","pnl_r","meta_json"
        ]

        self.active_trade = None

        if self._trade_log_disabled:
            self.file_pipeline_trades = ""
            self.file_trades_v4 = ""
            return

        if self._pipeline_mode:
            os.makedirs("results", exist_ok=True)
            self.file_pipeline_trades = os.getenv(
                "PIPELINE_TRADES_PATH",
                "results/pipeline_trades.csv",
            )
            if PIPELINE_WRITE_CSV and not os.path.exists(self.file_pipeline_trades):
                with open(self.file_pipeline_trades, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self._trades_header)

            # flush garantizado al terminar proceso
            atexit.register(self.close)

        else:
            os.makedirs("logs", exist_ok=True)
            self.file_trades_v4 = f"logs/{symbol}_shadow_trades_v4.csv"
            if not os.path.exists(self.file_trades_v4):
                with open(self.file_trades_v4, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self._trades_header)

    # -------------------------
    # Public: force flush
    # -------------------------
    def close(self) -> None:
        """Flush final (seguro) para PIPELINE."""
        if self._trade_log_disabled:
            return
        if self._pipeline_mode:
            try:
                self._flush_parquet()
            except Exception:
                # no reventar el shutdown
                pass

    # -------------------------
    # Parquet flush (PARTITIONED)
    # -------------------------
    def _flush_parquet(self):
        if self._trade_log_disabled:
            return
        if not self._parquet_buffer:
            return

        import pandas as pd
        from infra.parquet_writer import write_parquet_partitioned

        df = pd.DataFrame(self._parquet_buffer)
        self._parquet_buffer.clear()

        # symbol para partición: intenta meta_json, fallback self.symbol
        if "meta_json" in df.columns:
            df["symbol"] = df["meta_json"].apply(_extract_symbol)
        else:
            df["symbol"] = None

        df["symbol"] = df["symbol"].fillna(self.symbol).astype(str)

        # regime para partición: preferimos columna explícita, fallback meta_json.context.regime, fallback "UNKNOWN"
        if "regime" not in df.columns:
            df["regime"] = "UNKNOWN"
        df["regime"] = df["regime"].fillna("UNKNOWN").astype(str)

        # particionado físico (symbol, regime)
        write_parquet_partitioned(
            df,
            out_dir=PIPELINE_PARQUET_DIR,
            partition_cols=("symbol", "regime"),
        )

    # -------------------------
    # Adapter API
    # -------------------------
    def set_pending_meta(self, meta_json: Any) -> None:
        if self._trade_log_disabled:
            return
        self._pending_meta_json = _safe_json_str(meta_json)

    def _time_fields(self, ts_ms: int):
        dt_utc = _dt_from_ms(ts_ms)
        dt_local = dt_utc.astimezone(LOCAL_TZ)
        return {
            "ts_utc_iso": _iso(dt_utc),
            "ts_local_iso": _iso(dt_local),
            "date_local": dt_local.date().isoformat(),
            "time_local": dt_local.strftime("%H:%M:%S"),
            "weekday_local": dt_local.strftime("%a"),
            "hour_local": dt_local.hour,
            "minute_local": dt_local.minute,
        }

    def _write_row(self, row_v4: dict):
        if self._trade_log_disabled:
            return
        if self._pipeline_mode:
            self._parquet_buffer.append(row_v4)
            if len(self._parquet_buffer) >= PIPELINE_PARQUET_FLUSH:
                self._flush_parquet()

            if PIPELINE_WRITE_CSV:
                with open(self.file_pipeline_trades, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([row_v4.get(c, "") for c in self._trades_header])
        else:
            with open(self.file_trades_v4, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([row_v4.get(c, "") for c in self._trades_header])

    # -------------------------
    # Bars (solo MFE/MAE)
    # -------------------------
    def log_bar(self, **k):
        if self._trade_log_disabled:
            return
        candle = k.get("candle", k)
        ts_ms = _safe_int(candle.get("timestamp"), default=_ts_ms_now())
        close = _safe_float(candle.get("close"))
        if self.active_trade and close is not None:
            side = self.active_trade["side"]
            entry = self.active_trade["entry_price"]
            if side == "LONG":
                self.active_trade["mfe"] = max(self.active_trade["mfe"], close - entry)
                self.active_trade["mae"] = min(self.active_trade["mae"], close - entry)
            else:
                self.active_trade["mfe"] = max(self.active_trade["mfe"], entry - close)
                self.active_trade["mae"] = min(self.active_trade["mae"], entry - close)

    # -------------------------
    # Trades
    # -------------------------
    def log_trade(self, **k):
        if self._trade_log_disabled:
            return
        trade_type = k.get("type") or k.get("type_")
        if not trade_type:
            return

        ts_ms = _safe_int(k.get("timestamp"), default=_ts_ms_now())
        tinfo = self._time_fields(ts_ms)

        side = k.get("side")
        qty = _safe_float(k.get("qty"), 0.0) or 0.0
        price = _safe_float(k.get("price"), 0.0) or 0.0

        fee_rate = _safe_float(k.get("fee_rate"))
        fee_est = _safe_float(k.get("fee_est"))
        pnl_abs = _safe_float(k.get("pnl"))
        pnl_pct = _safe_float(k.get("pnl_pct"))

        equity_before = _safe_float(k.get("equity_before"))
        equity_after = _safe_float(k.get("equity_after"))

        atr_entry = _safe_float(k.get("atr_entry"))
        sl_initial = _safe_float(k.get("sl_initial"))
        risk_usdt = _safe_float(k.get("risk_usdt"))
        risk_pct_equity = _safe_float(k.get("risk_pct_equity"))

        reason = k.get("reason")
        regime = k.get("regime")

        meta_json_in = _safe_json_str(k.get("meta_json")) or (self._pending_meta_json or "")

        if self._pipeline_mode:
            ctx = {
                "run_mode": "PIPELINE",
                "run_id": os.getenv("PIPELINE_RUN_ID", ""),
                "window": os.getenv("PIPELINE_WINDOW", ""),
                "seed": os.getenv("PIPELINE_SEED", ""),
                "regime": os.getenv("PIPELINE_REGIME", ""),
                "symbol": self.symbol,
            }
            meta_json_in = _merge_meta_json(meta_json_in, ctx)

        if trade_type == "ENTRY":
            trade_id = str(uuid.uuid4())
            self.active_trade = {
                "trade_id": trade_id,
                "timestamp_ms": ts_ms,
                "side": side,
                "qty": qty,
                "entry_price": price,
                "regime": regime,
                "mfe": 0.0,
                "mae": 0.0,
                "risk_usdt": risk_usdt,
                "meta_json": meta_json_in,
            }
            self._pending_meta_json = ""

            row_v4 = {
                "trade_id": trade_id,
                "type": "ENTRY",
                "timestamp_ms": ts_ms,
                **tinfo,
                "side": side,
                "qty": qty,
                "price": price,
                "entry_price": price,
                "exit_price": "",
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "pnl_net_est": "",
                "fee_est": fee_est or "",
                "fee_rate": fee_rate or "",
                "reason": reason,
                "regime": regime,
                "holding_time_sec": "",
                "atr_entry": atr_entry or "",
                "sl_initial": sl_initial or "",
                "risk_usdt": risk_usdt or "",
                "risk_pct_equity": risk_pct_equity or "",
                "mfe": "",
                "mae": "",
                "equity_before": equity_before or "",
                "equity_after": equity_after or "",
                "pnl_r": "",
                "meta_json": meta_json_in,
            }
            self._write_row(row_v4)

        elif trade_type == "EXIT" and self.active_trade:
            entry = self.active_trade
            exit_price = price

            if pnl_abs is None:
                direction = 1.0 if entry["side"] == "LONG" else -1.0
                pnl_abs = (exit_price - entry["entry_price"]) * entry["qty"] * direction

            holding_time = int((ts_ms - entry["timestamp_ms"]) / 1000)

            if fee_est is None and fee_rate is not None:
                fee_est = (abs(entry["entry_price"] * entry["qty"]) + abs(exit_price * entry["qty"])) * fee_rate

            pnl_net_est = pnl_abs - fee_est if pnl_abs is not None and fee_est is not None else ""

            pnl_r = ""
            try:
                if pnl_abs is not None and entry.get("risk_usdt"):
                    pnl_r = float(pnl_abs) / float(entry["risk_usdt"])
            except Exception:
                pnl_r = ""

            row_v4 = {
                "trade_id": entry["trade_id"],
                "type": "EXIT",
                "timestamp_ms": ts_ms,
                **tinfo,
                "side": entry["side"],
                "qty": entry["qty"],
                "price": exit_price,
                "entry_price": entry["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl_abs or "",
                "pnl_pct": pnl_pct or "",
                "pnl_net_est": pnl_net_est,
                "fee_est": fee_est or "",
                "fee_rate": fee_rate or "",
                "reason": reason,
                "regime": entry.get("regime"),
                "holding_time_sec": holding_time,
                "atr_entry": atr_entry or "",
                "sl_initial": sl_initial or "",
                "risk_usdt": risk_usdt or "",
                "risk_pct_equity": risk_pct_equity or "",
                "mfe": entry.get("mfe", ""),
                "mae": entry.get("mae", ""),
                "equity_before": equity_before or "",
                "equity_after": equity_after or "",
                "pnl_r": pnl_r,
                "meta_json": entry.get("meta_json") or "",
            }
            self._write_row(row_v4)
            self.active_trade = None

    def log(self, **k):
        if self._trade_log_disabled:
            return
        if "open" in k or "candle" in k:
            self.log_bar(**k)
        else:
            self.log_trade(**k)


