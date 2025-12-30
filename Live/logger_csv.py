# logger_csv.py
import csv
import os
from datetime import datetime
import uuid


class TradeLogger:
    """
    Logger cuantitativo para SHADOW / LIVE (legacy-compatible).

    ✅ FIX (img3):
    - Loggea pnl_r en EXIT: pnl_r = pnl / risk_usdt
    - Sin romper compat:
        - Si el archivo existe con header viejo, NO lo pisa.
        - Crea y usa logs/{symbol}_shadow_trades_v2.csv con columnas extendidas.
    """

    def __init__(self, symbol: str):
        self.symbol = str(symbol).upper().strip()
        os.makedirs("logs", exist_ok=True)

        self.file_trades_v1 = f"logs/{self.symbol}_shadow_trades.csv"
        self.file_bars_v1 = f"logs/{self.symbol}_shadow_bars.csv"

        self.file_trades_v2 = f"logs/{self.symbol}_shadow_trades_v2.csv"
        self.file_bars_v2 = f"logs/{self.symbol}_shadow_bars_v2.csv"

        # =========================
        # TRADE STATE (para pairing)
        # =========================
        self.active_trade = None

        # Detect header existente (si existe)
        existing_header = self._read_header(self.file_trades_v1)
        if existing_header and ("pnl_r" not in existing_header or "risk_usdt" not in existing_header):
            # Usamos v2 para no romper el viejo
            self.file_trades = self.file_trades_v2
        else:
            self.file_trades = self.file_trades_v1

        existing_bars_header = self._read_header(self.file_bars_v1)
        if existing_bars_header:
            self.file_bars = self.file_bars_v1
        else:
            self.file_bars = self.file_bars_v1

        # =========================
        # HEADERS
        # =========================
        self._trades_header_v1 = [
            "trade_id",
            "type",               # ENTRY / EXIT
            "timestamp",
            "side",
            "qty",
            "price",
            "entry_price",
            "exit_price",
            "pnl",
            "pnl_pct",
            "reason",
            "regime",
            "holding_time_sec",
            "mfe",
            "mae",
        ]

        self._trades_header_v2 = self._trades_header_v1 + [
            "risk_usdt",
            "pnl_r",
        ]

        self._bars_header = [
            "timestamp",
            "open", "high", "low", "close", "volume",
            "atr",
            "atr_sma",
            "vwap",
            "regime",
            "inside_ratio",
            "range_invalid",
        ]

        # =========================
        # INIT FILES
        # =========================
        if not os.path.exists(self.file_trades):
            with open(self.file_trades, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = self._trades_header_v2 if self.file_trades.endswith("_v2.csv") else self._trades_header_v1
                w.writerow(header)

        if not os.path.exists(self.file_bars):
            with open(self.file_bars, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(self._bars_header)

    # -------------------------
    # helpers
    # -------------------------
    def _read_header(self, path: str):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                return next(r, None)
        except Exception:
            return None

    def _now_ms(self) -> int:
        return int(datetime.utcnow().timestamp() * 1000)

    def _safe_float(self, x, default=None):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    # ============================================================
    # LOG DE VELAS (BARRAS)
    # ============================================================
    def log_bar(self, **k):
        with open(self.file_bars, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                k.get("timestamp"),
                k.get("open"),
                k.get("high"),
                k.get("low"),
                k.get("close"),
                k.get("volume"),
                k.get("atr"),
                k.get("atr_sma"),
                k.get("vwap"),
                k.get("regime"),
                k.get("inside_ratio"),
                k.get("range_invalid"),
            ])

        # Update MFE / MAE
        if self.active_trade:
            price = self._safe_float(k.get("close"), default=None)
            if price is not None:
                side = self.active_trade["side"]
                entry = self.active_trade["entry_price"]

                if side == "LONG":
                    self.active_trade["mfe"] = max(self.active_trade["mfe"], price - entry)
                    self.active_trade["mae"] = min(self.active_trade["mae"], price - entry)
                else:
                    self.active_trade["mfe"] = max(self.active_trade["mfe"], entry - price)
                    self.active_trade["mae"] = min(self.active_trade["mae"], entry - price)

    # ============================================================
    # LOG DE TRADES
    # ============================================================
    def log_trade(self, **k):
        trade_type = k.get("type") or k.get("type_")
        timestamp = k.get("timestamp") or self._now_ms()

        if trade_type == "ENTRY":
            trade_id = str(uuid.uuid4())

            qty = self._safe_float(k.get("qty"), default=0.0) or 0.0
            price = self._safe_float(k.get("price"), default=0.0) or 0.0

            self.active_trade = {
                "trade_id": trade_id,
                "timestamp": int(timestamp),
                "side": k.get("side"),
                "qty": qty,
                "entry_price": price,
                "regime": k.get("regime"),
                "reason": k.get("reason"),
                "mfe": 0.0,
                "mae": 0.0,
            }

            row_v1 = [
                trade_id,
                "ENTRY",
                timestamp,
                k.get("side"),
                qty,
                price,
                price,
                "",
                0.0,
                0.0,
                k.get("reason"),
                k.get("regime"),
                "",
                "",
                "",
            ]

            if self.file_trades.endswith("_v2.csv"):
                risk_usdt = self._safe_float(k.get("risk_usdt"), default=None)
                row_v2 = row_v1 + [risk_usdt if risk_usdt is not None else "", ""]
                row = row_v2
            else:
                row = row_v1

            with open(self.file_trades, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(row)

        elif trade_type == "EXIT" and self.active_trade:
            entry = self.active_trade

            exit_price = self._safe_float(k.get("price"), default=None)
            if exit_price is None:
                return

            qty = float(entry["qty"])
            side = entry["side"]

            pnl = self._safe_float(k.get("pnl"), default=None)
            if pnl is None:
                # derive PnL
                pnl = (exit_price - entry["entry_price"]) * qty if side == "LONG" else (entry["entry_price"] - exit_price) * qty

            holding_time = int((int(timestamp) - int(entry["timestamp"])) / 1000)

            pnl_pct = k.get("pnl_pct")
            reason = k.get("reason")

            # ✅ FIX: pnl_r
            risk_usdt = self._safe_float(k.get("risk_usdt"), default=None)
            pnl_r = ""
            if risk_usdt is not None and risk_usdt > 0:
                pnl_r = float(pnl) / float(risk_usdt)

            row_v1 = [
                entry["trade_id"],
                "EXIT",
                timestamp,
                side,
                qty,
                exit_price,
                entry["entry_price"],
                exit_price,
                pnl,
                pnl_pct,
                reason,
                entry["regime"],
                holding_time,
                entry["mfe"],
                entry["mae"],
            ]

            if self.file_trades.endswith("_v2.csv"):
                row_v2 = row_v1 + [risk_usdt if risk_usdt is not None else "", pnl_r]
                row = row_v2
            else:
                row = row_v1

            with open(self.file_trades, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(row)

            self.active_trade = None

    # ============================================================
    # COMPAT GENÉRICA
    # ============================================================
    def log(self, **k):
        if "open" in k and "close" in k:
            self.log_bar(**k)
        else:
            self.log_trade(**k)
