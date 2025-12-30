import csv
import os
from typing import Optional
from datetime import datetime


class PaperBroker:
    """
    Simulador de broker en papel:
      - Maneja balance y posición
      - Aplica comisiones
      - Loguea trades a CSV
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.00055,
        log_path: str = "paper_trades.csv",
    ):
        self.balance: float = initial_balance  # PnL realizado
        self.position_side: Optional[str] = None  # "LONG" / "SHORT" / None
        self.entry_price: Optional[float] = None
        self.qty: float = 0.0
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.ts_open: Optional[str] = None
        self.commission: float = commission
        self.log_path: str = log_path

        # Crear CSV si no existe
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "ts_open",
                        "ts_close",
                        "side",
                        "entry_price",
                        "exit_price",
                        "qty",
                        "pnl_usd",
                        "pnl_pct",
                        "reason_exit",
                    ]
                )

    # -------------------------------------------------------------

    def _log_trade(
        self,
        ts_open: str,
        ts_close: str,
        side: str,
        entry: float,
        exit_p: float,
        qty: float,
        pnl_usd: float,
        pnl_pct: float,
        reason: str,
    ):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ts_open,
                    ts_close,
                    side,
                    round(entry, 6),
                    round(exit_p, 6),
                    round(qty, 6),
                    round(pnl_usd, 4),
                    round(pnl_pct, 6),
                    reason,
                ]
            )

    # -------------------------------------------------------------

    def open_position(
        self,
        ts: str,
        side: str,
        price: float,
        qty: float,
        sl: float,
        tp: float,
    ):
        """
        Abre una posición simulada.
        side: "LONG" o "SHORT"
        """
        if self.position_side is not None:
            # ya hay posición abierta, no abrimos otra
            return

        fee = qty * price * self.commission
        self.balance -= fee

        self.position_side = side
        self.entry_price = float(price)
        self.qty = float(qty)
        self.stop_loss = float(sl)
        self.take_profit = float(tp)
        self.ts_open = ts

    # -------------------------------------------------------------

    def close_position(self, ts: str, price: float, reason: str = "manual"):
        """
        Cierra la posición actual (si existe) al precio indicado.
        """
        if self.position_side is None or self.entry_price is None:
            return

        price = float(price)

        # PnL bruto
        if self.position_side == "LONG":
            pnl = (price - self.entry_price) * self.qty
        else:
            pnl = (self.entry_price - price) * self.qty

        # comisión de salida
        fee = price * self.qty * self.commission
        pnl -= fee

        self.balance += pnl

        pnl_pct = 0.0
        if self.entry_price > 0:
            pnl_pct = pnl / (self.entry_price * self.qty)

        self._log_trade(
            ts_open=self.ts_open or "",
            ts_close=ts,
            side=self.position_side,
            entry=self.entry_price,
            exit_p=price,
            qty=self.qty,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )

        # Reset posición
        self.position_side = None
        self.entry_price = None
        self.qty = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.ts_open = None

    # -------------------------------------------------------------

    def check_sl_tp(self, ts: str, high: float, low: float):
        """
        Verifica si durante la vela se habría tocado SL o TP.
        Usa high/low de la vela cerrada.
        """
        if self.position_side is None or self.stop_loss is None or self.take_profit is None:
            return

        high = float(high)
        low = float(low)

        if self.position_side == "LONG":
            if low <= self.stop_loss:
                self.close_position(ts, self.stop_loss, reason="SL")
            elif high >= self.take_profit:
                self.close_position(ts, self.take_profit, reason="TP")

        elif self.position_side == "SHORT":
            if high >= self.stop_loss:
                self.close_position(ts, self.stop_loss, reason="SL")
            elif low <= self.take_profit:
                self.close_position(ts, self.take_profit, reason="TP")

    # -------------------------------------------------------------

    def get_equity(self, current_price: float) -> float:
        """
        Equity flotante = balance realizado + PnL no realizado de la posición actual.
        """
        equity = self.balance
        if self.position_side is not None and self.entry_price is not None:
            current_price = float(current_price)
            if self.position_side == "LONG":
                pnl_f = (current_price - self.entry_price) * self.qty
            else:
                pnl_f = (self.entry_price - current_price) * self.qty
            equity += pnl_f
        return equity
