# Live/trend_risk_tools.py
# ================================================================
#   TrendRiskModule
#   - Filtro de tendencia basado en EMA rápida y EMA lenta
#   - Genera señales "long", "short" o None
#   - Diseñado para trading en 1m (robusto contra ruido)
# ================================================================

import pandas as pd


class TrendRiskModule:
    def __init__(self, data, ema_fast: int = 20, ema_slow: int = 50):
        """
        data: DataFrame con columnas:
            ['open', 'high', 'low', 'close', 'volume']
        """
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

        # Pre-calcular EMAs en el dataset inicial
        self.df = data.copy()
        if not self.df.empty:
            self.df["ema_fast"] = self.df["close"].ewm(
                span=self.ema_fast, adjust=False
            ).mean()
            self.df["ema_slow"] = self.df["close"].ewm(
                span=self.ema_slow, adjust=False
            ).mean()

    # ------------------------------------------------------------
    def update(self, candle: dict):
        """
        Agrega una nueva vela y actualiza EMAs.
        """
        new_row = {
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle["volume"]),
        }
        self.df.loc[len(self.df)] = new_row

        self.df["ema_fast"] = self.df["close"].ewm(
            span=self.ema_fast, adjust=False
        ).mean()
        self.df["ema_slow"] = self.df["close"].ewm(
            span=self.ema_slow, adjust=False
        ).mean()

    # ------------------------------------------------------------
    def get_signal(self, price: float):
        """
        Devuelve:
            "long"  → si ema_fast > ema_slow
            "short" → si ema_fast < ema_slow
            None    → si no hay claridad aún
        """
        if self.df.empty or "ema_fast" not in self.df or "ema_slow" not in self.df:
            return None

        fast = float(self.df["ema_fast"].iloc[-1])
        slow = float(self.df["ema_slow"].iloc[-1])

        if fast > slow:
            return "long"
        elif fast < slow:
            return "short"
        return None