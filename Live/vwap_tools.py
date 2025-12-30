# Live/vwap_tools.py
# ================================================================
#   VWAPModule
#   - Calcula VWAP en tiempo real
#   - Genera señales "long", "short" o None
#   - Incluye bandas de desviación estándar
# ================================================================

import pandas as pd
import numpy as np


class VWAPModule:
    def __init__(self, window: int = 30, band_mult: float = 1.5):
        """
        window      → cantidad de velas para calcular VWAP rolling
        band_mult   → multiplicador de desviación (bandas)
        """
        self.window = window
        self.band_mult = band_mult
        self.df = pd.DataFrame(columns=["close", "volume"])

    # ------------------------------------------------------------
    # ACTUALIZACIÓN DE DATA
    # ------------------------------------------------------------
    def update(self, candle: dict):
        """
        Recibe vela:
            {"close": float, "volume": float, ...}
        """
        row = {
            "close": float(candle["close"]),
            "volume": float(candle["volume"])
        }

        self.df.loc[len(self.df)] = row

        if len(self.df) > self.window * 2:
            self.df = self.df.iloc[-self.window * 2:]  # mantener liviano

    # ------------------------------------------------------------
    # CÁLCULOS DE VWAP
    # ------------------------------------------------------------
    def _compute_vwap(self):
        if len(self.df) < 3:
            return None, None, None

        # Rolling VWAP = sum(price*volume) / sum(volume)
        rolling = self.df[-self.window:]
        pv = (rolling["close"] * rolling["volume"]).sum()
        vol = rolling["volume"].sum()

        if vol == 0:
            return None, None, None

        vwap = pv / vol
        std = rolling["close"].std()

        upper = vwap + std * self.band_mult
        lower = vwap - std * self.band_mult

        return vwap, upper, lower

    # ------------------------------------------------------------
    # SEÑAL
    # ------------------------------------------------------------
    def get_signal(self, price: float):
        vwap, upper, lower = self._compute_vwap()

        if vwap is None:
            return None

        # Precio encima pero NO sobreextendido
        if price > vwap and price < upper:
            return "long"

        # Precio debajo pero NO sobreextendido
        if price < vwap and price > lower:
            return "short"

        # Demasiado extendido → evitar operar
        return None
