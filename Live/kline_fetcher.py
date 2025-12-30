# Live/kline_fetcher.py
# ================================================================
#   Descarga velas históricas desde Binance Futures (REST)
#   y permite ir agregando nuevas velas 1m al dataset.
# ================================================================

import requests
import pandas as pd
from datetime import datetime


BINANCE_FUTURES_REST = "https://fapi.binance.com/fapi/v1/klines"


class KlineFetcher:
    def __init__(self, symbol: str, interval: str = "1m", limit: int = 500):
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit

    # ------------------------------------------------------------
    # DESCARGA DE VELAS INICIALES
    # ------------------------------------------------------------
    def load_initial_data(self):
        """
        Devuelve un DataFrame con velas históricas.
        Columnas:
            timestamp, open, high, low, close, volume
        """
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.limit,
        }

        try:
            r = requests.get(BINANCE_FUTURES_REST, params=params, timeout=10)
            data = r.json()

            rows = []
            for k in data:
                rows.append({
                    "timestamp": int(k[6]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })

            df = pd.DataFrame(rows)
            return df

        except Exception as e:
            print("[KLINE_FETCHER] Error descargando velas iniciales:", e)
            return pd.DataFrame([])

    # ------------------------------------------------------------
    # ACTUALIZACIÓN DE VELAS
    # ------------------------------------------------------------
    def append_new_candle(self, df, candle: dict):
        """
        Agrega una nueva vela cerrada al DataFrame existente.
        """
        try:
            new_row = {
                "timestamp": int(candle["timestamp"]),
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle["volume"]),
            }
            df.loc[len(df)] = new_row
            return df
        except Exception as e:
            print("[KLINE_APPEND] Error agregando nueva vela:", e)
            return df
