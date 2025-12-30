# Live/stream_klines.py

import asyncio
import json
from datetime import datetime

import websockets  # pip install websockets


BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"


def _build_stream_name(symbol: str, interval: str) -> str:
    """
    Construye el nombre de stream para Binance Futures.
    Ej: SOLUSDT + 1m -> "solusdt@kline_1m"
    """
    return f"{symbol.lower()}@kline_{interval}"


async def stream_klines(symbol: str, interval: str):
    """
    Async generator que streamea velas (klines) de Binance Futures.

    Devuelve diccionarios con:
        symbol, interval,
        open, high, low, close, volume,
        ts_open, ts_close, ts_event,
        is_closed
    """
    stream_name = _build_stream_name(symbol, interval)
    url = f"{BINANCE_FUTURES_WS}/{stream_name}"

    print(f"[KLINES] Conectando a {url}")

    # Reconexión automática
    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                max_queue=2000,
            ) as ws:

                print("[KLINES] Conectado a Binance Futures stream")

                async for msg in ws:
                    data = json.loads(msg)
                    k = data.get("k")
                    if not k:
                        continue

                    candle = {
                        "symbol": k["s"],
                        "interval": k["i"],
                        "open": float(k["o"]),
                        "high": float(k["h"]),
                        "low": float(k["l"]),
                        "close": float(k["c"]),
                        "volume": float(k["v"]),
                        "ts_open": int(k["t"]),
                        "ts_close": int(k["T"]),
                        "ts_event": int(data.get("E", k["T"])),
                        "is_closed": bool(k["x"]),
                    }

                    yield candle

        except Exception as e:
            print(f"[KLINES] Error: {e}. Reintentando en 5s...")
            await asyncio.sleep(5)


# ============================================================
# TEST STANDALONE
# ============================================================

if __name__ == "__main__":
    async def _test():
        symbol = "SOLUSDT"
        interval = "1m"

        print(f"Testeando stream_klines({symbol}, {interval}) ...")
        count = 0

        async for candle in stream_klines(symbol, interval):
            if candle["is_closed"]:
                ts = datetime.fromtimestamp(candle["ts_close"] / 1000.0)
                print(
                    f"[{ts}] O={candle['open']} "
                    f"H={candle['high']} L={candle['low']} "
                    f"C={candle['close']} V={candle['volume']}"
                )
                count += 1

            if count >= 3:
                print("Recibidas 3 velas cerradas. Test OK.")
                break

    asyncio.run(_test())
