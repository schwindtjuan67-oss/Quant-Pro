import asyncio
from datetime import datetime
from Live.stream_klines import stream_klines


class ShadowTest:
    def __init__(self, symbol="SOLUSDT", interval="1m"):
        self.symbol = symbol
        self.interval = interval
        self.counter = 0

    async def start(self):
        print(f"[SHADOW-TEST] Iniciando Shadow Test para {self.symbol} {self.interval}")

        async for candle in stream_klines(self.symbol, self.interval):

            # Solo cuando la vela se cierra
            if candle["is_closed"]:
                self.counter += 1

                ts = datetime.fromtimestamp(candle["ts_close"] / 1000)

                print(
                    f"[{self.counter}] {ts} "
                    f"O={candle['open']} H={candle['high']} "
                    f"L={candle['low']} C={candle['close']} V={candle['volume']}"
                )


if __name__ == "__main__":
    engine = ShadowTest("SOLUSDT", "1m")
    asyncio.run(engine.start())
