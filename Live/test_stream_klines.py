# Al final de Live/stream_klines.py

if __name__ == "__main__":
    async def _test():
        symbol = "SOLUSDT"
        interval = "1m"

        print(f"Testeando stream_klines({symbol}, {interval}) ...")
        count = 0

        async for candle in stream_klines(symbol, interval):
            # Mostramos solo cuando la vela se cierra
            if candle["is_closed"]:
                ts = datetime.fromtimestamp(candle["ts_close"] / 1000.0)
                print(
                    f"[{ts}] O={candle['open']} "
                    f"H={candle['high']} L={candle['low']} "
                    f"C={candle['close']} V={candle['volume']}"
                )
                count += 1

            if count >= 5:
                print("Recibidas 5 velas cerradas. Test OK.")
                break

    asyncio.run(_test())
