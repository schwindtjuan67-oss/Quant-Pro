# Live/delta_live.py

import threading
import time
import json
import websocket
import traceback

class DeltaSnapshot:
    def __init__(self):
        self.last_ts = 0
        self.delta_candle = 0
        self.delta_candle_prev = 0
        self.delta_rolling_15s = 0
        self.delta_rolling_60s = 0
        self.trades_count_window = 0

    def as_dict(self):
        return {
            "last_ts": self.last_ts,
            "delta_candle": self.delta_candle,
            "delta_candle_prev": self.delta_candle_prev,
            "delta_rolling_15s": self.delta_rolling_15s,
            "delta_rolling_60s": self.delta_rolling_60s,
            "trades_count_window": self.trades_count_window,
        }


class DeltaLive:
    def __init__(self, symbol):
        self.symbol = symbol.lower()

        # FIX: Binance Futures NO soporta aggTrade → usar @trade
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@trade"

        self.snapshot = DeltaSnapshot()

        self.trades_60s = []
        self.trades_15s = []
        self.trades_candle = []

        self.ws = None
        self.thread = None
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_ws, daemon=True)
        self.thread.start()
        print(f"[DELTA] Iniciado DeltaLive para {self.symbol.upper()} en {self.ws_url}")

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

    def _run_ws(self):
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws.on_open = self._on_open
                self.ws.run_forever()
            except Exception:
                traceback.print_exc()

            print("[DELTA] WS cerrado, reconectando en 3s...")
            time.sleep(3)

    def _on_open(self, ws):
        print(f"[DELTA] WebSocket trade conectado para {self.symbol.upper()}")

    def _on_close(self, ws, *args):
        print("[DELTA] WebSocket cerrado")

    def _on_error(self, ws, e):
        print(f"[DELTA] ERROR: {e}")

    def _on_message(self, ws, msg):
        try:
            data = json.loads(msg)
            ts = int(data["T"])
            qty = float(data["q"])
            maker = data["m"]

            # igual que antes
            delta = -qty if maker else qty
            self.snapshot.last_ts = ts

            now = time.time()
            self.trades_60s.append((now, delta))
            self.trades_15s.append((now, delta))
            self.trades_candle.append((ts, delta))

            self._cleanup()

            self.snapshot.delta_rolling_15s = sum(x[1] for x in self.trades_15s)
            self.snapshot.delta_rolling_60s = sum(x[1] for x in self.trades_60s)
            self.snapshot.trades_count_window = len(self.trades_60s)

            current_candle = ts // 60000
            trades_current = [v for (t, v) in self.trades_candle if (t // 60000) == current_candle]
            trades_prev = [v for (t, v) in self.trades_candle if (t // 60000) == current_candle - 1]

            self.snapshot.delta_candle = sum(trades_current)
            self.snapshot.delta_candle_prev = sum(trades_prev)

        except:
            traceback.print_exc()

    def _cleanup(self):
        cutoff_60 = time.time() - 60
        cutoff_15 = time.time() - 15

        self.trades_60s = [(t, d) for (t, d) in self.trades_60s if t >= cutoff_60]
        self.trades_15s = [(t, d) for (t, d) in self.trades_15s if t >= cutoff_15]

    def get_snapshot(self):
        return self.snapshot
