# Live/ws_futures_1m.py
# ================================================================
#   WebSocket 1m para Binance Futures (SOLUSDT u otro símbolo)
#   - SOLO se encarga de recibir velas 1m y entregarlas al bot
#   - No conoce nada de estrategias, riesgo ni enrutadores
#   - Evita bucles de importación (no importa módulos Live.*)
# ================================================================

from __future__ import annotations

import asyncio
import json
import threading
import traceback
from typing import Any, Callable, Dict, Optional

try:
    import websockets  # type: ignore
except Exception:  # websockets no instalado
    websockets = None


def _build_candle_from_kline(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convierte el payload de kline de Binance en un diccionario simple.

    Espera un mensaje del estilo:
    {
        "e": "kline",
        "k": {
            "o": "...", "h": "...", "l": "...", "c": "...",
            "v": "...", "T": 1690000000000, "x": true/false, ...
        }
    }
    """
    try:
        k = msg["k"]
        return {
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "timestamp": int(k["T"]),
            "is_closed": bool(k["x"]),
        }
    except Exception:
        return None


class FuturesWS_1m:
    """
    WebSocket 1m desacoplado:

    - Recibe:
        * symbol: str (ej. "SOLUSDT")
        * on_candle: callback que recibe un dict con la vela cerrada
        * bus: EventBus opcional → emite evento "bar_1m"
        * delta_router: objeto opcional (se guarda por si quieres usarlo)
        * logger: TradeLogger opcional (se guarda para logs si hace falta)

    - Métodos públicos:
        * start()  → lanza thread con loop asyncio y mantiene el WS
        * stop()   → cierra el WS y apaga el loop
        * restart()→ stop() + start()
    """

    def __init__(
        self,
        symbol: str,
        on_candle: Callable[[Dict[str, Any]], None],
        bus: Optional[Any] = None,
        delta_router: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.symbol = symbol.lower()
        self.on_candle = on_candle
        self.bus = bus
        self.delta_router = delta_router
        self.logger = logger

        self._ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_1m"

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------
    # Control del thread / loop
    # ------------------------------------------------------------
    def start(self) -> None:
        """Arranca el WebSocket en un thread separado."""
        if websockets is None:
            print("[WS] ERROR: módulo 'websockets' no está instalado.")
            return

        if self._thread and self._thread.is_alive():
            # Ya está corriendo
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[WS] Iniciado FuturesWS_1m para {self.symbol.upper()}")

    def _run_loop(self) -> None:
        """Crea un event loop propio y ejecuta el loop del WebSocket."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._ws_loop())
        except Exception as e:
            print("[WS] ERROR fatal en loop asyncio:", e)
            traceback.print_exc()
        finally:
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                    self._loop.close()
            except Exception:
                pass

    async def _ws_loop(self) -> None:
        """Loop principal de conexión / reconexión al WebSocket."""
        assert websockets is not None

        while not self._stop_event.is_set():
            try:
                print(f"[WS] Conectando a {self._ws_url} ...")
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    print("[WS] Conectado correctamente.")

                    while not self._stop_event.is_set():
                        raw_msg = await ws.recv()
                        await self._handle_message(raw_msg)

            except asyncio.CancelledError:
                # Cierre limpio del loop
                break
            except Exception as e:
                print("[WS] Error en WebSocket, reconectando en 5s:", e)
                traceback.print_exc()
                await asyncio.sleep(5)

        print("[WS] Loop WebSocket terminado.")

    async def _handle_message(self, raw_msg: str) -> None:
        """Parsea el mensaje crudo y dispara eventos / callbacks."""
        try:
            data = json.loads(raw_msg)

            # Solo nos interesan mensajes de tipo kline
            if data.get("e") != "kline":
                return

            candle = _build_candle_from_kline(data)
            if not candle:
                return

            # Sólo procesamos velas cerradas
            if not candle["is_closed"]:
                return

            # 1) Callback directo hacia la estrategia
            if self.on_candle is not None:
                try:
                    self.on_candle(candle)
                except Exception as e:
                    print("[WS] Error en callback on_candle:", e)
                    traceback.print_exc()

            # 2) EventBus opcional
            if self.bus is not None:
                try:
                    self.bus.emit("bar_1m", candle)
                except Exception as e:
                    print("[WS] Error emitiendo evento 'bar_1m':", e)

            # 3) DeltaRouter opcional (si más adelante quieres enganchar algo aquí)
            if self.delta_router is not None and hasattr(self.delta_router, "on_new_candle"):
                try:
                    self.delta_router.on_new_candle(candle)  # type: ignore[attr-defined]
                except Exception as e:
                    print("[WS] Error notificando DeltaRouter:", e)

        except json.JSONDecodeError:
            # Mensaje corrupto / ping-pong interno, lo ignoramos
            return
        except Exception as e:
            print("[WS] Error procesando mensaje:", e)
            traceback.print_exc()

    # ------------------------------------------------------------
    #  API de parada / reinicio
    # ------------------------------------------------------------
    def stop(self) -> None:
        """Detiene el WebSocket y cierra el event loop."""
        self._stop_event.set()

        if self._loop is not None:
            try:
                # Pedimos que se detenga el loop desde el thread principal
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        print("[WS] FuturesWS_1m detenido.")

    def restart(self) -> None:
        """Reinicia el WebSocket (stop + start)."""
        self.stop()
        self.start()


# ================================================================
#  Prueba rápida standalone (no se ejecuta en run_live)
# ================================================================
if __name__ == "__main__":
    import time

    def _debug_on_candle(candle: Dict[str, Any]) -> None:
        print("[DEBUG] Nueva vela cerrada:", candle)

    ws = FuturesWS_1m(symbol="btcusdt", on_candle=_debug_on_candle)
    ws.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ws.stop()
        print("Bye!")

