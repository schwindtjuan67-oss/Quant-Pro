# Live/run_shadow.py
# ============================================================
#     SHADOW MODE — LIVE 1M (SIN ENVIAR ÓRDENES REALES)
# ============================================================

import asyncio
import json
import traceback

from Live.logger_pro import TradeLogger
from Live.binance_client import BinanceClient
from Live.order_router import OrderRouter
from Live.risk_manager import RiskManager
from Live.event_bus import EventBus
from Live.position_recovery import PositionRecovery
from Live.hybrid_scalper_pro import HybridScalperPRO

try:
    import websockets
except Exception:
    websockets = None

SYMBOL = "SOLUSDT"
API_KEY = "SHADOW_KEY"
API_SECRET = "SHADOW_SECRET"
USE_TESTNET = False
PAPER_MODE = True


def main():
    print("\n===============================")
    print("     SHADOW MODE — BOT 1m")
    print("===============================\n")

    logger = TradeLogger(SYMBOL)
    print("[SHADOW] TradeLogger inicializado.\n")

    client = BinanceClient(
        symbol=SYMBOL,
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=USE_TESTNET,
        paper_trade=PAPER_MODE,
        initial_balance=100.0,
    )

    router = OrderRouter(client=client, symbol=SYMBOL, logger=logger)

    risk_manager = RiskManager(
        starting_equity=client.balance,
        max_loss_pct=0.03,
        max_dd_pct=0.04,
        max_trades=12,
    )

    event_bus = EventBus()

    print("[SHADOW] Ejecutando Position Recovery...")
    recovery = PositionRecovery(symbol=SYMBOL, router=router, risk_manager=risk_manager, logger=logger)
    state = recovery.recover()
    if state and "equity" in state:
        risk_manager.update_equity(state["equity"])
    print("[SHADOW] Estado inicial cargado.\n")

    strategy = HybridScalperPRO(
        symbol=SYMBOL,
        router=router,
        delta_router=None,
        risk_manager=risk_manager,
        event_bus=event_bus,
        logger=logger,
    )
    print("[SHADOW] HybridScalperPRO inicializado. Conectando stream 1m...\n")

    def process_candle(candle):
        try:
            strategy.on_bar(candle)
        except Exception:
            print("[SHADOW] Error al procesar vela:")
            traceback.print_exc()

    async def _ws_loop():
        if websockets is None:
            print("[SHADOW] websockets no disponible.")
            while True:
                await asyncio.sleep(1)

        stream = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@kline_1m"

        while True:
            try:
                print("[SHADOW] Conectando websocket...")
                async with websockets.connect(stream, ping_interval=20) as ws:
                    print("[SHADOW] Websocket conectado. Escuchando 1m...\n")
                    async for raw in ws:
                        data = json.loads(raw)
                        if "k" not in data:
                            continue
                        k = data["k"]
                        if not k["x"]:
                            continue

                        candle = {
                            "open": float(k["o"]),
                            "high": float(k["h"]),
                            "low": float(k["l"]),
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                            "timestamp": int(k["T"]),
                        }
                        process_candle(candle)

            except Exception as e:
                print("[SHADOW] Error socket, reconectando en 5s:", e)
                await asyncio.sleep(5)

    try:
        asyncio.run(_ws_loop())
    except KeyboardInterrupt:
        print("\n[SHADOW] Bot detenido manualmente.")


if __name__ == "__main__":
    main()
