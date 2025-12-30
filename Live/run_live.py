# Live/run_live.py
# ============================================================
#       RUNNER LIVE FINAL — Hybrid Scalper PRO (1m)
# ============================================================

import time
import json
import traceback

from Live.binance_client import BinanceClient
from Live.order_router import OrderRouter
from Live.risk_manager import RiskManager
from Live.logger_pro import TradeLogger
from Live.delta_router import DeltaRouter
from Live.event_bus import EventBus
from Live.position_recovery import PositionRecovery

from .hybrid_scalper_pro import HybridScalperPRO
from .ws_futures_1m import FuturesWS_1m      # Tu WebSocket 1m real


# ============================================================
#                CONFIGURACIÓN PRINCIPAL
# ============================================================

SYMBOL = "SOLUSDT"

API_KEY    = "TU_API_KEY"
API_SECRET = "TU_API_SECRET"

USE_TESTNET = False
PAPER_MODE  = False         # EN LIVE REAL: False


# ============================================================
#                MAIN: CREACIÓN DE COMPONENTES
# ============================================================

def main():

    print("\n==========================")
    print("   INICIANDO LIVE BOT")
    print("==========================\n")

    # -----------------------------------------
    # 1. LOGGER (barras, trades, estado)
    # -----------------------------------------
    logger = TradeLogger(SYMBOL)

    # -----------------------------------------
    # 2. CLIENT → Binance
    # -----------------------------------------
    client = BinanceClient(
        symbol=SYMBOL,
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=USE_TESTNET,
        paper_trade=PAPER_MODE,
        initial_balance=100.0        # Se corrige luego con equity real
    )

    # -----------------------------------------
    # 3. ROUTER (con client y logger)
    # -----------------------------------------
    router = OrderRouter(
        client=client,
        symbol=SYMBOL,
        logger=logger
    )

    # -----------------------------------------
    # 4. RISK MANAGER — se actualiza con equity real
    # -----------------------------------------
    risk_manager = RiskManager(
        starting_equity=client.balance,
        max_loss_pct=0.03,
        max_dd_pct=0.04,
        max_trades=12
    )

    # -----------------------------------------
    # 5. EVENT BUS → para Telegram alerts
    # -----------------------------------------
    bus = EventBus()

    # -----------------------------------------
    # 6. DELTA ROUTER (si lo usás)
    # -----------------------------------------
    delta_router = DeltaRouter(bus)

    # -----------------------------------------
    # 7. POSITION RECOVERY — CRUCIAL
    # -----------------------------------------
    print("[LIVE] Ejecutando Position Recovery...")
    recovery = PositionRecovery(
        symbol=SYMBOL,
        router=router,
        risk_manager=risk_manager,
        logger=logger
    )

    state = recovery.recover()

    # Si existe equity guardada → actualizar RM
    if state and "equity" in state:
        risk_manager.update_equity(state["equity"])

    print("[LIVE] Estado inicial listo.\n")

    # -----------------------------------------
    # 8. CREAR ESTRATEGIA SCALPER PRO
    # -----------------------------------------
    strategy = HybridScalperPRO(
        symbol=SYMBOL,
        router=router,
        delta_router=delta_router,
        risk_manager=risk_manager,
        event_bus=bus,
        logger=logger
    )

    # -----------------------------------------
    # 9. CREAR WEBSOCKET
    # -----------------------------------------
    ws = FuturesWS_1m(
        symbol=SYMBOL,
        on_candle=strategy.on_bar,
        bus=bus,
        delta_router=delta_router,
        logger=logger
    )

    # -----------------------------------------
    # 10. LOOP PRINCIPAL DEL BOT
    # -----------------------------------------
    print("[LIVE] Conectando WebSocket...")
    ws.start()

    print("[LIVE] Bot funcionando en tiempo real.\n")

    # Loop de mantenimiento
    while True:
        try:
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n[STOP] Bot detenido manualmente.")
            ws.stop()
            break

        except Exception as e:
            print("\n[ERROR] en loop principal:", e)
            traceback.print_exc()
            ws.restart()   # Asume que tu ws tiene método restart()


# ============================================================
#                    EJECUCIÓN DIRECTA
# ============================================================
if __name__ == "__main__":
    main()
