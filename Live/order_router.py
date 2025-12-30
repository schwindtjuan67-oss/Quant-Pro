# Live/order_router.py

import time
import traceback
import os
import sys
from typing import Optional
from dotenv import load_dotenv

from Live.binance_client import BinanceClient

# Asegurar ROOT en sys.path
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Cargar .env del root del proyecto si existe
load_dotenv(os.path.join(ROOT, ".env"))


class OrderRouter:
    """
    Envoltura sobre BinanceClient para:
      - Enviar órdenes BUY/SELL (LONG/SHORT)
      - Cooldown inteligente:
          * NO bloquea "cierres" (reduce-only)
          * Sí bloquea spam de "entradas" consecutivas
      - Estado mínimo de posición (paper): evita desincronización si hay flips.
    """

    def __init__(
        self,
        client: Optional[BinanceClient] = None,
        symbol: str = "SOLUSDT",
        paper_trade: bool = True,
        testnet: bool = False,
        logger=None,
        # Cooldown recomendado para evitar spam (solo para ENTRADAS nuevas)
        entry_cooldown_secs: float = 8.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.paper_trade = paper_trade
        self.testnet = testnet
        self.logger = logger

        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("BINANCE_API_KEY") or ""
            api_secret = os.getenv("BINANCE_API_SECRET") or ""

            if not self.paper_trade and (not api_key or not api_secret):
                print(
                    "[ORDER_ROUTER] ADVERTENCIA: paper_trade=False pero faltan BINANCE_API_KEY/BINANCE_API_SECRET en .env. Pasando a paper."
                )
                self.paper_trade = True

            self.client = BinanceClient(
                symbol=self.symbol,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                paper_trade=self.paper_trade,
            )

        # Cooldown: sólo aplica a ENTRADAS nuevas (no a cierres)
        self.entry_cooldown_secs = float(entry_cooldown_secs)
        self._last_entry_ts = 0.0

        # Estado mínimo de posición (paper)
        # side: "LONG" | "SHORT" | None
        self.position_side: Optional[str] = None
        self.position_qty: float = 0.0

        print(
            f"[ORDER_ROUTER] Iniciado para symbol={self.symbol} paper_trade={self.paper_trade} testnet={self.testnet} entry_cooldown_secs={self.entry_cooldown_secs}"
        )

    # ----------------------------------------------------------
    def get_balance(self) -> float:
        try:
            return float(getattr(self.client, "balance", 0.0))
        except Exception:
            return 0.0

    # ----------------------------------------------------------
    def _normalize_side(self, side: str) -> str:
        s = (side or "").upper().strip()
        if s in ("BUY", "LONG"):
            return "BUY"
        if s in ("SELL", "SHORT"):
            return "SELL"
        return s

    def _is_close_or_reduce(self, side_u: str) -> bool:
        """
        Determina si la orden es un cierre/reducción de posición actual.
        BUY cierra/reduce SHORT
        SELL cierra/reduce LONG
        """
        if self.position_side is None or self.position_qty <= 0:
            return False

        if side_u == "BUY" and self.position_side == "SHORT":
            return True
        if side_u == "SELL" and self.position_side == "LONG":
            return True
        return False

    def _apply_paper_position(self, side_u: str, qty: float) -> None:
        """
        Actualiza estado mínimo local (paper) para evitar desync.
        Suposición:
          - BUY abre/aumenta LONG o reduce/cierra SHORT
          - SELL abre/aumenta SHORT o reduce/cierra LONG
        """
        qty = float(qty or 0.0)
        if qty <= 0:
            return

        if self.position_side is None or self.position_qty <= 0:
            # Abrir desde FLAT
            self.position_side = "LONG" if side_u == "BUY" else "SHORT"
            self.position_qty = qty
            return

        if self.position_side == "LONG":
            if side_u == "BUY":
                # aumentar long
                self.position_qty += qty
            else:
                # SELL reduce/cierra long
                self.position_qty -= qty
                if self.position_qty <= 0:
                    self.position_side = None
                    self.position_qty = 0.0

        elif self.position_side == "SHORT":
            if side_u == "SELL":
                # aumentar short
                self.position_qty += qty
            else:
                # BUY reduce/cierra short
                self.position_qty -= qty
                if self.position_qty <= 0:
                    self.position_side = None
                    self.position_qty = 0.0

    # ----------------------------------------------------------
    def market_order(self, side: str, qty: float) -> bool:
        """
        Ejecuta una orden de mercado BUY/SELL.
        side: "BUY", "SELL", "LONG", "SHORT"
        """
        side_u = self._normalize_side(side)
        if side_u not in ("BUY", "SELL"):
            print(f"[ORDER_ROUTER] Side desconocido: {side}")
            return False

        qty = float(qty or 0.0)
        if qty <= 0:
            print("[ORDER_ROUTER] qty inválida (<=0). Se ignora.")
            return False

        # Cooldown inteligente:
        # - Si es cierre/reduce (opuesto a posición actual), NO bloqueamos.
        is_reduce = self._is_close_or_reduce(side_u)
        now = time.time()

        if (not is_reduce) and (now - self._last_entry_ts < self.entry_cooldown_secs):
            print(f"[ORDER_ROUTER] Entry cooldown activo ({self.entry_cooldown_secs}s), se ignora orden {side_u}")
            return False

        try:
            if side_u == "BUY":
                print(f"[ORDER_ROUTER] Enviando MARKET BUY qty={qty:.6f} reduce_only={is_reduce}")
                ok = self.client.market_buy(qty)
            else:
                print(f"[ORDER_ROUTER] Enviando MARKET SELL qty={qty:.6f} reduce_only={is_reduce}")
                ok = self.client.market_sell(qty)

            ok = bool(ok)
            if not ok:
                print("[ORDER_ROUTER] La orden no fue confirmada por BinanceClient")
                return False

            # Si fue una entrada (no reduce), registrar cooldown
            if not is_reduce:
                self._last_entry_ts = now

            # Actualizar estado mínimo local (paper)
            # (incluso si el client ya tiene algo, esto te salva de desync en flips)
            self._apply_paper_position(side_u, qty)

            return True

        except Exception as e:
            print(f"[ORDER_ROUTER] ERROR enviando orden: {e}")
            traceback.print_exc()
            return False
