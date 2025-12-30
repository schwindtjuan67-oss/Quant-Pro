# Live/binance_client.py

from binance.client import Client as _BinanceClient


class BinanceClient:
    def __init__(
        self,
        symbol,
        api_key,
        api_secret,
        testnet=False,
        paper_trade=True,
        initial_balance=100.0
    ):
        self.symbol = symbol
        self.paper_trade = paper_trade
        self.balance = float(initial_balance)
        self.testnet = testnet

        self._client = None

        if not paper_trade:
            try:
                self._client = _BinanceClient(
                    api_key,
                    api_secret,
                    testnet=testnet
                )
                print("[BINANCE_CLIENT] Live mode activado.")
            except Exception as e:
                print(f"[BINANCE_CLIENT] ERROR iniciando cliente live: {e}")
                print("[BINANCE_CLIENT] Revirtiendo a PAPER MODE.")
                self.paper_trade = True

        if self.paper_trade:
            print(f"[BINANCE_CLIENT] PAPER MODE | balance_virtual={self.balance}")

    # =============================================================
    #  LIVE ORDER EXECUTION
    # =============================================================
    def _live_market_order(self, side, qty):
        try:
            if not self._client:
                return False

            side_u = "BUY" if side.upper() == "BUY" else "SELL"

            self._client.futures_create_order(
                symbol=self.symbol,
                side=side_u,
                type="MARKET",
                quantity=float(qty),
            )
            return True

        except Exception as e:
            print(f"[BINANCE_CLIENT] Error enviando orden live: {e}")
            return False

    # =============================================================
    #  PUBLIC OPERATIONS
    # =============================================================
    def market_buy(self, qty):
        if self.paper_trade:
            print(f"[PAPER] BUY {self.symbol} qty={qty}")
            return True
        return self._live_market_order("BUY", qty)

    def market_sell(self, qty):
        if self.paper_trade:
            print(f"[PAPER] SELL {self.symbol} qty={qty}")
            return True
        return self._live_market_order("SELL", qty)

    # =============================================================
    #   POSITION READER
    # =============================================================
    def get_open_position(self):
        """
        Devuelve dict con:
        - positionAmt
        - side ("LONG" | "SHORT")
        - entryPrice
        Si no hay posición → None
        """
        try:
            if self.paper_trade or not self._client:
                return None

            info = self._client.futures_position_information(symbol=self.symbol)
            if not info:
                return None

            pos = info[0]
            amt = float(pos["positionAmt"])
            entry = float(pos["entryPrice"])

            if amt == 0:
                return None

            side = "LONG" if amt > 0 else "SHORT"

            return {
                "positionAmt": amt,
                "side": side,
                "entryPrice": entry,
            }

        except Exception as e:
            print(f"[BINANCE_CLIENT] Error leyendo posición real: {e}")
            return None

    # =============================================================
    #   BALANCE READER
    # =============================================================
    def sync_balance_from_exchange(self):
        """
        Actualiza self.balance con equity real de Binance Futures
        """
        try:
            if self.paper_trade or not self._client:
                return self.balance

            balances = self._client.futures_account_balance()

            for b in balances:
                if b["asset"] == "USDT":
                    self.balance = float(b["balance"])
                    break

            return self.balance

        except Exception as e:
            print(f"[BINANCE_CLIENT] Error leyendo balance real: {e}")
            return self.balance

    # =============================================================
    def get_balance(self):
        return self.balance

