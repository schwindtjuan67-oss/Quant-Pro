# Live/order_manager_shadow.py

class FakeClientShadow:
    """
    Cliente falso para ShadowOrderManager.
    Simula lo mínimo que HybridScalperPRO necesita:
    - balance
    - sync_balance_from_exchange()
    """
    def __init__(self, starting_equity=1000.0):
        self.balance = starting_equity

    def sync_balance_from_exchange(self):
        # En modo shadow NO hay exchange; solo devolvemos el balance actual
        return self.balance


class ShadowOrderManager:
    """
    Order manager para modo SHADOW.
    No ejecuta órdenes reales, solo simula entradas/salidas en memoria.
    """

    def __init__(self, symbol, config):
        self.symbol = symbol
        self.balance = 1000.0   # Equity base para shadow
        self.client = FakeClientShadow(self.balance)   # <<<<<< 🔥 FIX CLAVE
        self.position = None
        self.qty = 0.0

    def get_balance(self):
        return self.client.balance

    def market_order(self, side, qty):
        """
        En shadow, no enviamos órdenes reales:
        solo marcamos estado interno.
        """
        self.position = side
        self.qty = qty
        return True
