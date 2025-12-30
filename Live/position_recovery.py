# Live/position_recovery.py

import os
import json

class PositionRecovery:
    """
    Sincroniza el estado local con la posición REAL en Binance Futures.
    NOTA IMPORTANTE:
    - Este módulo NO cierra ni abre posiciones.
    - Solo alinea:
        * position_qty
        * position_side
        * entry_price
        * equity (balance real)
    - Evita inconsistencias al reiniciar el bot.
    """

    def __init__(self, symbol, router, risk_manager, logger):
        self.symbol = symbol
        self.router = router          # OrderRouter
        self.risk_manager = risk_manager
        self.logger = logger          # TradeLogger (opcional)

    # --------------------------------------------
    #   Lectura del estado local (JSON)
    # --------------------------------------------
    def _load_local_state(self):
        try:
            folder = os.path.join("Live", "state")
            path = os.path.join(folder, f"state_{self.symbol}.json")

            if not os.path.exists(path):
                return None

            with open(path, "r") as f:
                return json.load(f)

        except Exception as e:
            print(f"[RECOVERY] Error cargando estado local: {e}")
            return None

    # --------------------------------------------
    #   Guardado del estado local
    # --------------------------------------------
    def _save_local_state(self, state):
        try:
            folder = os.path.join("Live", "state")
            os.makedirs(folder, exist_ok=True)

            path = os.path.join(folder, f"state_{self.symbol}.json")

            with open(path, "w") as f:
                json.dump(state, f, indent=4)

        except Exception as e:
            print(f"[RECOVERY] Error guardando estado local: {e}")

    # --------------------------------------------
    #   Procedimiento de Recovery
    # --------------------------------------------
    def recover(self):
        """
        Procedimiento principal:

        1. Lee posición REAL en Binance via router.client.get_open_position()
        2. Sincroniza equity REAL via sync_balance_from_exchange()
        3. Ajusta el estado local en base a la posición real
        4. Guarda state_X.json en Live/state/
        5. Devuelve el estado alineado
        """

        print(f"[RECOVERY] Iniciando Position Recovery para {self.symbol}...")

        # 1) Posición real en Binance
        exch_pos = self.router.client.get_open_position()

        # 2) Equity real
        real_equity = self.router.client.sync_balance_from_exchange()
        self.risk_manager.update_equity(real_equity)
        print(f"[RECOVERY] Equity real detected: {real_equity}")

        # 3) Estado local (si existe)
        state = self._load_local_state() or {}

        local_qty = float(state.get("position_qty", 0.0))
        local_side = state.get("position_side", "FLAT")

        # ---------------------------------
        #   EXCHANGE → FLAT
        # ---------------------------------
        if exch_pos is None:
            print("[RECOVERY] Exchange: FLAT")

            # Si local tenía posición, corregir
            if local_qty != 0:
                print(f"[RECOVERY] Corrigiendo estado local ({local_side} {local_qty}) → FLAT.")

            state["position_qty"] = 0.0
            state["position_side"] = "FLAT"
            state["entry_price"] = None

        # ---------------------------------
        #   EXCHANGE → TIENE POSICIÓN
        # ---------------------------------
        else:
            print(f"[RECOVERY] Exchange: {exch_pos}")

            state["position_qty"] = float(exch_pos["positionAmt"])
            state["position_side"] = exch_pos["side"]
            state["entry_price"] = float(exch_pos["entryPrice"])

        # 4) Guardar equity real
        state["equity"] = real_equity

        # Guardar estado final
        self._save_local_state(state)

        print("[RECOVERY] Estado alineado con Binance ✓")
        return state
