import json
import os
from datetime import datetime


class StateManager:
    """
    Guarda estado continuo del bot para analytics, dashboard y recovery.
    Archivos:
        - state.json: estado global
    """

    def __init__(self, symbol):
        self.symbol = symbol.upper()

        os.makedirs("state", exist_ok=True)
        self.file = f"state/{self.symbol}_state.json"

        # Estado inicial
        self.state = {
            "timestamp": None,
            "equity": 1000.0,
            "max_equity": 1000.0,
            "drawdown": 0.0,
            "position": {
                "side": None,
                "qty": 0.0,
                "entry_price": None,
                "unrealized_pnl": 0.0
            },
            "risk": {},
            "delta_snapshot": {},
            "signals": {
                "trend_long": False,
                "trend_short": False,
                "vwap_long": False,
                "vwap_short": False,
                "delta_long": False,
                "delta_short": False,
                "score_long": 0,
                "score_short": 0
            }
        }

        self.save()

    # -------------------------------------------------------------
    def update_equity(self, equity, max_equity, drawdown):
        self.state["equity"] = equity
        self.state["max_equity"] = max_equity
        self.state["drawdown"] = drawdown
        self.save()

    def update_position(self, side, qty, entry_price, unreal):
        self.state["position"] = {
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "unrealized_pnl": unreal
        }
        self.save()

    def update_signals(self, **signals):
        for k, v in signals.items():
            self.state["signals"][k] = v
        self.save()

    def update_delta(self, snapshot):
        self.state["delta_snapshot"] = snapshot
        self.save()

    def update_risk(self, risk_dict):
        self.state["risk"] = risk_dict
        self.save()

    # -------------------------------------------------------------
    def save(self):
        self.state["timestamp"] = datetime.utcnow().isoformat()
        with open(self.file, "w") as f:
            json.dump(self.state, f, indent=2)
