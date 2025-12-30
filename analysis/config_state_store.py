# analysis/config_state_store.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _utc_ts() -> int:
    return int(time.time())


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@dataclass
class ConfigStateStore:
    """
    Persistencia industrial:
      logs/active_config_state.json

    Estructura (por símbolo):
      state["symbols"][SYM] = {
        "active": {...},
        "pending": {...} | null,
        "prev": [ {...}, ... ],
        "last_switch_ts": int,
        "eval": {...} | null,
      }
    """
    state_path: str = "logs/active_config_state.json"
    max_prev: int = 5

    def load(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "symbols" not in data or not isinstance(data["symbols"], dict):
                        data["symbols"] = {}
                    return data
        except Exception:
            pass
        return {"version": 1, "created_at": _utc_ts(), "symbols": {}}

    def save(self, state: Dict[str, Any]) -> None:
        _ensure_dir(os.path.dirname(self.state_path) or ".")
        _atomic_write_json(self.state_path, state)

    def get_symbol_state(self, symbol: str) -> Dict[str, Any]:
        s = (symbol or "").upper().strip()
        state = self.load()
        sym_state = state["symbols"].get(s)
        if not isinstance(sym_state, dict):
            sym_state = {
                "active": None,
                "pending": None,
                "prev": [],
                "last_switch_ts": 0,
                "eval": None,
            }
            state["symbols"][s] = sym_state
            self.save(state)
        return sym_state

    def set_symbol_state(self, symbol: str, sym_state: Dict[str, Any]) -> None:
        s = (symbol or "").upper().strip()
        state = self.load()
        state["symbols"][s] = sym_state
        self.save(state)

    def push_prev(self, sym_state: Dict[str, Any], active_snapshot: Dict[str, Any]) -> None:
        prev = sym_state.get("prev")
        if not isinstance(prev, list):
            prev = []
        prev.insert(0, active_snapshot)
        prev = prev[: int(self.max_prev)]
        sym_state["prev"] = prev

    def now(self) -> int:
        return _utc_ts()

    def cooldown_passed(self, sym_state: Dict[str, Any], cooldown_sec: int) -> bool:
        last_ts = int(sym_state.get("last_switch_ts") or 0)
        return (self.now() - last_ts) >= int(cooldown_sec)

    def mark_switch(self, sym_state: Dict[str, Any]) -> None:
        sym_state["last_switch_ts"] = self.now()
