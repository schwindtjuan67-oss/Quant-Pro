# Live/regime_controller.py
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


@dataclass
class MetaDecision:
    allow: bool
    vol_mult: float
    reason: str


class RegimeController:
    """
    Consume flags horarios + (opcional) status kill/promote por régimen.
    Produce:
      - allow(regime): bool
      - vol_multiplier(regime): float   (exposure scaling)
    """

    def __init__(
        self,
        flags_path: Optional[str] = None,
        rules_path: Optional[str] = None,
        default_tz: str = "America/Argentina/Buenos_Aires",
        min_trades_hour: int = 5,
    ):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.flags_path = flags_path or os.path.join(root, "logs", "SOLUSDT_hourly_regime_flags.json")
        self.rules_path = rules_path or os.path.join(root, "analysis", "regime_rules.yaml")

        self.default_tz = default_tz
        self.min_trades_hour = int(min_trades_hour)

        self._cache_flags: Dict[str, Any] = {}
        self._cache_mtime: float = 0.0

    # ---------------- time ----------------
    def _now_local(self) -> datetime:
        tz_name = self._cache_flags.get("timezone") or self.default_tz
        if ZoneInfo is None:
            return datetime.now()
        try:
            return datetime.now(ZoneInfo(tz_name))
        except Exception:
            return datetime.now()

    def current_hour(self) -> int:
        return int(self._now_local().hour)

    # ---------------- flags io ----------------
    def _load_flags(self) -> Dict[str, Any]:
        try:
            mtime = os.path.getmtime(self.flags_path)
        except Exception:
            return {}

        if self._cache_flags and mtime == self._cache_mtime:
            return self._cache_flags

        try:
            with open(self.flags_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        self._cache_flags = data if isinstance(data, dict) else {}
        self._cache_mtime = mtime
        return self._cache_flags

    # ---------------- schema adapters ----------------
    def _get_regime_node(self, flags: Dict[str, Any], regime: str) -> Dict[str, Any]:
        # schema flexible:
        # flags["regimes"][regime] or flags[regime] or {}
        regimes = flags.get("regimes")
        if isinstance(regimes, dict) and isinstance(regimes.get(regime), dict):
            return regimes.get(regime, {})
        if isinstance(flags.get(regime), dict):
            return flags.get(regime, {})
        return {}

    def _extract_hours_list(self, node: Dict[str, Any], key: str) -> Tuple[int, ...]:
        # key could be "kill_hours" / "risk_hours" / "promote_hours"
        v = node.get(key, [])
        if isinstance(v, dict):  # sometimes {"hours":[...]}
            v = v.get("hours", [])
        if not isinstance(v, (list, tuple)):
            return tuple()
        out = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return tuple(sorted(set([h for h in out if 0 <= h <= 23])))

    def _hourly_stats(self, node: Dict[str, Any], hour: int) -> Dict[str, Any]:
        # accept node["hourly"][str(hour)] or node["hours"][str(hour)]
        hourly = node.get("hourly") or node.get("hours") or {}
        if not isinstance(hourly, dict):
            return {}
        key = str(int(hour))
        val = hourly.get(key, {})
        return val if isinstance(val, dict) else {}

    # ---------------- public API ----------------
    def decide(self, regime: str) -> MetaDecision:
        flags = self._load_flags()
        hour = self.current_hour()

        node = self._get_regime_node(flags, regime)

        kill_hours = self._extract_hours_list(node, "kill_hours")
        risk_hours = self._extract_hours_list(node, "risk_hours")
        promote_hours = self._extract_hours_list(node, "promote_hours")

        # defaults
        allow = True
        vol_mult = 1.0
        reason = "DEFAULT"

        # 1) hard kill hour => no trade
        if hour in kill_hours:
            return MetaDecision(False, 0.0, f"KILLED_HOUR_{hour:02d}")

        # 2) "horas en riesgo" => penalizar
        if hour in risk_hours:
            vol_mult *= 0.40
            reason = f"RISK_HOUR_{hour:02d}"

        # 3) promote hour => pequeño boost (si no está en risk)
        if hour in promote_hours and hour not in risk_hours:
            vol_mult *= 1.15
            reason = f"PROMOTE_HOUR_{hour:02d}"

        # 4) edge-based scaling (usa hourly stats si existen)
        st = self._hourly_stats(node, hour)
        trades_h = _safe_int(st.get("trades"), 0)
        sum_r_h = _safe_float(st.get("sum_r"), 0.0)
        avg_r_h = _safe_float(st.get("avg_r"), 0.0)

        # si no hay datos confiables, no escalar demasiado
        if trades_h >= self.min_trades_hour:
            # si hora viene mala => recortar
            if sum_r_h < 0:
                vol_mult *= 0.60
                reason = f"EDGE_NEG_{hour:02d}"
            # si hora viene muy buena => subir suave
            elif sum_r_h > 0 and avg_r_h > 0:
                vol_mult *= 1.10
                reason = f"EDGE_POS_{hour:02d}"

        # clamp final (evitar locuras)
        vol_mult = max(0.0, min(float(vol_mult), 1.50))

        return MetaDecision(allow, vol_mult, reason)

    def allow(self, regime: str) -> bool:
        return self.decide(regime).allow

    def vol_multiplier(self, regime: str) -> float:
        return self.decide(regime).vol_mult
