from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np


# ============================================================
# Param spec
# ============================================================

@dataclass(frozen=True)
class Choice:
    values: List[Any]

@dataclass(frozen=True)
class IntRange:
    lo: int
    hi: int
    step: int = 1

@dataclass(frozen=True)
class FloatRange:
    lo: float
    hi: float
    step: float

Spec = Any


def _frange(lo: float, hi: float, step: float) -> List[float]:
    # inclusivo con tolerancia
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    out = [lo + i * step for i in range(n)]
    # clamp por floating error
    out = [min(hi, max(lo, x)) for x in out]
    return out


def sample_params(
    space: Dict[str, Spec],
    rng: np.random.Generator,
    hard_constraints=True,
) -> Dict[str, Any]:
    """
    Muestrea 1 config del espacio.
    - Usa random sampling (anti overfit por default, evita mallas enormes).
    - Aplica restricciones duras por coherencia del sistema.
    """
    p: Dict[str, Any] = {}

    for k, spec in space.items():
        if isinstance(spec, Choice):
            p[k] = spec.values[int(rng.integers(0, len(spec.values)))]
        elif isinstance(spec, IntRange):
            vals = list(range(spec.lo, spec.hi + 1, spec.step))
            p[k] = int(vals[int(rng.integers(0, len(vals)))])
        elif isinstance(spec, FloatRange):
            vals = _frange(spec.lo, spec.hi, spec.step)
            p[k] = float(vals[int(rng.integers(0, len(vals)))])
        else:
            raise TypeError(f"Spec desconocido para {k}: {spec}")

    # ==========================
    # Hard constraints (ANTI-RUIDO)
    # ==========================
    if hard_constraints:
        # EMA coherente
        if p["ema_fast"] >= p["ema_slow"]:
            p["ema_fast"], p["ema_slow"] = min(p["ema_fast"], p["ema_slow"] - 1), p["ema_slow"]
            if p["ema_fast"] < 2:
                p["ema_fast"] = 2
                p["ema_slow"] = max(p["ema_slow"], 5)

        # RR mínimo y SL/TP coherentes
        # (ej: tp_mult >= sl_mult * rr_min)
        rr_min = p.get("rr_min", 1.2)
        if p["tp_atr_mult"] < p["sl_atr_mult"] * rr_min:
            p["tp_atr_mult"] = round(p["sl_atr_mult"] * rr_min, 2)

        # Cooldown no ridículo
        if p["cooldown_sec"] < 0:
            p["cooldown_sec"] = 0

        # Evitar configs ultra-finas (mucho overfit) en umbrales
        # (si querés, dejalo)
        # Ejemplo: delta_threshold siempre múltiplo de 5
        if "delta_threshold" in p:
            p["delta_threshold"] = int(round(p["delta_threshold"] / 5) * 5)
            p["delta_threshold"] = max(5, p["delta_threshold"])

    return p


# ============================================================
# Default space (ejemplo para HybridScalper/1m)
# Ajustalo a los nombres reales de tus params
# ============================================================

def default_param_space() -> Dict[str, Spec]:
    return {
        # Tendencia
        "ema_fast": IntRange(6, 20, 2),
        "ema_slow": IntRange(18, 80, 4),

        # Volatilidad / riesgo (ATR)
        "atr_len": IntRange(7, 21, 2),
        "sl_atr_mult": FloatRange(0.8, 2.0, 0.1),
        "tp_atr_mult": FloatRange(1.0, 4.0, 0.2),
        "rr_min": Choice([1.2, 1.3, 1.4, 1.5]),

        # Microestructura / delta (si aplica)
        "delta_threshold": IntRange(20, 160, 10),
        "delta_rolling_sec": Choice([15, 30, 60]),

        # Filtros operativos
        "cooldown_sec": Choice([0, 15, 30, 60, 120]),
        "max_trades_day": Choice([8, 12, 16, 24]),

        # Sesiones / horarios (si lo usás)
        "use_time_filter": Choice([True, False]),
        "hour_start": Choice([0, 2, 4, 6]),
        "hour_end": Choice([18, 20, 22, 24]),
    }
