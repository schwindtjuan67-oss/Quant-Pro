# Live/delta_router.py
# ================================================================
#   Adaptador entre DeltaLive y las estrategias LIVE/SHADOW
#   - Expone una interfaz simple:
#       * get_snapshot(symbol) -> dict
#       * allows_long(symbol, snap) -> bool
#       * allows_short(symbol, snap) -> bool
#   - Evita bucles de importación (NO importa Hybrid, ni ws, etc.)
# ================================================================

from __future__ import annotations

from typing import Any, Dict, Optional

from .delta_live import DeltaLive, DeltaSnapshot


class DeltaRouter:
    """
    Enrutador/filtro de DELTA para:

      - HybridScalperPRO (Live/hybrid_scalper_pro.py)
      - StrategyLiveHybrid (Live/strategy_live_hybrid.py)
      - ShadowEngine (Live/shadow_runner.py)

    Función principal:
      - Obtener un snapshot de DeltaLive.
      - Decidir si DELTA permite o no operar en LONG / SHORT.

    NOTAS:
      - Si no hay suficientes trades, el filtro NO bloquea nada (degrada a "off").
      - Si no hay delta_live disponible, devuelve snapshot neutro y no bloquea.
    """

    def __init__(
        self,
        source: Optional[Any] = None,
        z_entry: float = 0.5,
        min_trades: int = 10,
    ) -> None:
        """
        `source` puede ser:

          - DeltaLive: caso recomendado (Shadow / Live moderno).
          - Cualquier otro objeto (p.ej. EventBus de versiones viejas):
            en ese caso, el router funciona en modo "pasivo" y NO bloquea.

        `z_entry`   → umbral de intensidad para aceptar delta (tipo z-score).
        `min_trades`→ mínimo de trades en ventana para considerar delta.
        """
        self.z_entry = float(z_entry)
        self.min_trades = int(min_trades)

        self.delta_live: Optional[DeltaLive] = None

        if isinstance(source, DeltaLive):
            self.delta_live = source

        # Para posibles usos futuros (ej: actualización manual por EventBus)
        self._external_snapshots: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------
    # Gestión de snapshots
    # ------------------------------------------------------------
    def set_external_snapshot(self, symbol: str, snap: Dict[str, Any]) -> None:
        """
        Permite, en caso de no usar DeltaLive, inyectar snapshots desde fuera.
        Hoy no lo usa nadie, pero deja el camino abierto para Run Live custom.
        """
        self._external_snapshots[symbol.upper()] = dict(snap)

    def _default_snapshot(self) -> Dict[str, Any]:
        return {
            "last_ts": None,
            "delta_candle": 0.0,
            "delta_candle_prev": 0.0,
            "delta_rolling_15s": 0.0,
            "delta_rolling_60s": 0.0,
            "trades_count_window": 0,
        }

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Devuelve un dict con el snapshot de delta para la estrategia.

        Prioridad:
          1) DeltaLive (si se inicializó con DeltaLive).
          2) Snapshot externo inyectado (set_external_snapshot).
          3) Snapshot neutro por defecto.
        """
        # 1) DeltaLive
        if self.delta_live is not None:
            snap_obj = self.delta_live.get_snapshot()
            if isinstance(snap_obj, DeltaSnapshot):
                return snap_obj.as_dict()
            if isinstance(snap_obj, dict):
                return snap_obj

            # Fallback muy defensivo: convertir atributos a dict
            return {
                "last_ts": getattr(snap_obj, "last_ts", None),
                "delta_candle": float(getattr(snap_obj, "delta_candle", 0.0)),
                "delta_candle_prev": float(getattr(snap_obj, "delta_candle_prev", 0.0)),
                "delta_rolling_15s": float(getattr(snap_obj, "delta_rolling_15s", 0.0)),
                "delta_rolling_60s": float(getattr(snap_obj, "delta_rolling_60s", 0.0)),
                "trades_count_window": int(
                    getattr(snap_obj, "trades_count_window", 0)
                ),
            }

        # 2) Snapshot externo (si lo hubiera)
        key = symbol.upper()
        if key in self._external_snapshots:
            return dict(self._external_snapshots[key])

        # 3) Default neutro
        return self._default_snapshot()

    # ------------------------------------------------------------
    # Lógica del filtro de Delta
    # ------------------------------------------------------------
    def _compute_z(self, snap: Dict[str, Any]) -> float:
        """
        Calcula un pseudo z-score:
            z = delta_rolling_15s / (|delta_rolling_60s| + eps)

        No es estadístico "puro", pero sirve como medida de intensidad,
        evitando divisiones por cero.
        """
        num = float(snap.get("delta_rolling_15s", 0.0))
        denom = abs(float(snap.get("delta_rolling_60s", 0.0))) + 1e-6
        return num / denom

    def _enough_data(self, snap: Dict[str, Any]) -> bool:
        trades = int(snap.get("trades_count_window", 0))
        return trades >= self.min_trades

    def allows_long(self, symbol: str, snap: Dict[str, Any]) -> bool:
        """
        Devuelve True si DELTA permite entrar/seguir en LONG.

        Política:
          - Si no hay suficientes datos (pocos trades) → no bloqueamos (True).
          - Si hay datos, exigimos que el z-score sea >= z_entry.
        """
        if not self._enough_data(snap):
            return True  # degradar a "sin filtro" cuando no hay info fiable

        z = self._compute_z(snap)
        return z >= self.z_entry

    def allows_short(self, symbol: str, snap: Dict[str, Any]) -> bool:
        """
        Devuelve True si DELTA permite entrar/seguir en SHORT.

        Política:
          - Si no hay suficientes datos (pocos trades) → no bloqueamos (True).
          - Si hay datos, exigimos que el z-score sea <= -z_entry.
        """
        if not self._enough_data(snap):
            return True

        z = self._compute_z(snap)
        return z <= -self.z_entry

    # ------------------------------------------------------------
    # Hook opcional para ws_futures_1m (no usado hoy, pero neutro)
    # ------------------------------------------------------------
    def on_new_candle(self, candle: Dict[str, Any]) -> None:
        """
        Método dummy para que, si en algún flujo viejo se le pasa un candle
        desde ws_futures_1m, no rompa. Hoy no usamos velas aquí; todo el
        delta viene del stream de trades (DeltaLive).
        """
        return
