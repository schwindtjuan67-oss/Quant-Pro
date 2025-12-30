# Live/strategy_live_hybrid.py
# ===================================================================
# Estrategia LIVE híbrida usada por ShadowEngine
# Combina:
#   - TrendRiskModule  (filtro de tendencia profesional)
#   - VWAPModule       (filtro de desvío institucional)
#   - DeltaRouter      (filtro de flujo de órdenes en tiempo real)
# ===================================================================

import time
from typing import Dict, Optional


class StrategyLiveHybrid:
    """
    Estrategia LIVE que combina:
      • Filtro de tendencia (TrendRiskModule)
      • Filtro VWAP
      • Filtro Delta (DeltaRouter)
      • Gestión de cooldown, estado y reversión

    Se ejecuta desde:
        ShadowEngine.on_new_kline()
    """

    def __init__(self, engine, config):
        """
        engine  → instancia del ShadowEngine
        config  → config_shadow.json
        """
        self.engine = engine
        self.config = config

        # ==========================================================
        # 1) DATA INICIAL (FIX IMPORTANTE)
        # ==========================================================
        # ShadowEngine ya descargó velas históricas antes de iniciar el loop.
        self.data = engine.data  # <--- FIX CLAVE

        # ==========================================================
        # 2) CARGA DE MÓDULOS PRO
        # ==========================================================
        # --- Trend (requiere data) ---
        from .trend_risk_tools import TrendRiskModule
        self.trend_module = TrendRiskModule(self.data)

        # --- VWAP (no requiere data inicial) ---
        from .vwap_tools import VWAPModule
        self.vwap_module = VWAPModule()

        # --- DeltaRouter (inyectado desde el engine) ---
        self.delta_router = engine.delta_router

        # ==========================================================
        # 3) PARÁMETROS
        # ==========================================================
        self.cooldown_sec = config.get("cooldown_sec", 8)
        self.max_position_time = config.get("max_position_time_sec", 600)

        # ==========================================================
        # 4) ESTADO DE LA ESTRATEGIA
        # ==========================================================
        self.position: Optional[str] = None     # "long", "short" o None
        self.entry_price: Optional[float] = None
        self.entry_time: float = 0
        self.cooldown_until: float = 0
        self.last_signal: Optional[str] = None

    # ===================================================================
    # AUXILIARES
    # ===================================================================

    def _in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    def _start_cooldown(self):
        self.cooldown_until = time.time() + self.cooldown_sec

    # ===================================================================
    # LÓGICA PRINCIPAL
    # ===================================================================

    def on_new_kline(self, candle: Dict):
        """
        Llamada por ShadowEngine cada vez que llega un nuevo kline (1m).

        candle = {
            "open": float,
            "high": float,
            "low":  float,
            "close": float,
            "volume": float,
            "ts": int
        }
        """
        price = float(candle["close"])

        # ==========================================================
        # 1) CHEQUEAR COOLDOWN
        # ==========================================================
        if self._in_cooldown():
            return None

        # ==========================================================
        # 2) OBTENER SNAPSHOT DELTA
        # ==========================================================
        snap = self.delta_router.get_snapshot(self.engine.symbol)

        # ==========================================================
        # 3) FILTROS DE MERCADO
        # ==========================================================
        trend_sig = self.trend_module.get_signal(price)
        vwap_sig = self.vwap_module.get_signal(price)

        # ==========================================================
        # 4) DECISIONES
        # ==========================================================
        long_ok = (
            trend_sig == "long" and
            vwap_sig == "long" and
            self.delta_router.allows_long(self.engine.symbol, snap)
        )

        short_ok = (
            trend_sig == "short" and
            vwap_sig == "short" and
            self.delta_router.allows_short(self.engine.symbol, snap)
        )

        # ==========================================================
        # 5) GESTIÓN DE POSICIONES
        # ==========================================================

        # ----------------------
        # Si no estamos en posición
        # ----------------------
        if self.position is None:

            if long_ok:
                self.position = "long"
                self.entry_price = price
                self.entry_time = time.time()
                self.last_signal = "long"
                self._start_cooldown()
                return {"signal": "long", "price": price}

            if short_ok:
                self.position = "short"
                self.entry_price = price
                self.entry_time = time.time()
                self.last_signal = "short"
                self._start_cooldown()
                return {"signal": "short", "price": price}

            return None

        # ----------------------
        # Si estamos en LONG
        # ----------------------
        if self.position == "long":

            # Si el filtro delta ahora PROHÍBE longs → cerrar
            if not self.delta_router.allows_long(self.engine.symbol, snap):
                self.position = None
                self._start_cooldown()
                return {"signal": "close", "price": price}

            # Si se invierte tendencia → cerrar
            if trend_sig == "short":
                self.position = None
                self._start_cooldown()
                return {"signal": "close", "price": price}

            return None

        # ----------------------
        # Si estamos en SHORT
        # ----------------------
        if self.position == "short":

            if not self.delta_router.allows_short(self.engine.symbol, snap):
                self.position = None
                self._start_cooldown()
                return {"signal": "close", "price": price}

            if trend_sig == "long":
                self.position = None
                self._start_cooldown()
                return {"signal": "close", "price": price}

            return None

        return None
