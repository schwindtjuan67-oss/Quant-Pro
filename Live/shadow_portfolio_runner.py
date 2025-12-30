# Live/shadow_portfolio_runner.py
import asyncio
from typing import Dict, Any, List, Optional

from analysis.allocator import AllocatorConfig
from analysis.allocator_wrapper import AllocatorWrapper
from analysis.correlation_guard import CorrelationGuard, CorrGuardConfig

from Live.shadow_runner import ShadowEngine


class ShadowPortfolio:
    """
    FASE 8 — Portfolio shadow multi-símbolo.

    - 1 Allocator compartido
    - 1 CorrGuard compartido
    - N ShadowEngine (uno por símbolo)

    Nota:
    - ShadowEngine actual maneja 1 símbolo.
    - Acá lo orquestamos en paralelo.
    """

    def __init__(self, symbols: List[str], interval: str, config: Dict[str, Any]):
        self.symbols = [s.upper() for s in symbols]
        self.interval = interval
        self.config = config

        # Allocator global
        self.allocator = AllocatorWrapper(cfg=AllocatorConfig(
            total_risk_usdt=config.get("allocator_total_risk", 10.0),
            base_risk_usdt=config.get("allocator_base_risk", 5.0),
            min_risk_usdt=config.get("allocator_min_risk", 0.5),
            max_risk_usdt=config.get("allocator_max_risk", 20.0),
            max_positions=config.get("allocator_max_positions", 3),
            corr_threshold=config.get("corr_threshold", 0.85),
            hard_block_if_corr_ge=config.get("hard_block_if_corr_ge", 0.98),
        ))

        # Corr guard global
        self.corr_guard = CorrelationGuard(cfg=CorrGuardConfig(
            window=config.get("corr_window", 120),
            min_points=config.get("corr_min_points", 30),
            method=config.get("corr_method", "pearson"),
        ))

        # Engines por símbolo
        self.engines: List[ShadowEngine] = []
        for sym in self.symbols:
            eng = ShadowEngine(symbol=sym, interval=interval, config=config)

            # inyectar allocator + corr_guard si están expuestos
            # (tu HybridAdapterShadow acepta allocator; ShadowEngine ya lo crea en F7.3)
            # Para F8, lo reemplazamos por el global:
            try:
                eng.allocator = self.allocator
                eng.strategy.allocator = self.allocator
            except Exception:
                pass

            # corr guard accesible desde engine (para luego alimentar con trades)
            eng.corr_guard = self.corr_guard

            self.engines.append(eng)

    async def start(self):
        tasks = [asyncio.create_task(e.start()) for e in self.engines]
        await asyncio.gather(*tasks)
