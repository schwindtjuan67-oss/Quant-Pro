import asyncio
from typing import Dict

from Live.shadow_runner import ShadowEngine
from analysis.portfolio_state import PortfolioState
from analysis.allocator_bridge import AllocatorBridge
from analysis.allocator import AllocatorConfig


class ShadowRunnerMulti:
    """
    Runner industrial multi-symbol (FASE 8)
    """

    def __init__(self, symbols: list[str], interval: str, config: dict):
        self.portfolio = PortfolioState(window=100)

        alloc_cfg = AllocatorConfig(
            total_risk_usdt=config.get("total_risk_usdt", 10.0),
            max_positions=config.get("max_positions", 3),
        )
        self.allocator = AllocatorBridge(self.portfolio, alloc_cfg)

        self.engines: Dict[str, ShadowEngine] = {}
        for s in symbols:
            self.engines[s] = ShadowEngine(
                symbol=s,
                interval=interval,
                config=config,
                portfolio=self.portfolio,
                allocator=self.allocator,
            )

    async def start(self):
        await asyncio.gather(
            *[engine.start() for engine in self.engines.values()]
        )
