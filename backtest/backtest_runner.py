from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import os

os.environ["QS_BACKTEST"] = "1"
RUN_MODE = os.getenv("RUN_MODE", "LIVE").upper().strip()
PIPELINE_VERBOSE_DIAGNOSTICS = os.getenv("PIPELINE_VERBOSE_DIAGNOSTICS", "0").strip().lower() in ("1", "true", "yes")
PIPELINE_DISABLE_GPU = RUN_MODE == "PIPELINE" and os.getenv("PIPELINE_DISABLE_GPU", "0").strip().lower() in ("1", "true", "yes")

def _bt_print(msg: str) -> None:
    # En PIPELINE default = silencioso. Opt-in con PIPELINE_VERBOSE_DIAGNOSTICS=1
    if RUN_MODE != "PIPELINE" or PIPELINE_VERBOSE_DIAGNOSTICS:
        print(msg)

# ------------------------------------------------------------
# GPU feeder (opcional)
# ------------------------------------------------------------
try:
    from infra.gpu_candle_feeder import GPUCandleFeeder
    _HAS_GPU_FEEDER = True
except Exception:
    GPUCandleFeeder = None
    _HAS_GPU_FEEDER = False


@dataclass
class BacktestResult:
    equity_r: float
    trades: int
    notes: str = ""


class BacktestRunner:
    """
    Backtest runner REAL:
    - Ejecuta HybridAdapterShadow offline
    - Captura trades desde TradeLogger
    - Equity en R
    - GPU acceleration vía GPUCandleFeeder (opcional)
    """

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        candles: List[Dict[str, Any]],
        symbol: str,
        interval: str = "1m",
        warmup_candles: int = 500,
        use_gpu: bool = True,
        gpu_batch_size: int = 256,
    ):
        self.config = config or {}
        self.candles = candles or []
        self.symbol = symbol.upper()
        self.interval = interval
        self.warmup = int(warmup_candles)

        self.use_gpu = bool(use_gpu and _HAS_GPU_FEEDER and not PIPELINE_DISABLE_GPU)
        self.gpu_batch_size = int(gpu_batch_size)

    # ------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        from Live.order_manager_shadow import ShadowOrderManager
        from Live.risk_manager import RiskManager
        from Live.logger_pro import TradeLogger
        from Live.hybrid_scalper_pro_adapter import HybridAdapterShadow

        # ---------------- sanity ----------------
        if len(self.candles) == 0:
            _bt_print("[WARN] BacktestRunner: candles=0")
            return {"equity_r": 0.0, "trades": 0, "notes": "candles=0"}

        if len(self.candles) <= self.warmup:
            _bt_print("[WARN] BacktestRunner: candles <= warmup")
            return {"equity_r": 0.0, "trades": 0, "notes": "candles<=warmup"}

        # ---------------- Delta (opcional) ----------------
        try:
            from Live.delta_live import DeltaLive
            from Live.delta_router import DeltaRouter
            delta_ok = True
        except Exception:
            delta_ok = False

        # ---------------- Logger ----------------
        class BacktestLogger(TradeLogger):
            def __init__(self, symbol: str):
                super().__init__(symbol)
                self.trades = []
                self.trades_list: List[Dict[str, Any]] = []
                self.equity_r = 0.0

            def log_trade(self, **k):
                super().log_trade(**k)
                if k.get("type") == "EXIT":
                    pnl = float(k.get("pnl", 0.0))
                    risk = float(k.get("risk_usdt", 0.0))
                    if risk > 0:
                        r = pnl / risk
                        self.equity_r += r
                        self.trades.append(r)
                        trade: Dict[str, Any] = {
                            "type": "EXIT",
                            "pnl_r": float(r),
                            "pnl": float(pnl),
                            "risk_usdt": float(risk),
                        }
                        for key in ("timestamp_ms", "ts_ms", "exit_ts_ms", "close_ts_ms"):
                            if key in k and k[key] not in (None, ""):
                                trade["timestamp_ms"] = int(float(k[key]))
                                break
                        if "timestamp_ms" not in trade:
                            for key in ("timestamp", "ts", "exit_ts", "close_ts", "open_time"):
                                if key in k and k[key] not in (None, ""):
                                    v = int(float(k[key]))
                                    trade["timestamp_ms"] = v if v >= 10_000_000_000 else v * 1000
                                    break
                        for extra in ("holding_time_sec", "symbol"):
                            if extra in k and k[extra] not in (None, ""):
                                trade[extra] = k[extra]
                        self.trades_list.append(trade)

        # ---------------- Offline Engine ----------------
        class OfflineEngine:
            def __init__(self, outer: BacktestRunner):
                self.symbol = outer.symbol
                self.interval = outer.interval
                self.config = outer.config

                self.is_backtest = True
                self.logger = BacktestLogger(self.symbol)

                self.order_manager = ShadowOrderManager(self.symbol, self.config)
                self.router = self.order_manager

                self.risk_manager = RiskManager(
                    max_loss_pct=self.config.get("max_loss", 0.03),
                    max_dd_pct=self.config.get("max_dd", 0.04),
                    max_trades=self.config.get("max_trades", 12),
                    starting_equity=1000.0,
                )

                if delta_ok:
                    self.delta_live = DeltaLive(self.symbol)
                    self.delta_router = DeltaRouter(self.delta_live)
                else:
                    self.delta_live = None
                    self.delta_router = None

                self.strategy = HybridAdapterShadow(
                    engine=self,
                    allocator=None,
                    config_selector=None,
                    hotswap_enabled=False,
                    verbose=False,
                )

            def on_new_candle(self, candle: Dict[str, Any]):
                self.strategy.on_new_candle(candle)

        engine = OfflineEngine(self)

        # ---------------- Warmup ----------------
        for i in range(self.warmup):
            c = self.candles[i]
            engine.on_new_candle(c)

        live_candles = self.candles[self.warmup :]

        # ---------------- MAIN LOOP ----------------
        # (extra safety, aunque ya está contemplado en __init__)
        if self.use_gpu and PIPELINE_DISABLE_GPU:
            self.use_gpu = False

        if self.use_gpu:
            _bt_print("[BACKTEST] GPUCandleFeeder ENABLED")
            feeder = GPUCandleFeeder(
                batch_size=self.gpu_batch_size,
                use_gpu=True,
                streams=2,
            )
            feeder.run(
                live_candles,
                on_bar=engine.on_new_candle,
            )
        else:
            _bt_print("[BACKTEST] CPU loop")
            for c in live_candles:
                engine.on_new_candle(c)

        # ---------------- Diagnostics ----------------
        adapter = getattr(engine, "strategy", None)
        core = getattr(adapter, "hybrid", None) if adapter else None
        diag = getattr(core, "_entry_diag", None) if core else None

        if isinstance(diag, dict) and diag:
            if not (RUN_MODE == "PIPELINE" and not PIPELINE_VERBOSE_DIAGNOSTICS):
                print("\nENTRY DIAGNOSTICS")
                for k, v in diag.items():
                    print(f"{k:12}: {v}")

        # ---------------- Result ----------------
        trades_list = list(engine.logger.trades_list)
        trades = len(trades_list)
        equity_r = engine.logger.equity_r

        return {
            "equity_r": round(float(equity_r), 6),
            "trades": int(trades),
            "trades_list": trades_list,
            "notes": "BacktestRunner GPU OK" if self.use_gpu else "BacktestRunner CPU OK",
        }
