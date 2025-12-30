import asyncio
import traceback
import subprocess
import sys
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import pytz

from utils.runtime_state import safe_write_json

from Live.ws_futures_1m import FuturesWS_1m
from Live.kline_fetcher import KlineFetcher
from Live.order_manager_shadow import ShadowOrderManager
from Live.delta_live import DeltaLive
from Live.delta_router import DeltaRouter
from Live.risk_manager import RiskManager
from Live.logger_pro import TradeLogger
from Live.hybrid_scalper_pro_adapter import HybridAdapterShadow

# Allocator / Portfolio (industrial)
from analysis.allocator import AllocatorConfig
from analysis.allocator_bridge import AllocatorBridge
from analysis.portfolio_state import PortfolioState

# FASE 10.1 selector
try:
    from analysis.config_selector import ConfigSelector
except Exception:
    ConfigSelector = None


# ============================================================
# Utils
# ============================================================
def _now_ts_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _risk_dd_pct_from_risk_manager(rm: Any) -> float:
    try:
        equity = _safe_float(getattr(rm, "equity", None), 0.0)
        peak = _safe_float(getattr(rm, "equity_peak", None), 0.0)
        if peak > 0:
            return max(0.0, (peak - equity) / peak)
    except Exception:
        pass
    return 0.0


def _portfolio_equity_r_total(portfolio: Any) -> float:
    for name in ("equity_r_total", "get_equity_r_total", "total_r", "sum_r"):
        try:
            fn = getattr(portfolio, name, None)
            if callable(fn):
                return _safe_float(fn(), 0.0)
        except Exception:
            pass
    return 0.0


# ============================================================
# ⏰ FILTRO HORARIO (EDGE)
# ============================================================
def _hour_allowed(candle: dict, *, config: dict, regime: str) -> bool:
    hf = (config or {}).get("hour_filter") or {}
    if not hf.get("enabled", False):
        return True

    apply_to = hf.get("apply_to", {}) or {}
    if not apply_to.get(str(regime).upper(), False):
        return True

    tz_name = hf.get("timezone", "UTC")
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.UTC

    ts = candle.get("timestamp")
    if ts is None:
        return True

    ts_sec = ts / 1000.0 if ts > 1e12 else ts
    hour = datetime.fromtimestamp(ts_sec, tz=tz).hour

    killed = set(hf.get("killed_hours", []))
    return hour not in killed


# ============================================================
# Logger wrapper → alimenta PortfolioState en EXIT
# ============================================================
class PortfolioAwareLogger:
    def __init__(self, base_logger: TradeLogger, portfolio: PortfolioState, symbol: str):
        self._base = base_logger
        self._portfolio = portfolio
        self._symbol = symbol.upper().strip()

    def __getattr__(self, name):
        return getattr(self._base, name)

    def log_trade(self, **k):
        self._base.log_trade(**k)
        if k.get("type") != "EXIT":
            return
        try:
            pnl = float(k.get("pnl"))
            risk = float(k.get("risk_usdt"))
            if risk > 0:
                self._portfolio.on_trade_close(self._symbol, pnl / risk)
        except Exception:
            pass


# ============================================================
# ShadowEngine (por símbolo)
# ============================================================
class ShadowEngine:
    def __init__(
        self,
        symbol: str,
        interval: str,
        config: dict,
        *,
        allocator: AllocatorBridge,
        portfolio: PortfolioState,
        config_selector: Optional[Any] = None,
    ):
        self.symbol = symbol.upper()
        self.interval = interval
        self.config = config
        self.event_bus = None

        self.last_price: float = 0.0
        self.last_ts: int = 0

        print(
            f"[RISK] Día iniciado ({self.symbol}) – equity_inicial=1000.00 "
            f"max_loss={config.get('max_loss', 0.03)} "
            f"max_dd={config.get('max_dd', 0.04)} "
            f"max_trades={config.get('max_trades', 12)}"
        )

        base_logger = TradeLogger(self.symbol)
        self.logger = PortfolioAwareLogger(base_logger, portfolio, self.symbol)

        self.kline_fetcher = KlineFetcher(symbol, interval)
        self.data = self.kline_fetcher.load_initial_data()

        self.delta_live = DeltaLive(symbol)
        self.delta_router = DeltaRouter(self.delta_live)

        self.order_manager = ShadowOrderManager(symbol, config)
        self.router = self.order_manager

        self.risk_manager = RiskManager(
            max_loss_pct=config.get("max_loss", 0.03),
            max_dd_pct=config.get("max_dd", 0.04),
            max_trades=config.get("max_trades", 12),
            starting_equity=1000.0,
        )

        self.allocator = allocator

        selector_cfg = (config or {}).get("selector", {}) or {}
        self.strategy = HybridAdapterShadow(
            engine=self,
            allocator=self.allocator,
            config_selector=config_selector,
            hotswap_enabled=bool(selector_cfg.get("hotswap_enabled", True)),
            hotswap_interval_sec=int(selector_cfg.get("hotswap_interval_sec", 60)),
            hotswap_only_uppercase=bool(selector_cfg.get("hotswap_only_uppercase", True)),
            verbose=bool(selector_cfg.get("verbose", True)),
        )

        self.ws = FuturesWS_1m(
            symbol=symbol,
            on_candle=lambda c: asyncio.create_task(self.on_new_kline(c)),
        )

        print(f"[SHADOW] ShadowEngine inicializado ({self.symbol})")

    async def on_new_kline(self, candle: dict):
        try:
            self.last_price = _safe_float(candle.get("close"), self.last_price)
            ts = candle.get("timestamp")
            self.last_ts = _safe_int(ts, self.last_ts) if ts is not None else self.last_ts

            # -------- FILTRO HORARIO --------
            try:
                regime = (
                    candle.get("regime")
                    or getattr(self.strategy.hybrid, "last_regime", None)
                    or "NOISE"
                )
                if not _hour_allowed(candle, config=self.config, regime=regime):
                    return
            except Exception:
                pass

            self.data = self.kline_fetcher.append_new_candle(self.data, candle)
            self.strategy.on_new_candle(candle)

        except Exception as e:
            print(f"[ERROR on_new_kline {self.symbol}]", e)
            traceback.print_exc()

    def runtime_snapshot(self, portfolio: PortfolioState) -> dict:
        hybrid = getattr(self.strategy, "hybrid", None)

        ps = getattr(hybrid, "position_side", None) if hybrid else None
        qty = _safe_float(getattr(hybrid, "position_qty", 0.0), 0.0) if hybrid else 0.0
        ep = _safe_float(getattr(hybrid, "entry_price", 0.0), 0.0) if hybrid else 0.0
        sl = _safe_float(getattr(hybrid, "current_sl", 0.0), 0.0) if hybrid else 0.0
        reg = getattr(hybrid, "position_regime", None) if hybrid else None

        unreal = 0.0
        if ps in ("LONG", "SHORT") and qty > 0 and self.last_price > 0:
            unreal = (self.last_price - ep) * qty * (1.0 if ps == "LONG" else -1.0)

        try:
            risk_state = self.risk_manager.as_dict()
        except Exception:
            risk_state = {}

        return {
            "symbol": self.symbol,
            "ts_ms": self.last_ts or _now_ts_ms(),
            "last_price": self.last_price,
            "position": {
                "side": ps or "FLAT",
                "qty": qty,
                "entry_price": ep,
                "sl": sl,
                "regime": (str(reg).upper() if reg else "UNKNOWN"),
                "unrealized_pnl": unreal,
            },
            "metrics": {
                "equity_r": _safe_float(_portfolio_equity_r_total(portfolio), 0.0),
                "dd_day": _safe_float(_risk_dd_pct_from_risk_manager(self.risk_manager), 0.0),
            },
            "risk_state": risk_state,
        }

    async def start(self):
        self.delta_live.start()
        self.ws.start()
        while True:
            await asyncio.sleep(1)


# ============================================================
# Autotuning Worker
# ============================================================
class AutoTuningWorker:
    def __init__(
        self,
        *,
        logs_dir: str = "logs",
        interval_sec: int = 1800,
        min_trades: int = 12,
        top_k: int = 20,
        symbol: Optional[str] = None,
    ):
        self.logs_dir = logs_dir
        self.interval_sec = interval_sec
        self.min_trades = min_trades
        self.top_k = top_k
        self.symbol = symbol

    async def run(self):
        print("[AUTOTUNE] Worker iniciado")
        while True:
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "analysis.autotuning_fase8_1",
                    "--logs_dir",
                    self.logs_dir,
                    "--min_trades",
                    str(self.min_trades),
                    "--top_k",
                    str(self.top_k),
                ]
                if self.symbol:
                    cmd.extend(["--symbol", self.symbol])
                subprocess.run(cmd, check=False)
            except Exception as e:
                print("[AUTOTUNE] ERROR:", e)

            await asyncio.sleep(self.interval_sec)


# ============================================================
# RuntimeState Writer (FIX DEFINITIVO)
# ============================================================
class RuntimeStateWriter:
    def __init__(self, *, path: str, interval_sec: float, runner_ref: "ShadowRunner"):
        self.path = path
        self.interval_sec = interval_sec
        self.runner_ref = runner_ref

    async def run(self):
        print(f"[RUNTIME] Writer iniciado path={self.path}")
        while True:
            try:
                r = self.runner_ref
                payload = {
                    "generated_at_ms": _now_ts_ms(),
                    "mode": "shadow",
                    "symbols": list(r.symbols),
                    "metrics": {
                        "equity_r": _safe_float(_portfolio_equity_r_total(r.portfolio), 0.0),
                        "dd_day": _safe_float(
                            _risk_dd_pct_from_risk_manager(r.engines[0].risk_manager), 0.0
                        )
                        if r.engines else 0.0,
                    },
                    "by_symbol": {
                        e.symbol: e.runtime_snapshot(r.portfolio)
                        for e in r.engines
                    },
                }
                safe_write_json(self.path, payload)
            except Exception as e:
                print("[RUNTIME] ERROR write:", e)

            await asyncio.sleep(self.interval_sec)


# ============================================================
# ShadowRunner
# ============================================================
class ShadowRunner:
    def __init__(
        self,
        *,
        interval: str = "1m",
        config: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
    ):
        self.interval = interval
        self.config = config or {}

        self.symbols = (
            [str(s).upper().strip() for s in self.config.get("symbols", [])]
            or [str(symbol or self.config.get("symbol") or "SOLUSDT").upper().strip()]
        )

        self.portfolio = PortfolioState(
            window=int(self.config.get("corr_window", 100)),
            min_samples_per_symbol=int(self.config.get("corr_min_samples", 5)),
            min_symbols=int(self.config.get("corr_min_symbols", 2)),
        )

        allocator_cfg = AllocatorConfig(
            total_risk_usdt=float(self.config.get("allocator_total_risk", 10.0)),
            base_risk_usdt=float(self.config.get("allocator_base_risk", 5.0)),
            max_positions=int(self.config.get("allocator_max_positions", len(self.symbols))),
            corr_threshold=float(self.config.get("allocator_corr_threshold", 0.85)),
            hard_block_if_corr_ge=float(self.config.get("allocator_hard_block", 0.98)),
        )

        self.allocator = AllocatorBridge(
            cfg=allocator_cfg,
            portfolio=self.portfolio,
            min_vol_target=float(self.config.get("allocator_min_vol_target", 0.0)),
            max_vol_target=float(self.config.get("allocator_max_vol_target", 1.0)),
            apply_risk_mult_to_budget=bool(self.config.get("allocator_apply_risk_mult_to_budget", True)),
            use_portfolio_corr_by_default=True,
        )

        selector_cfg = (self.config or {}).get("selector", {}) or {}
        self.config_selector = (
            ConfigSelector(
                top_k_path=selector_cfg.get("top_k_path", "logs/top_k.json"),
                min_trades=int(selector_cfg.get("min_trades", 12)),
                max_dd_limit=float(selector_cfg.get("max_dd_limit", 0.30)),
                cooldown_sec=int(selector_cfg.get("cooldown_sec", 1800)),
            )
            if ConfigSelector and selector_cfg.get("enabled", True)
            else None
        )

        self.engines = [
            ShadowEngine(
                symbol=s,
                interval=self.interval,
                config=self.config,
                allocator=self.allocator,
                portfolio=self.portfolio,
                config_selector=self.config_selector,
            )
            for s in self.symbols
        ]

        autotune_cfg = self.config.get("autotuning", {}) or {}
        self.autotuner = AutoTuningWorker(
            logs_dir=autotune_cfg.get("logs_dir", "logs"),
            interval_sec=int(autotune_cfg.get("interval_sec", 1800)),
            min_trades=int(autotune_cfg.get("min_trades", 12)),
            top_k=int(autotune_cfg.get("top_k", 20)),
            symbol=autotune_cfg.get("symbol"),
        )

        rs_cfg = self.config.get("runtime_state", {}) or {}
        self.runtime_writer = (
            RuntimeStateWriter(
                path=str(rs_cfg.get("path", "logs/runtime_state_shadow.json")),
                interval_sec=float(rs_cfg.get("interval_sec", 1.0)),
                runner_ref=self,
            )
            if rs_cfg.get("enabled", True)
            else None
        )

    async def start(self):
        tasks = [*(e.start() for e in self.engines), self.autotuner.run()]
        if self.runtime_writer:
            tasks.append(self.runtime_writer.run())
        await asyncio.gather(*tasks)


# ============================================================
# Entrypoint
# ============================================================
async def main(symbol: str = "SOLUSDT", interval: str = "1m", config: Optional[dict] = None):
    runner = ShadowRunner(symbol=symbol, interval=interval, config=config or {})
    await runner.start()


if __name__ == "__main__":
    asyncio.run(main())
