from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import inspect

# Fase A -> B handoff contract

os.environ["QS_BACKTEST"] = "1"
RUN_MODE = os.getenv("RUN_MODE", "LIVE").upper().strip()
PIPELINE_MODES = {"PIPELINE", "BACKTEST"}
PIPELINE_VERBOSE_DIAGNOSTICS = os.getenv("PIPELINE_VERBOSE_DIAGNOSTICS", "0").strip().lower() in ("1", "true", "yes")
PIPELINE_DISABLE_GPU = RUN_MODE == "PIPELINE" and os.getenv("PIPELINE_DISABLE_GPU", "0").strip().lower() in ("1", "true", "yes")

def _bt_print(msg: str) -> None:
    # En PIPELINE default = silencioso. Opt-in con PIPELINE_VERBOSE_DIAGNOSTICS=1
    if RUN_MODE != "PIPELINE" or PIPELINE_VERBOSE_DIAGNOSTICS:
        print(msg)

def _expand_strategy_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(kwargs, dict):
        return {}
    mapping = {
        # Map snake_case search params to HybridScalperPRO __init__ kwargs.
        # ema_fast / ema_slow match directly (see Live/hybrid_scalper_pro.py).
        "atr_len": "atr_n",
        "sl_atr_mult": "atr_stop_mult",
        "tp_atr_mult": "atr_trail_mult",
        "max_trades_day": "risk_max_trades",
        "cooldown_sec": "cooldown_after_loss_sec",
    }
    expanded = dict(kwargs)
    for key, mapped in mapping.items():
        if key in kwargs and mapped not in expanded:
            expanded[mapped] = kwargs[key]
    return expanded

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
    - GPU acceleration vÃ­a GPUCandleFeeder (opcional)
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
        from Live.hybrid_scalper_pro import HybridScalperPRO

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
                self.strategy, self._strategy_handler = self._build_strategy(
                    adapter_cls=HybridAdapterShadow,
                    hybrid_cls=HybridScalperPRO,
                )

            def on_new_candle(self, candle: Dict[str, Any]):
                self._strategy_handler(candle)

            def _get_strategy_spec(self) -> Tuple[Optional[str], Dict[str, Any]]:
                cfg = getattr(self, "config", None)
                if not isinstance(cfg, dict):
                    return None, {}
                strategy = cfg.get("strategy")
                if not isinstance(strategy, dict):
                    return None, self._resolve_strategy_kwargs(cfg)
                name = str(strategy.get("name") or "").strip()
                if not name:
                    return None, self._resolve_strategy_kwargs(cfg)
                kwargs = self._resolve_strategy_kwargs(cfg)
                return name, kwargs

            def _resolve_strategy_kwargs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
                if not isinstance(cfg, dict):
                    return {}
                strategy = cfg.get("strategy")
                if RUN_MODE in PIPELINE_MODES:
                    self._merge_pipeline_params(cfg)
                if isinstance(strategy, dict) and isinstance(strategy.get("kwargs"), dict):
                    return _expand_strategy_kwargs(strategy.get("kwargs") or {})
                if isinstance(cfg.get("strategy_kwargs"), dict):
                    return _expand_strategy_kwargs(cfg.get("strategy_kwargs") or {})
                if isinstance(cfg.get("params"), dict):
                    return _expand_strategy_kwargs(cfg.get("params") or {})
                if isinstance(cfg.get("strategy_params"), dict):
                    return _expand_strategy_kwargs(cfg.get("strategy_params") or {})
                if isinstance(strategy, dict) and isinstance(strategy.get("params"), dict):
                    return _expand_strategy_kwargs(strategy.get("params") or {})
                return {}

            def _merge_pipeline_params(self, cfg: Dict[str, Any]) -> None:
                # En PIPELINE/BACKTEST aseguramos que cfg.params incluya overrides
                # para que la estrategia los pueda leer sin depender del schema.
                if not isinstance(cfg, dict):
                    return
                if not isinstance(cfg.get("params"), dict):
                    cfg["params"] = {}
                params_bucket = cfg["params"]
                if not isinstance(params_bucket, dict):
                    return
                strategy = cfg.get("strategy")
                merge_sources: List[Dict[str, Any]] = []
                for key in ("strategy_params", "strategy_kwargs"):
                    if isinstance(cfg.get(key), dict):
                        merge_sources.append(cfg[key])
                if isinstance(strategy, dict):
                    if isinstance(strategy.get("params"), dict):
                        merge_sources.append(strategy["params"])
                    if isinstance(strategy.get("kwargs"), dict):
                        merge_sources.append(strategy["kwargs"])
                for src in merge_sources:
                    params_bucket.update(src)

            def _split_kwargs(
                self,
                cls: Any,
                kwargs: Dict[str, Any],
            ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                try:
                    params = inspect.signature(cls.__init__).parameters
                    accepted = {k for k in params if k != "self"}
                except Exception:
                    accepted = set()
                ctor_kwargs = {k: v for k, v in (kwargs or {}).items() if k in accepted}
                extra_kwargs = {k: v for k, v in (kwargs or {}).items() if k not in accepted}
                return ctor_kwargs, extra_kwargs

            def _apply_param_overrides(self, strategy: Any, params: Dict[str, Any]) -> None:
                if not isinstance(params, dict):
                    return
                if hasattr(strategy, "apply_param_overrides"):
                    try:
                        strategy.apply_param_overrides(params)
                        return
                    except Exception:
                        pass
                # Map snake_case params into uppercase attrs when the Hybrid expects constants.
                # This keeps backward compatibility when __init__ doesn't expose them.
                mapping = {
                    "ema_fast": ("EMA_FAST",),
                    "ema_slow": ("EMA_SLOW",),
                    "atr_len": ("ATR_LEN", "ATR_N"),
                    "sl_atr_mult": ("ATR_STOP_MULT", "RANGE_STOP_ATR_MULT"),
                    "tp_atr_mult": ("ATR_TRAIL_MULT", "RANGE_TP_TO_VWAP_ATR"),
                    "cooldown_sec": ("cooldown_after_loss_sec", "cooldown_after_win_sec", "reentry_block_sec"),
                }
                for key, value in params.items():
                    if hasattr(strategy, key):
                        try:
                            setattr(strategy, key, value)
                        except Exception:
                            pass
                    for mapped in mapping.get(key, ()):
                        if hasattr(strategy, mapped):
                            try:
                                setattr(strategy, mapped, value)
                            except Exception:
                                pass

            def _self_check_params(self, strategy: Any, expected: Dict[str, Any]) -> None:
                if RUN_MODE not in PIPELINE_MODES:
                    return
                if os.getenv("PIPELINE_PARAM_SELF_CHECK", "0").strip().lower() not in ("1", "true", "yes"):
                    return
                if not isinstance(expected, dict):
                    return
                checks = {
                    "ema_fast": ("EMA_FAST", "ema_fast"),
                    "ema_slow": ("EMA_SLOW", "ema_slow"),
                    "atr_len": ("ATR_N", "atr_n"),
                    "sl_atr_mult": ("ATR_STOP_MULT", "atr_stop_mult"),
                    "tp_atr_mult": ("ATR_TRAIL_MULT", "atr_trail_mult"),
                    "max_trades_day": ("risk_max_trades",),
                    "cooldown_sec": ("cooldown_after_loss_sec", "cooldown_after_win_sec", "reentry_block_sec"),
                }
                for key, attrs in checks.items():
                    if key not in expected:
                        continue
                    target = None
                    for attr in attrs:
                        if hasattr(strategy, attr):
                            target = attr
                            break
                    if not target:
                        _bt_print(f"[PARAM-CHECK] {key}: no target attr found")
                        continue
                    try:
                        val = getattr(strategy, target)
                    except Exception:
                        _bt_print(f"[PARAM-CHECK] {key}: failed to read attr {target}")
                        continue
                    if val != expected[key]:
                        _bt_print(f"[PARAM-CHECK] {key}: expected={expected[key]} got={val} via {target}")

            def _apply_extra_kwargs(self, strategy: Any, extra: Dict[str, Any]) -> None:
                for k, v in (extra or {}).items():
                    try:
                        setattr(strategy, k, v)
                    except Exception:
                        pass

            def _apply_legacy_params(self, strategy: Any) -> None:
                cfg = getattr(self, "config", None)
                if not isinstance(cfg, dict):
                    return
                if (
                    isinstance(cfg.get("strategy"), dict)
                    and isinstance(cfg["strategy"].get("kwargs"), dict)
                ) or isinstance(cfg.get("strategy_kwargs"), dict) or isinstance(cfg.get("params"), dict):
                    return
                params = {}
                if isinstance(cfg.get("strategy_params"), dict):
                    params = cfg["strategy_params"]
                elif isinstance(cfg.get("strategy"), dict) and isinstance(cfg["strategy"].get("params"), dict):
                    params = cfg["strategy"]["params"]
                self._apply_extra_kwargs(strategy, params)

            def _resolve_strategy_handler(self, strategy: Any) -> Any:
                for name in ("on_new_candle", "on_new_bar", "on_bar", "on_candle"):
                    fn = getattr(strategy, name, None)
                    if callable(fn):
                        return fn
                raise AttributeError("Strategy does not expose candle handler")

            def _build_strategy(
                self,
                *,
                adapter_cls: Any,
                hybrid_cls: Any,
            ) -> Tuple[Any, Any]:
                name, kwargs = self._get_strategy_spec()
                if name:
                    name_norm = name.strip().lower()
                    if name_norm in ("hybrid_scalper_pro", "hybridscalperpro", "hybrid_scalper"):
                        ctor_kwargs, extra_kwargs = self._split_kwargs(hybrid_cls, kwargs)
                        strategy = hybrid_cls(
                            symbol=self.symbol,
                            router=self.router,
                            delta_router=self.delta_router,
                            risk_manager=self.risk_manager,
                            event_bus=None,
                            logger=self.logger,
                            **ctor_kwargs,
                        )
                        self._apply_extra_kwargs(strategy, extra_kwargs)
                        self._apply_param_overrides(strategy, kwargs)
                        self._apply_legacy_params(strategy)
                        self._self_check_params(strategy, kwargs)
                        return strategy, self._resolve_strategy_handler(strategy)
                    raise ValueError(f"BacktestRunner: unsupported strategy name {name!r}")

                strategy = adapter_cls(
                    engine=self,
                    allocator=None,
                    config_selector=None,
                    hotswap_enabled=False,
                    verbose=False,
                )
                return strategy, self._resolve_strategy_handler(strategy)

        engine = OfflineEngine(self)

        # ---------------- Warmup ----------------
        for i in range(self.warmup):
            c = self.candles[i]
            engine.on_new_candle(c)

        live_candles = self.candles[self.warmup :]

        # ---------------- MAIN LOOP ----------------
        # (extra safety, aunque ya estÃ¡ contemplado en __init__)
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
