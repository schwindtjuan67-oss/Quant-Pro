from __future__ import annotations

import os
import json
import inspect
from typing import Any, Dict, Optional, Tuple

from Live.hybrid_scalper_pro import HybridScalperPRO
from analysis.config_state_store import ConfigStateStore

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DEFAULT_STATE_PATH = os.path.join(ROOT, "logs", "hotswap_state.json")


def _stable_params_key(params: Dict[str, Any]) -> str:
    try:
        return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(params)


def _pick_attr(obj: Any, names: Tuple[str, ...], default=None):
    for n in names:
        try:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
        except Exception:
            continue
    return default


class HybridAdapterShadow:
    DEFAULT_PARAM_KEYS = (
        "ema_fast",
        "ema_slow",
        "atr_period",
        "atr_mult",
        "take_profit",
        "stop_loss",
        "max_trades_per_day",
        "cooldown_after_loss_sec",
        "cooldown_after_win_sec",
        "use_delta_filter",
        "delta_absorb_threshold",
        "delta_roll_sec",
        "vwap_dev_mult",
        "min_volatility",
        "max_volatility",
    )

    def __init__(
        self,
        symbol: Optional[str] = None,
        *,
        engine: Any,
        allocator: Any = None,
        config_selector: Any = None,
        state_store: Optional[ConfigStateStore] = None,
        hotswap_enabled: bool = True,
        verbose: bool = True,
        debug_hotswap: bool = False,
        hotswap_interval_sec: int = 60,
        eval_trades: int = 40,
        min_mean_r_improve: float = 0.002,
        rollback_enabled: bool = True,
        rollback_check_interval_sec: int = 60,
        rollback_min_trades: int = 25,
        rollback_min_mean_r: float = -0.002,
    ):
        self.engine = engine

        if not symbol:
            symbol = getattr(self.engine, "symbol", None)
        if not symbol:
            raise ValueError("HybridAdapterShadow: missing 'symbol'")
        self.symbol = str(symbol)

        self.allocator = allocator
        self.config_selector = config_selector

        self.is_backtest = bool(
            getattr(self.engine, "is_backtest", False)
            or os.environ.get("QS_BACKTEST") == "1"
        )

        self.state_store = state_store or ConfigStateStore(DEFAULT_STATE_PATH)

        self.hotswap_enabled = bool(hotswap_enabled)
        self.verbose = bool(verbose)
        self.debug_hotswap = bool(debug_hotswap)
        self.rollback_enabled = bool(rollback_enabled)

        if self.is_backtest:
            self.verbose = False
            self.debug_hotswap = False
            self.hotswap_enabled = False
            self.rollback_enabled = False

        self._sym_state = self.state_store.get_symbol_state(self.symbol) or {}

        router = _pick_attr(self.engine, ("router", "order_router"))
        delta_router = _pick_attr(self.engine, ("delta_router",))
        risk_manager = _pick_attr(self.engine, ("risk_manager", "rm"))
        event_bus = _pick_attr(self.engine, ("event_bus",))
        logger = _pick_attr(self.engine, ("logger", "trade_logger", "tl"))

        if router is None:
            raise RuntimeError("HybridAdapterShadow: router no detectado")

        self.logger = logger

        raw_kwargs = self._resolve_strategy_kwargs_from_engine()
        expanded_kwargs = self._expand_strategy_kwargs(raw_kwargs)
        init_kwargs, _extras = self._split_init_kwargs(expanded_kwargs)

        self.hybrid = HybridScalperPRO(
            symbol=self.symbol,
            router=router,
            delta_router=delta_router,
            risk_manager=risk_manager,
            event_bus=event_bus,
            logger=logger,
            **init_kwargs,
        )

        # === PATCH PIPELINE ===
        self._apply_pipeline_overrides(expanded_kwargs)

        active = self._sym_state.get("active")
        if isinstance(active, dict) and active.get("params"):
            self._apply_params_to_hybrid(active["params"])

        self._inject_params_meta()

        self._hybrid_handler_fn = None

    # -----------------------
    # PATCH CORE
    # -----------------------
    def _resolve_strategy_kwargs_from_engine(self) -> Dict[str, Any]:
        cfg = getattr(self.engine, "config", None)
        if not isinstance(cfg, dict):
            return {}
        strategy = cfg.get("strategy")
        if isinstance(strategy, dict) and isinstance(strategy.get("kwargs"), dict):
            return strategy.get("kwargs") or {}
        if isinstance(cfg.get("strategy_kwargs"), dict):
            return cfg.get("strategy_kwargs") or {}
        if isinstance(cfg.get("params"), dict):
            return cfg.get("params") or {}
        if isinstance(cfg.get("strategy_params"), dict):
            return cfg.get("strategy_params") or {}
        if isinstance(strategy, dict) and isinstance(strategy.get("params"), dict):
            return strategy.get("params") or {}
        return {}

    def _expand_strategy_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        mapping = {
            "atr_len": "atr_n",
            "sl_atr_mult": "atr_stop_mult",
            "tp_atr_mult": "atr_trail_mult",
            "max_trades_day": "risk_max_trades",
            "cooldown_sec": "cooldown_after_loss_sec",
        }
        expanded = dict(kwargs or {})
        for key, mapped in mapping.items():
            if key in expanded and mapped not in expanded:
                expanded[mapped] = expanded[key]
        return expanded

    def _split_init_kwargs(self, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            params = inspect.signature(HybridScalperPRO.__init__).parameters
            accepted = {k for k in params if k != "self"}
        except Exception:
            accepted = set()
        init_kwargs = {k: v for k, v in (kwargs or {}).items() if k in accepted}
        extra_kwargs = {k: v for k, v in (kwargs or {}).items() if k not in accepted}
        return init_kwargs, extra_kwargs

    def _apply_pipeline_overrides(self, kwargs: Dict[str, Any]) -> None:
        applied = []
        skipped = []
        mapping = {
            # Map snake_case params into uppercase attrs when HybridScalperPRO stores constants.
            "ema_fast": ("EMA_FAST",),
            "ema_slow": ("EMA_SLOW",),
            "atr_len": ("ATR_LEN", "ATR_N"),
            "sl_atr_mult": ("ATR_STOP_MULT", "RANGE_STOP_ATR_MULT"),
            "tp_atr_mult": ("ATR_TRAIL_MULT", "RANGE_TP_TO_VWAP_ATR"),
            "cooldown_sec": ("cooldown_after_loss_sec", "cooldown_after_win_sec", "reentry_block_sec"),
        }
        for k, v in (kwargs or {}).items():
            if hasattr(self.hybrid, k):
                try:
                    setattr(self.hybrid, k, v)
                    applied.append(k)
                    continue
                except Exception:
                    pass
            upper = k.upper()
            if hasattr(self.hybrid, upper):
                try:
                    setattr(self.hybrid, upper, v)
                    applied.append(upper)
                    continue
                except Exception:
                    pass
            for mapped in mapping.get(k, ()):
                if hasattr(self.hybrid, mapped):
                    try:
                        setattr(self.hybrid, mapped, v)
                        applied.append(mapped)
                        break
                    except Exception:
                        pass
            skipped.append(k)

        if os.getenv("PIPELINE_VERBOSE_DIAGNOSTICS", "").strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON"):
            init_kwargs, _extras = self._split_init_kwargs(kwargs)
            print("[ADAPTER][DIAG] strategy_kwargs=", json.dumps(kwargs, ensure_ascii=False, sort_keys=True))
            print("[ADAPTER][DIAG] init_kwargs=", json.dumps(init_kwargs, ensure_ascii=False, sort_keys=True))
            print("[ADAPTER][DIAG] overrides_applied=", json.dumps(sorted(set(applied)), ensure_ascii=False))
            print("[ADAPTER][DIAG] overrides_skipped=", json.dumps(sorted(set(skipped)), ensure_ascii=False))

    def _inject_params_meta(self) -> None:
        if not self.logger or not hasattr(self.logger, "set_pending_meta"):
            return

        params = {}
        for k in self.DEFAULT_PARAM_KEYS:
            try:
                params[k] = getattr(self.hybrid, k)
            except Exception:
                params[k] = None  # clave: mantener keys estables

        meta = {
            "params": params,
            "params_key": _stable_params_key(params),
            "source": os.getenv("RUN_MODE", "LIVE"),
        }

        try:
            self.logger.set_pending_meta(meta)
        except Exception:
            pass

    # -----------------------
    # Strategy hook
    # -----------------------
    def _resolve_hybrid_handler(self):
        for name in ("on_bar", "on_new_candle", "on_new_bar", "on_candle"):
            fn = getattr(self.hybrid, name, None)
            if callable(fn):
                self._hybrid_handler_fn = fn
                return
        raise AttributeError("HybridScalperPRO no expone handler")

    def on_new_candle(self, candle: dict) -> None:
        if self._hybrid_handler_fn is None:
            self._resolve_hybrid_handler()

        try:
            self._hybrid_handler_fn(candle)
        except Exception:
            if self.is_backtest:
                return
            raise

    # -----------------------
    # Helpers
    # -----------------------
    def _apply_params_to_hybrid(self, params: Dict[str, Any]) -> None:
        for k, v in (params or {}).items():
            if hasattr(self.hybrid, k):
                try:
                    setattr(self.hybrid, k, v)
                    continue
                except Exception:
                    pass
            upper = k.upper()
            if hasattr(self.hybrid, upper):
                try:
                    setattr(self.hybrid, upper, v)
                except Exception:
                    pass
