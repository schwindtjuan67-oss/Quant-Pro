from __future__ import annotations

import os
import json
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

        self.hybrid = HybridScalperPRO(
            symbol=self.symbol,
            router=router,
            delta_router=delta_router,
            risk_manager=risk_manager,
            event_bus=event_bus,
            logger=logger,
        )

        # === PATCH PIPELINE ===
        self._apply_pipeline_params_from_engine()

        active = self._sym_state.get("active")
        if isinstance(active, dict) and active.get("params"):
            self._apply_params_to_hybrid(active["params"])

        self._inject_params_meta()

        self._hybrid_handler_fn = None

    # -----------------------
    # PATCH CORE
    # -----------------------
    def _apply_pipeline_params_from_engine(self) -> None:
        cfg = getattr(self.engine, "config", None)
        if not isinstance(cfg, dict):
            return
        params = cfg.get("params") or cfg.get("strategy_params")
        if not isinstance(params, dict):
            return
        for k, v in params.items():
            try:
                setattr(self.hybrid, k, v)
            except Exception:
                pass

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
            try:
                setattr(self.hybrid, k, v)
            except Exception:
                pass


