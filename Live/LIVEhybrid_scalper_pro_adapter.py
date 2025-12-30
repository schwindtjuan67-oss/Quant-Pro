from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, Optional, Tuple

from Live.hybrid_scalper_pro import HybridScalperPRO
from analysis.config_state_store import ConfigStateStore


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DEFAULT_STATE_PATH = os.path.join(ROOT, "logs", "hotswap_state.json")


def _now_ts() -> int:
    return int(time.time())


def _upper(x: str) -> str:
    try:
        return str(x).upper().strip()
    except Exception:
        return ""


def _stable_params_key(params: Dict[str, Any]) -> str:
    try:
        return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(params)


def _pick_attr(obj: Any, names: Tuple[str, ...], default=None):
    """Devuelve el primer atributo existente y no-None."""
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
    """
    Adapter para:
      - levantar HybridScalperPRO con firma real
      - conectar ConfigSelector + StateStore (hotswap + rollback)
      - correr tickers desde ShadowEngine

    PATCH BACKTEST:
      - si detecta engine.is_backtest o env QS_BACKTEST=1 o config.backtest=true:
          * verbose=False
          * debug_hotswap=False
          * hotswap_enabled=False
          * rollback_enabled=False
    """

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
        symbol: Optional[str] = None,  # compat: opcional
        *,
        engine: Any,
        allocator: Any = None,
        config_selector: Any = None,
        state_store: Optional[ConfigStateStore] = None,
        hotswap_enabled: bool = True,
        verbose: bool = True,
        debug_hotswap: bool = False,
        hotswap_only_uppercase: bool = False,  # compat legacy (se acepta, se ignora)
        hotswap_interval_sec: int = 60,
        eval_trades: int = 40,
        min_mean_r_improve: float = 0.002,
        rollback_enabled: bool = True,
        rollback_check_interval_sec: int = 60,
        rollback_min_trades: int = 25,
        rollback_min_mean_r: float = -0.002,
    ):
        self.engine = engine

        # -------------------------------------------------
        # FIX: autodetección de symbol si no lo pasan
        # -------------------------------------------------
        if not symbol:
            symbol = getattr(self.engine, "symbol", None)

        if not symbol:
            raise ValueError(
                "HybridAdapterShadow: missing required 'symbol' and could not infer from engine.symbol"
            )

        self.symbol = str(symbol)
        self.allocator = allocator
        self.config_selector = config_selector

        # -------------------------------------------------
        # BACKTEST DETECTION
        # -------------------------------------------------
        self.is_backtest = False
        try:
            if bool(getattr(self.engine, "is_backtest", False)):
                self.is_backtest = True
        except Exception:
            pass

        try:
            if os.environ.get("QS_BACKTEST", "").strip() == "1":
                self.is_backtest = True
        except Exception:
            pass

        try:
            cfg = getattr(self.engine, "config", None) or {}
            if isinstance(cfg, dict) and bool(cfg.get("backtest", False)):
                self.is_backtest = True
        except Exception:
            pass

        # -------------------------------------------------
        # State store
        # -------------------------------------------------
        self.state_store = state_store or ConfigStateStore(DEFAULT_STATE_PATH)

        self.hotswap_enabled = bool(hotswap_enabled)
        self.verbose = bool(verbose)
        self.debug_hotswap = bool(debug_hotswap)
        self.hotswap_interval_sec = int(hotswap_interval_sec)
        self.eval_trades = int(eval_trades)
        self.min_mean_r_improve = float(min_mean_r_improve)

        self.rollback_enabled = bool(rollback_enabled)
        self.rollback_check_interval_sec = int(rollback_check_interval_sec)
        self.rollback_min_trades = int(rollback_min_trades)
        self.rollback_min_mean_r = float(rollback_min_mean_r)

        # -------------------------------------------------
        # PATCH: forzar silencio + off features en BACKTEST
        # -------------------------------------------------
        if self.is_backtest:
            self.verbose = False
            self.debug_hotswap = False
            self.hotswap_enabled = False
            self.rollback_enabled = False

        self._last_hotswap_check_ts = 0
        self._last_rollback_check_ts = 0

        # Estado por símbolo (persistible)
        self._sym_state = self.state_store.get_symbol_state(self.symbol) or {
            "active": None,
            "pending": None,
            "prev": [],
            "eval": None,
            "last_switch_ts": 0,
        }

        # =====================================================
        # Descubrimiento de engine: rutas comunes
        # =====================================================
        router = _pick_attr(self.engine, ("router", "order_router"), default=None)
        delta_router = _pick_attr(self.engine, ("delta_router",), default=None)
        risk_manager = _pick_attr(self.engine, ("risk_manager", "rm"), default=None)
        event_bus = _pick_attr(self.engine, ("event_bus",), default=None)
        logger = _pick_attr(self.engine, ("logger", "trade_logger", "tl"), default=None)

        if router is None:
            attrs = sorted([a for a in dir(self.engine) if not a.startswith("_")])
            raise RuntimeError(
                "No pude detectar router en engine. Asegurate de que ShadowEngine expone:\n"
                "  self.router = <OrderRouter>\n"
                "  self.risk_manager = <RiskManager>\n"
                "  self.delta_router = <DeltaRouter o None>\n"
                "  self.logger = <TradeLogger o None>\n"
                "  self.event_bus = <EventBus o None>\n\n"
                f"Atributos detectados en ShadowEngine (muestra): {attrs}"
            )

        # =====================================================
        # Hybrid REAL (firma correcta)
        # =====================================================
        self.hybrid = HybridScalperPRO(
            symbol=self.symbol,
            router=router,
            delta_router=delta_router,
            risk_manager=risk_manager,
            event_bus=event_bus,
            logger=logger,
        )

        # Boot: aplicar params activos si existen
        try:
            active = self._sym_state.get("active")
            if isinstance(active, dict) and active.get("params"):
                self._apply_params_to_hybrid(active["params"])
        except Exception:
            pass

    # =========================================================
    # Persist helpers
    # =========================================================
    def _persist_state(self) -> None:
        try:
            self.state_store.save_symbol_state(self.symbol, self._sym_state)
        except Exception:
            pass

    def _push_prev(self, snapshot: Dict[str, Any]) -> None:
        try:
            prev = self._sym_state.setdefault("prev", [])
            if not isinstance(prev, list):
                prev = []
                self._sym_state["prev"] = prev
            prev.insert(0, snapshot)

            max_prev = int(getattr(self.state_store, "max_prev", 5))
            if len(prev) > max_prev:
                self._sym_state["prev"] = prev[:max_prev]
        except Exception:
            pass

    # =========================================================
    # Public API (ShadowEngine / BacktestRunner hook)
    # =========================================================
    def on_new_candle(self, candle: dict) -> None:
        """
        Entrada principal desde ShadowEngine / BacktestRunner.
        Forwardea la vela al core HybridScalperPRO (preferimos on_bar).
        """
        try:
            if hasattr(self.hybrid, "on_bar"):
                self.hybrid.on_bar(candle)
            elif hasattr(self.hybrid, "on_new_candle"):
                self.hybrid.on_new_candle(candle)
            else:
                raise AttributeError("HybridScalperPRO has neither on_bar nor on_new_candle")
        except Exception as e:
            # En backtest: 1 línea útil y continuar (sin spamear)
            if self.is_backtest:
                n = getattr(self, "_bt_err_count", 0) + 1
                self._bt_err_count = n
                if n <= 5:
                    print("[BACKTEST][strategy ERROR]", repr(e))
                elif n == 6:
                    print("[BACKTEST][strategy ERROR] (silenciando errores repetidos...)")
                return
            # En live: mejor reventar para no operar “a ciegas”
            raise

        # Hotswap/Rollback solo fuera de backtest
        if not self.is_backtest:
            try:
                self._hotswap_tick(candle)
            except Exception:
                pass
            try:
                self._rollback_tick()
            except Exception:
                pass

    # =========================================================
    # Meta helper (para logger)
    # =========================================================
    def _collect_current_params_for_meta(self) -> Dict[str, Any]:
        params = None
        try:
            params = getattr(self.hybrid, "params", None)
        except Exception:
            params = None

        if isinstance(params, dict) and params:
            out: Dict[str, Any] = {}
            for k in self.DEFAULT_PARAM_KEYS:
                if k in params:
                    out[k] = params.get(k)
            for k in sorted([k for k in params.keys() if k not in out]):
                out[k] = params.get(k)
            return out

        out = {}
        for k in self.DEFAULT_PARAM_KEYS:
            try:
                out[k] = getattr(self.hybrid, k)
            except Exception:
                pass
        return out

    def _inject_meta_json_for_next_trade(self, *, regime: str) -> None:
        lg = getattr(self.engine, "logger", None) or _pick_attr(self.engine, ("trade_logger", "tl"), default=None)
        if lg is None or not hasattr(lg, "set_pending_meta"):
            return

        meta = {
            "params": self._collect_current_params_for_meta(),
            "regime": str(regime or "").upper().strip() or "NOISE",
        }

        try:
            lg.set_pending_meta(meta)  # type: ignore
        except Exception:
            pass

    # =========================================================
    # Internal helpers
    # =========================================================
    def _has_open_position(self) -> bool:
        """Detecta posición abierta de forma robusta (engine o hybrid)."""
        # 1) Engine.position (si existe)
        try:
            pos = getattr(self.engine, "position", None)
            if isinstance(pos, dict) and pos:
                if "is_open" in pos:
                    return bool(pos.get("is_open"))
                side = pos.get("side") or pos.get("position_side")
                qty = pos.get("qty") if "qty" in pos else pos.get("position_qty")
                try:
                    qty_f = float(qty or 0.0)
                except Exception:
                    qty_f = 0.0
                return bool(side) and qty_f > 0
            if pos is not None and hasattr(pos, "is_open"):
                try:
                    return bool(getattr(pos, "is_open"))
                except Exception:
                    pass
        except Exception:
            pass

        # 2) Hybrid (estado interno)
        try:
            side = getattr(self.hybrid, "position_side", None)
            qty = getattr(self.hybrid, "position_qty", 0.0)
            try:
                qty_f = float(qty or 0.0)
            except Exception:
                qty_f = 0.0
            return bool(side) and qty_f > 0
        except Exception:
            return False

    def _apply_params_to_hybrid(self, params: Dict[str, Any]) -> None:
        for k, v in (params or {}).items():
            try:
                setattr(self.hybrid, k, v)
            except Exception:
                pass

    def _portfolio_for_symbol(self):
        try:
            return getattr(self.allocator, "portfolio", None)
        except Exception:
            return None

    def _rolling_mean_r(self, n: int) -> Optional[float]:
        p = self._portfolio_for_symbol()
        if p is None:
            return None
        try:
            dq = p.pnl_r_by_symbol.get(self.symbol)  # type: ignore
            if not dq:
                return None
            vals = list(dq)[-int(n):]
            if not vals:
                return None
            return float(sum(vals) / len(vals))
        except Exception:
            return None

    def _cooldown_remaining_sec(self, cooldown_sec: int) -> int:
        try:
            last_ts = int((self._sym_state or {}).get("last_switch_ts") or 0)
        except Exception:
            last_ts = 0
        if last_ts <= 0:
            return 0
        rem = int(cooldown_sec - (_now_ts() - last_ts))
        return rem if rem > 0 else 0

    def _cooldown_passed(self, cooldown_sec: int) -> bool:
        return self._cooldown_remaining_sec(cooldown_sec) == 0

    # -----------------------------------------------------
    # HOTSWAP helpers
    # -----------------------------------------------------
    def _dlog(self, msg: str) -> None:
        if self.is_backtest:
            return
        if getattr(self, "debug_hotswap", False):
            try:
                print(msg)
            except Exception:
                pass

    # =========================================================
    # HOTSWAP + ROLLBACK
    # =========================================================
    def _hotswap_tick(self, candle: dict) -> None:
        if not self.hotswap_enabled or self.config_selector is None:
            return

        now = _now_ts()
        if (now - self._last_hotswap_check_ts) < self.hotswap_interval_sec:
            return
        self._last_hotswap_check_ts = now

        raw_regime = candle.get("regime") or getattr(self.hybrid, "last_regime", None)
        regime = str(raw_regime or "").upper().strip()

        if not regime:
            self._dlog(f"[HOTSWAP] regime aún no calculado {self.symbol}")
            return

        if regime == "NOISE":
            self._dlog(f"[HOTSWAP] NOISE -> skip {self.symbol}")
            return

        cooldown_sec = int(getattr(self.config_selector, "cooldown_sec", 1800))
        if not self._cooldown_passed(cooldown_sec):
            rem = self._cooldown_remaining_sec(cooldown_sec)
            self._dlog(f"[HOTSWAP] cooldown activo {self.symbol} rem={rem}s")
            return

        params, sel_meta = self.config_selector.select(self.symbol, regime=regime)  # type: ignore
        if not isinstance(params, dict) or not params:
            self._dlog(f"[HOTSWAP] no params para {self.symbol} regime={regime}")
            return

        if self._has_open_position():
            self._sym_state["pending"] = {
                "regime": regime,
                "params": dict(params),
                "params_key": _stable_params_key(params),
                "selector_meta": dict(sel_meta or {}),
                "created_at_ts": _now_ts(),
            }
            self._persist_state()
            self._dlog(f"[HOTSWAP] open position -> pending {self.symbol}")
            return

        self._do_apply_switch(regime, params, sel_meta, "direct_apply")

    def _do_apply_switch(self, regime, params, selector_meta, reason):
        active = self._sym_state.get("active")
        if isinstance(active, dict) and active:
            self._push_prev(active)

        pre_mean = self._rolling_mean_r(int(self.eval_trades))

        snapshot = {
            "id": f"{self.symbol}:{regime}:{_now_ts()}",
            "symbol": self.symbol,
            "regime": str(regime or "").upper().strip() or "NOISE",
            "params": dict(params),
            "params_key": _stable_params_key(params),
            "selector_meta": dict(selector_meta or {}),
            "applied_at_ts": _now_ts(),
            "reason": reason,
            "pre_mean_r": pre_mean,
        }

        self._apply_params_to_hybrid(params)
        self._inject_meta_json_for_next_trade(regime=snapshot["regime"])

        self._sym_state["active"] = snapshot
        self._sym_state["pending"] = None
        self._sym_state["eval"] = {
            "started_at_ts": _now_ts(),
            "regime": snapshot["regime"],
            "eval_trades": int(self.eval_trades),
            "pre_mean_r": pre_mean,
            "status": "running",
        }

        self._sym_state["last_switch_ts"] = _now_ts()
        self._persist_state()

        if self.verbose and (not self.is_backtest):
            if self.debug_hotswap:
                print(f"[HOTSWAP] params_key={snapshot['params_key']}")
            print(
                f"[HOTSWAP] APPLIED {self.symbol} regime={snapshot['regime']} "
                f"reason={reason} pre_mean_r={pre_mean}"
            )

    def _rollback_tick(self) -> None:
        if not self.rollback_enabled:
            return

        now = _now_ts()
        if (now - self._last_rollback_check_ts) < self.rollback_check_interval_sec:
            return
        self._last_rollback_check_ts = now

        active = self._sym_state.get("active")
        eval_state = self._sym_state.get("eval")
        if not isinstance(active, dict) or not isinstance(eval_state, dict):
            return
        if eval_state.get("status") != "running":
            return

        if self._has_open_position():
            return

        mean_r = self._rolling_mean_r(int(self.rollback_min_trades))
        if mean_r is None:
            return

        if float(mean_r) > float(self.rollback_min_mean_r):
            return

        pre_mean = eval_state.get("pre_mean_r")
        if pre_mean is None:
            pre_mean = active.get("pre_mean_r")

        reason = f"mean_r<{self.rollback_min_mean_r} over {self.rollback_min_trades} trades"
        self._do_rollback(reason, float(mean_r), pre_mean)

    def _do_rollback(self, reason, mean_r, pre_mean):
        prev = self._sym_state.get("prev")
        if not isinstance(prev, list) or not prev:
            return

        last_good = prev[0]
        if not isinstance(last_good, dict):
            return

        params = last_good.get("params")
        if not isinstance(params, dict) or not params:
            return

        self._apply_params_to_hybrid(params)

        self._sym_state["active"] = dict(last_good)
        self._sym_state["pending"] = None
        self._sym_state["eval"] = {
            "status": "rolled_back",
            "reason": reason,
            "mean_r": mean_r,
            "pre_mean_r": pre_mean,
            "ended_at_ts": _now_ts(),
        }
        self._sym_state["last_switch_ts"] = _now_ts()

        self._persist_state()

        if self.verbose and (not self.is_backtest):
            print(
                f"[HOTSWAP] ROLLBACK {self.symbol} reason={reason} "
                f"mean_r={mean_r} pre_mean_r={pre_mean}"
            )

    def maybe_apply_pending(self) -> None:
        pend = self._sym_state.get("pending")
        if not isinstance(pend, dict) or not pend:
            return

        if self._has_open_position():
            return

        params = pend.get("params")
        if not isinstance(params, dict) or not params:
            self._sym_state["pending"] = None
            self._persist_state()
            return

        self._do_apply_switch(
            regime=str(pend.get("regime") or "NOISE").upper().strip(),
            params=params,
            selector_meta=pend.get("selector_meta") or {},
            reason="apply_pending_after_close",
        )
