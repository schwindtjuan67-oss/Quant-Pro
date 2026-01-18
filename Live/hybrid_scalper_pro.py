# Live/hybrid_scalper_pro.py

import os
# ============================================================
# RUN MODE (PIPELINE | SHADOW | LIVE)
# ============================================================
RUN_MODE = os.getenv("RUN_MODE", "LIVE").upper()
PIPELINE_VERBOSE_DIAGNOSTICS = os.getenv("PIPELINE_VERBOSE_DIAGNOSTICS", "0").lower() in ("1", "true", "yes")
PIPELINE_DISABLE_GPU = RUN_MODE == "PIPELINE" and os.getenv("PIPELINE_DISABLE_GPU", "0").lower() in ("1", "true", "yes")


import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from Live.order_router import OrderRouter
from Live.risk_manager import RiskManager
from Live.logger_pro import TradeLogger
from Live.delta_router import DeltaRouter
from Live.event_bus import EventBus

from strategies.trend_risk_tools import TrendRiskModule
from strategies.vwap_tools import VWAPModule

# ============================================================
# GPU FEATURES (IMG1) — HOOK 100% (ATR / VWAP / ROLLING MEANS)
# - Try import your repo engine
# - Else: internal fallback (uses CuPy if available, else NumPy)
# ============================================================

try:
    from infra.gpu_features import GPUFeatureEngine  # type: ignore
except Exception:
    class GPUFeatureEngine:
        """
        Fallback self-contained engine:
        - Stores rolling OHLCV (for VWAP)
        - Updates ATR with same recurrence you use (alpha=1/ATR_N)
        - Keeps rolling history for atr_mean and vol_mean
        - Uses CuPy if available; else NumPy.
        """

        def __init__(self, max_len: int = 2000, atr_n: int = 14, max_recent: int = 300):
            self.max_len = int(max_len)
            self.atr_n = int(atr_n)
            self.max_recent = int(max_recent)

            # OHLCV rolling (python lists; computed on xp arrays)
            self._c: List[float] = []
            self._h: List[float] = []
            self._l: List[float] = []
            self._v: List[float] = []

            # Rolling history (for means)
            self._atr_hist: List[float] = []
            self._vol_hist: List[float] = []

            # ATR state
            self._tr: Optional[float] = None
            self._atr: Optional[float] = None

            # pick xp
            self._xp = None
            try:
                import cupy as cp  # type: ignore
                self._xp = cp
            except Exception:
                self._xp = np

        # ---------- rolling inputs ----------
        def push_candle(self, h: float, l: float, c: float, v: float) -> None:
            self._c.append(float(c))
            self._h.append(float(h))
            self._l.append(float(l))
            self._v.append(float(v))
            if len(self._c) > self.max_len:
                self._c.pop(0); self._h.pop(0); self._l.pop(0); self._v.pop(0)

        def push_volume(self, v: float) -> None:
            self._vol_hist.append(float(v))
            if len(self._vol_hist) > self.max_recent:
                self._vol_hist.pop(0)

        def push_prev_atr(self, atr_value: Optional[float]) -> None:
            if atr_value is None:
                return
            self._atr_hist.append(float(atr_value))
            if len(self._atr_hist) > self.max_recent:
                self._atr_hist.pop(0)

        # ---------- ATR update ----------
        def update_atr(self, high: float, low: float, close: float, prev_close: float) -> Tuple[float, float]:
            """
            Matches your recurrence:
              tr = max(h-l, |h-prev_close|, |l-prev_close|)
              atr = tr if None else alpha*tr + (1-alpha)*atr, alpha=1/atr_n
            """
            h = float(high); l = float(low); pc = float(prev_close)
            tr = max(h - l, abs(h - pc), abs(l - pc))
            self._tr = tr

            alpha = 1.0 / float(self.atr_n)
            if self._atr is None:
                self._atr = tr
            else:
                self._atr = alpha * tr + (1.0 - alpha) * float(self._atr)

            return float(self._tr), float(self._atr)

        def atr_value(self) -> Optional[float]:
            return None if self._atr is None else float(self._atr)

        def tr_value(self) -> Optional[float]:
            return None if self._tr is None else float(self._tr)

        def atr_hist_len(self) -> int:
            return int(len(self._atr_hist))

        # ---------- rolling means ----------
        def atr_mean(self, min_len: int = 10) -> Optional[float]:
            if len(self._atr_hist) < int(min_len):
                return None
            xp = self._xp
            a = xp.asarray(self._atr_hist, dtype=float)
            m = float(xp.mean(a))
            return m

        def vol_mean(self, min_len: int = 30) -> Optional[float]:
            if len(self._vol_hist) < int(min_len):
                return None
            xp = self._xp
            a = xp.asarray(self._vol_hist, dtype=float)
            m = float(xp.mean(a))
            return m

        # ---------- VWAP ----------
        def vwap(self, window: int) -> float:
            n = len(self._c)
            if n == 0:
                return 0.0
            w = min(int(window), n)
            xp = self._xp
            c = xp.asarray(self._c[-w:], dtype=float)
            h = xp.asarray(self._h[-w:], dtype=float)
            l = xp.asarray(self._l[-w:], dtype=float)
            v = xp.asarray(self._v[-w:], dtype=float)
            typical = (h + l + c) / 3.0
            pv = typical * v
            volsum = float(xp.sum(v))  # may be 0
            out = float(xp.sum(pv) / (volsum + 1e-9))
            return out


# ===== META CONTROL (FASE 5) =====
try:
    from Live.regime_controller import RegimeController  # recomendado
except Exception:
    RegimeController = None  # fallback: no rompe

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None  # fallback


class HybridScalperPRO:
    # -------------------------------
    # Parámetros base (optimizables)
    # -------------------------------
    EMA_FAST = 9
    EMA_SLOW = 21

    ATR_N = 14
    ATR_STOP_MULT = 1.5
    ATR_TRAIL_MULT = 1.0

    VOL_TARGET = 0.02

    MIN_SCORE_LONG = 2
    MIN_SCORE_SHORT = 2

    RISK_MAX_LOSS_PCT = 0.03
    RISK_MAX_DD_PCT = 0.04
    RISK_MAX_TRADES = 12

    TREND_FAST = 20
    TREND_SLOW = 80

    VWAP_WINDOW = 96
    VWAP_BAND_MULT = 2.0

    # ---- Fees (estimación)
    FEE_RATE_EST = 0.0004  # 0.04% típico; ajustalo a tu realidad si querés

    # -------------------------------
    # RANGE (mínima viable)
    # -------------------------------
    RANGE_ENABLED = True
    RANGE_ENTRY_ATR = 1.5
    RANGE_NO_TRADE_NEAR_ATR = 0.8
    RANGE_BREAK_DIST_ATR = 2.2
    RANGE_BREAK_ATR_RATIO = 1.15
    RANGE_INSIDE_N = 60
    RANGE_INSIDE_MIN = 0.80
    RANGE_ATR_COMP_RATIO = 0.90
    RANGE_TP_TO_VWAP_ATR = 0.25
    RANGE_MAX_HOLD_BARS = 35
    RANGE_STOP_ATR_MULT = 1.2

    # -------------------------------
    # HOURLY GATING (Kill horario por régimen)
    # -------------------------------
    HOURLY_GATING_ENABLED = True
    HOURLY_FLAGS_REFRESH_SEC = 30  # refresca flags cada N segundos (runtime)
    HOURLY_DEFAULT_TZ = "America/Argentina/Buenos_Aires"
    HOURLY_FLAGS_FILENAME_TEMPLATE = "{symbol}_hourly_regime_flags.json"

    # -------------------------------
    # META-CONTROL knobs (FASE 5)
    # -------------------------------
    META_CONTROL_ENABLED = True
    META_REGIME_WEIGHT_TREND = 1.00
    META_REGIME_WEIGHT_RANGE = 0.80
    META_VOL_MULT_MIN = 0.00
    META_VOL_MULT_MAX = 1.50

    # -------------------------------
    # FASE 7.2 — Risk Softening knobs
    # -------------------------------
    CONSERVATIVE_SCORE_BONUS = 0  # +2 es “institucional”: baja trades, sube selectividad

    def __init__(
        self,
        symbol: str,
        router: OrderRouter,
        delta_router: Optional[DeltaRouter],
        risk_manager: RiskManager,
        event_bus: Optional[EventBus] = None,
        logger: Optional[TradeLogger] = None,
        ema_fast: int = EMA_FAST,
        ema_slow: int = EMA_SLOW,
        atr_n: int = ATR_N,
        atr_stop_mult: float = ATR_STOP_MULT,
        atr_trail_mult: float = ATR_TRAIL_MULT,
        vol_target: float = VOL_TARGET,
        min_score_long: int = MIN_SCORE_LONG,
        min_score_short: int = MIN_SCORE_SHORT,
        risk_max_loss_pct: float = RISK_MAX_LOSS_PCT,
        risk_max_dd_pct: float = RISK_MAX_DD_PCT,
        risk_max_trades: int = RISK_MAX_TRADES,
    ):

        self._printed_entry_diag = False

        # --- ENTRY INSTRUMENTATION ---
        self._entry_diag = {
            "candles": 0,
            "trend_ok": 0,
            "vwap_ok": 0,
            "delta_ok": 0,
            "score_ok": 0,
            "risk_ok": 0,
            "all_ok": 0,
        }

        # ### BEGIN PATCH 2: FREQUENCY STATE (anti-churn) ###
        self.cooldown_after_loss_sec = float(getattr(self, "cooldown_after_loss_sec", 45))
        self.cooldown_after_win_sec  = float(getattr(self, "cooldown_after_win_sec", 0))
        self.reentry_block_sec       = float(getattr(self, "reentry_block_sec", 60))

        self.next_entry_allowed_ts = 0.0
        self.last_exit_ts = 0
        self.last_exit_regime = None
        self.last_exit_side = None

        self.last_delta_flip_ts = 0
        self.last_delta_flip_from = None
        self.last_delta_flip_to = None
        # ### END PATCH 2 ###

        self.symbol = symbol
        self.router = router
        self.delta_router = delta_router
        self.event_bus = event_bus
        self.logger = logger

        self.EMA_FAST = ema_fast
        self.EMA_SLOW = ema_slow
        # Guard defensivo: evitar ventanas inválidas (0 o negativas)
        self.ATR_N = max(1, int(atr_n))
        self.ATR_STOP_MULT = atr_stop_mult
        self.ATR_TRAIL_MULT = atr_trail_mult
        self.VOL_TARGET = vol_target
        self.MIN_SCORE_LONG = min_score_long
        self.MIN_SCORE_SHORT = min_score_short

        self.risk_manager = risk_manager
        self.risk_manager.max_loss_pct = risk_max_loss_pct
        self.risk_manager.max_dd_pct = risk_max_dd_pct
        self.risk_manager.max_trades = risk_max_trades

        self.position_side: Optional[str] = None
        self.position_qty: float = 0.0
        self.entry_price: Optional[float] = None
        self.initial_atr: Optional[float] = None
        self.current_sl: Optional[float] = None
        self.best_price: Optional[float] = None

        self.position_regime: Optional[str] = None
        self.position_entry_bar_index: Optional[int] = None

        # NEW: entry risk bookkeeping
        self.atr_entry: Optional[float] = None
        self.sl_initial: Optional[float] = None
        self.risk_usdt: Optional[float] = None
        self.fee_rate_est: float = float(self.FEE_RATE_EST)
        self.fee_est_entry: Optional[float] = None
        self.fee_est_total: Optional[float] = None

        self.ema_fast = None
        self.ema_slow = None
        self.atr = None
        self.tr = None

        self.close_buffer: List[float] = []
        self.high_buffer: List[float] = []
        self.low_buffer: List[float] = []
        self.volume_buffer: List[float] = []

        self.recent_atr: List[float] = []
        self.recent_volume: List[float] = []

        # ============================================================
        # GPU ENGINE STATE (IMG1)
        # - used for ATR/VWAP/rolling means
        # - keeps working even without GPU (fallback uses numpy)
        # ============================================================
        # ------------------------------------------------------------
        # GPUFeatureEngine: compat con firmas distintas (infra vs fallback)
        # ------------------------------------------------------------
        self.VWAP_WINDOW = max(1, int(getattr(self, "VWAP_WINDOW", 96)))
        self.gpu = None
        if not PIPELINE_DISABLE_GPU:
            try:
                # firma fallback (max_recent)
                self.gpu = GPUFeatureEngine(max_len=2000, atr_n=int(self.ATR_N), max_recent=300)
            except TypeError:
                # firma infra.gpu_features (vwap_window/vol_window)
                self.gpu = GPUFeatureEngine(
                    max_len=2000,
                    atr_n=int(self.ATR_N),
                    vwap_window=int(self.VWAP_WINDOW),
                    vol_window=max(1, int(getattr(self, "VOL_WINDOW", 30))),
                )

        # -------- HOURLY FLAGS state
        self._hourly_flags_last_load_ts: float = 0.0
        self._hourly_kill_hours_by_regime: Dict[str, set] = {}
        self._hourly_tz_name: str = str(self.HOURLY_DEFAULT_TZ)

        # -------- META CONTROL state
        self.regime_ctl = None
        if self.META_CONTROL_ENABLED and RegimeController is not None:
            try:
                self.regime_ctl = RegimeController()
            except Exception as e:
                # self._p puede no existir todavía, fallback seguro:
                try:
                    print(f"[META] No pude inicializar RegimeController: {e}")
                except Exception:
                    pass
                self.regime_ctl = None

        try:
            self.trend_module = TrendRiskModule(
                fast_window=self.TREND_FAST,
                slow_window=self.TREND_SLOW,
                mode="trend",
            )
        except Exception:
            class _TrendAdapter:
                def __init__(self, fast, slow, mode="trend"):
                    self.fast = fast
                    self.slow = slow
                    self.mode = mode

                def _ema(self, arr, span):
                    import numpy as _np
                    a = _np.asarray(arr, dtype=float)
                    if a.size == 0:
                        return a
                    out = _np.empty_like(a)
                    alpha = 2.0 / (span + 1.0)
                    out[0] = a[0]
                    for i in range(1, a.size):
                        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
                    return out

                def check_trend(self, close_series):
                    import numpy as _np
                    close = _np.asarray(close_series, dtype=float)
                    if close.size < 2:
                        return False, False
                    ef = self._ema(close, self.fast)
                    es = self._ema(close, self.slow)
                    return bool(ef[-1] > es[-1]), bool(ef[-1] < es[-1])

            self.trend_module = _TrendAdapter(self.TREND_FAST, self.TREND_SLOW, mode="trend")

        try:
            self.vwap_module = VWAPModule(
                window=self.VWAP_WINDOW,
                band_mult=self.VWAP_BAND_MULT,
                mode="trend",
                strict=True,
            )
        except Exception:
            class _VWAPAdapter:
                def __init__(self, window, band_mult, mode="trend", strict=True):
                    self.window = window
                    self.band_mult = band_mult
                    self.mode = mode
                    self.strict = strict

                def _vwap_and_bands(self, close, high, low, volume, window, mult):
                    import numpy as _np
                    c = _np.asarray(close, dtype=float)
                    h = _np.asarray(high, dtype=float)
                    l = _np.asarray(low, dtype=float)
                    v = _np.asarray(volume, dtype=float)
                    n = c.size
                    if n == 0:
                        return 0.0, 0.0, 0.0
                    w = min(window, n)
                    typical = (h[-w:] + l[-w:] + c[-w:]) / 3.0
                    pv = typical * v[-w:]
                    volsum = float(v[-w:].sum())
                    vwap = float(pv.sum() / (volsum + 1e-9))
                    std = float(_np.nanstd(typical)) if typical.size > 1 else 0.0
                    upper = vwap + mult * std
                    lower = vwap - mult * std
                    return vwap, upper, lower

                def check_vwap(self, close_series, high_series, low_series, volume_series):
                    import numpy as _np
                    c = _np.asarray(close_series, dtype=float)
                    h = _np.asarray(high_series, dtype=float)
                    l = _np.asarray(low_series, dtype=float)
                    v = _np.asarray(volume_series, dtype=float)
                    n = c.size
                    if n == 0:
                        return False, False
                    vwap, _, _ = self._vwap_and_bands(c, h, l, v, self.window, self.band_mult)
                    last = float(c[-1])
                    if n >= 2:
                        vwap_prev, _, _ = self._vwap_and_bands(
                            c[:-1], h[:-1], l[:-1], v[:-1], self.window, self.band_mult
                        )
                        prev = float(c[-2])
                        cross_up = (prev < vwap_prev) and (last > vwap)
                        cross_dn = (prev > vwap_prev) and (last < vwap)
                    else:
                        cross_up = last > vwap
                        cross_dn = last < vwap

                    return (bool(cross_up), bool(cross_dn)) if self.strict else (bool(last > vwap), bool(last < vwap))

            self.vwap_module = _VWAPAdapter(self.VWAP_WINDOW, self.VWAP_BAND_MULT, mode="trend", strict=True)

        # ============================================================
        # BACKTEST / SHADOW SILENT MODE
        # ============================================================
        self._silent = False
        try:
            self._silent = bool(getattr(self.router, "is_backtest", False))
        except Exception:
            self._silent = False

        def _p(*args, **kwargs):
            if not self._silent:
                print(*args, **kwargs)
        self._p = _p

        self._p(f"[HYBRID] Inicializado HybridScalperPRO para {self.symbol}")

        # ------------------------------------------------------------
        # first load hourly flags (si existen)
        # ------------------------------------------------------------
        self._reload_hourly_flags(force=True)

        self.last_candle_ts: Optional[int] = None
        self.last_regime: Optional[str] = None
        self.last_regime_ts: int = 0

    # ============================================================
    # FASE 7.2 — Helpers (NO destructivos)
    # ============================================================
    def _risk_mult(self) -> float:
        """Lee risk_mult si existe, sino 1.0 (compatibilidad)."""
        try:
            rm = float(getattr(self.risk_manager, "risk_mult", 1.0))
            if not np.isfinite(rm):
                return 1.0
            return max(0.0, rm)
        except Exception:
            return 1.0

    def _is_conservative(self) -> bool:
        try:
            return bool(getattr(self.risk_manager, "conservative_mode", False))
        except Exception:
            return False

    def _effective_min_scores(self) -> Tuple[int, int]:
        """Si conservative_mode → sube thresholds."""
        ml = int(self.MIN_SCORE_LONG)
        ms = int(self.MIN_SCORE_SHORT)
        if self._is_conservative():
            ml += int(self.CONSERVATIVE_SCORE_BONUS)
            ms += int(self.CONSERVATIVE_SCORE_BONUS)
        return ml, ms

    # ============================================================
    # HOURLY FLAGS (Kill horario por régimen)
    # ============================================================
    def _hourly_flags_path(self) -> str:
        logs_dir = os.path.join(ROOT, "logs")
        fname = self.HOURLY_FLAGS_FILENAME_TEMPLATE.format(symbol=self.symbol)
        return os.path.join(logs_dir, fname)

    @staticmethod
    def _coerce_hours_list(x) -> List[int]:
        out: List[int] = []
        if x is None:
            return out
        if isinstance(x, (list, tuple, set)):
            items = list(x)
        elif isinstance(x, dict):
            items = list(x.keys())
        else:
            items = [x]

        for it in items:
            try:
                h = int(it)
                if 0 <= h <= 23:
                    out.append(h)
            except Exception:
                continue
        return sorted(list(set(out)))

    def _parse_hourly_flags(self, data: dict) -> Tuple[str, Dict[str, set]]:
        tz_name = str(data.get("timezone") or data.get("tz") or self.HOURLY_DEFAULT_TZ)

        regimes_block = data.get("regimes", None)
        if isinstance(regimes_block, dict):
            base = regimes_block
        else:
            base = data

        kill_by_regime: Dict[str, set] = {}
        for rg in ["TREND", "RANGE"]:
            block = base.get(rg, None)
            if isinstance(block, dict):
                hours = (
                    block.get("kill_hours")
                    or block.get("killed_hours")
                    or block.get("hours_killed")
                    or block.get("kill")
                    or block.get("killed")
                )
                kill_by_regime[rg] = set(self._coerce_hours_list(hours))
            else:
                if isinstance(block, (list, tuple, set)):
                    kill_by_regime[rg] = set(self._coerce_hours_list(block))
                else:
                    kill_by_regime[rg] = set()

        gblock = base.get("GLOBAL", None)
        if isinstance(gblock, dict):
            ghours = gblock.get("kill_hours") or gblock.get("killed_hours") or gblock.get("hours_killed")
            kill_by_regime["GLOBAL"] = set(self._coerce_hours_list(ghours))
        elif isinstance(gblock, (list, tuple, set)):
            kill_by_regime["GLOBAL"] = set(self._coerce_hours_list(gblock))
        else:
            kill_by_regime.setdefault("GLOBAL", set())

        return tz_name, kill_by_regime

    def _reload_hourly_flags(self, force: bool = False) -> None:
        if not self.HOURLY_GATING_ENABLED:
            return

        now = time.time()
        if (not force) and (now - self._hourly_flags_last_load_ts) < float(self.HOURLY_FLAGS_REFRESH_SEC):
            return

        path = self._hourly_flags_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tz_name, kill_by_regime = self._parse_hourly_flags(data)
                self._hourly_tz_name = tz_name
                self._hourly_kill_hours_by_regime = kill_by_regime
                self._p(f"[HYBRID][HOURLY] Flags cargados: {path} | tz={tz_name} | kill={ {k:list(v) for k,v in kill_by_regime.items()} }")
            else:
                self._hourly_kill_hours_by_regime = {"TREND": set(), "RANGE": set(), "GLOBAL": set()}
        except Exception as e:
            self._p(f"[HYBRID][HOURLY] Error leyendo flags: {e} (mantengo flags previos)")

        self._hourly_flags_last_load_ts = now

    def _to_local_dt(self, ts_ms: int) -> datetime:
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)
        try:
            ts_i = int(ts_ms)
        except Exception:
            ts_i = int(time.time() * 1000)

        if ts_i < 1_000_000_000_000:
            ts_i = ts_i * 1000

        dt_utc = datetime.fromtimestamp(ts_i / 1000.0, tz=timezone.utc)

        tzname = self._hourly_tz_name or self.HOURLY_DEFAULT_TZ
        if ZoneInfo is not None:
            try:
                return dt_utc.astimezone(ZoneInfo(tzname))
            except Exception:
                return dt_utc
        return dt_utc

    def _hourly_allows_entry(self, regime: str, ts_ms: int) -> Tuple[bool, int, str]:
        if RUN_MODE == "PIPELINE":
            return True, -1, "PIPELINE_BYPASS"

        self._reload_hourly_flags(force=False)

        dt_local = self._to_local_dt(ts_ms)
        hour = int(dt_local.hour)

        kill_global = self._hourly_kill_hours_by_regime.get("GLOBAL", set())
        kill_reg = self._hourly_kill_hours_by_regime.get(regime, set())

        if hour in kill_global:
            return False, hour, "KILL_HOUR_GLOBAL"
        if hour in kill_reg:
            return False, hour, f"KILL_HOUR_{regime}"
        return True, hour, "OK"

    def _log_skip(self, ts_ms: int, regime: str, hour_local: int, reason: str, price: float, snap: Dict[str, Any]) -> None:
        msg = f"[HYBRID][SKIP] {self.symbol} regime={regime} hour={hour_local:02d} reason={reason}"
        self._p(msg)

        if self.logger is not None and RUN_MODE != "PIPELINE":
            try:
                self.logger.log_trade(
                    type="SKIP",
                    timestamp=ts_ms,
                    reason=reason,
                    regime=regime,
                    side="NONE",
                    qty=0.0,
                    price=price,
                    entry_price=0.0,
                    pnl=0.0,
                    pnl_pct=0.0,
                    atr_entry=self.atr_entry,
                    sl_initial=self.sl_initial,
                    risk_usdt=0.0,
                    fee_rate=float(self.fee_rate_est),
                    fee_est_entry=0.0,
                    fee_est_total=0.0,
                    delta_snapshot=snap,
                    risk_state=self.risk_manager.as_dict(),
                    hour_local=int(hour_local),
                )
            except Exception as e:
                self._p(f"[HYBRID][SKIP] logger.log_trade falló: {e}")

        if self.event_bus:
            try:
                self.event_bus.emit("TRADE_SKIP", {
                    "symbol": self.symbol,
                    "regime": regime,
                    "hour_local": int(hour_local),
                    "reason": reason,
                    "price": float(price),
                    "risk_state": self.risk_manager.as_dict(),
                    "delta_snapshot": snap,
                })
            except Exception:
                pass

    # ============================================================
    # META CONTROL helpers (FASE 5)
    # ============================================================
    def _meta_decide(self, regime: str) -> Tuple[bool, float, str]:
        if (not self.META_CONTROL_ENABLED) or (self.regime_ctl is None):
            return True, 1.0, "META_DISABLED_OR_MISSING"

        try:
            md = self.regime_ctl.decide(regime)
            vol_mult = float(md.vol_mult)
            vol_mult = max(float(self.META_VOL_MULT_MIN), min(float(self.META_VOL_MULT_MAX), vol_mult))
            return bool(md.allow), vol_mult, str(md.reason)
        except Exception as e:
            return True, 1.0, f"META_ERR_{e}"

    def _regime_weight(self, regime: str) -> float:
        if regime == "TREND":
            return float(self.META_REGIME_WEIGHT_TREND)
        if regime == "RANGE":
            return float(self.META_REGIME_WEIGHT_RANGE)
        return 1.0

    # ---------------- utils ----------------
    def _append_ohlcv(self, o: float, h: float, l: float, c: float, v: float):
        self.close_buffer.append(c)
        self.high_buffer.append(h)
        self.low_buffer.append(l)
        self.volume_buffer.append(v)

        if len(self.close_buffer) > 2000:
            self.close_buffer.pop(0)
            self.high_buffer.pop(0)
            self.low_buffer.pop(0)
            self.volume_buffer.pop(0)

        # ====== legacy rolling lists (se conservan) ======
        if self.atr is not None:
            self.recent_atr.append(self.atr)
            if len(self.recent_atr) > 300:
                self.recent_atr.pop(0)

        self.recent_volume.append(v)
        if len(self.recent_volume) > 300:
            self.recent_volume.pop(0)

        # ====== GPU ENGINE HOOK (IMG1) ======
        try:
            if getattr(self, "gpu", None) is not None:
                # mantiene mismo "lag" que tu código: empuja ATR previo antes de recalcular ATR actual
                self.gpu.push_prev_atr(self.atr)
                self.gpu.push_volume(v)
                self.gpu.push_candle(h=h, l=l, c=c, v=v)
        except Exception:
            pass

    def _update_ema(self, close_price: float):
        if self.ema_fast is None:
            self.ema_fast = close_price
        else:
            alpha_f = 2 / (self.EMA_FAST + 1)
            self.ema_fast = alpha_f * close_price + (1 - alpha_f) * self.ema_fast

        if self.ema_slow is None:
            self.ema_slow = close_price
        else:
            alpha_s = 2 / (self.EMA_SLOW + 1)
            self.ema_slow = alpha_s * close_price + (1 - alpha_s) * self.ema_slow

        if self.ema_fast is None or self.ema_slow is None:
            return False, False

        return (self.ema_fast > self.ema_slow), (self.ema_fast < self.ema_slow)

    def _update_atr(self, high: float, low: float, close: float):
        # ============================================================
        # GPU HOOK (IMG1): ATR computation
        # - preserva tu fórmula original (incluye el mismo prev_close lookup)
        # ============================================================
        if getattr(self, "gpu", None) is not None:
            try:
                prev_close = self.close_buffer[-1] if self.close_buffer else close
                tr, atr = self.gpu.update_atr(high=high, low=low, close=close, prev_close=prev_close)
                self.tr = tr
                self.atr = atr
                return
            except Exception:
                pass

        # fallback legacy CPU (por si algo explota)
        if self.tr is None:
            tr = high - low
        else:
            prev_close = self.close_buffer[-1] if self.close_buffer else close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self.tr = tr

        alpha = 1 / self.ATR_N
        if self.atr is None:
            self.atr = tr
        else:
            self.atr = alpha * tr + (1 - alpha) * self.atr

    def _filters_trend_vwap(self):
        trend_long_ok, trend_short_ok = self.trend_module.check_trend(close_series=np.array(self.close_buffer))
        vwap_long_ok, vwap_short_ok = self.vwap_module.check_vwap(
            close_series=np.array(self.close_buffer),
            high_series=np.array(self.high_buffer),
            low_series=np.array(self.low_buffer),
            volume_series=np.array(self.volume_buffer),
        )
        return trend_long_ok, trend_short_ok, vwap_long_ok, vwap_short_ok

    def _get_delta_snapshot(self) -> Dict[str, Any]:
        if self.delta_router is None:
            return {
                "last_ts": None,
                "delta_candle": 0.0,
                "delta_candle_prev": 0.0,
                "delta_rolling_15s": 0.0,
                "delta_rolling_60s": 0.0,
                "trades_count_window": 0,
            }
        return self.delta_router.get_snapshot(self.symbol)

    def _delta_allows_long(self, snap: Dict[str, Any]) -> bool:
        if self.delta_router is None:
            return True
        return self.delta_router.allows_long(self.symbol, snap)

    def _delta_allows_short(self, snap: Dict[str, Any]) -> bool:
        if self.delta_router is None:
            return True
        return self.delta_router.allows_short(self.symbol, snap)

    def _calc_vwap_simple(self, window: int) -> float:
        # ============================================================
        # GPU HOOK (IMG1): VWAP simple
        # ============================================================
        if getattr(self, "gpu", None) is not None:
            try:
                return float(self.gpu.vwap(window))
            except Exception:
                pass

        n = len(self.close_buffer)
        if n == 0:
            return 0.0
        w = min(window, n)
        c = np.asarray(self.close_buffer[-w:], dtype=float)
        h = np.asarray(self.high_buffer[-w:], dtype=float)
        l = np.asarray(self.low_buffer[-w:], dtype=float)
        v = np.asarray(self.volume_buffer[-w:], dtype=float)
        typical = (h + l + c) / 3.0
        pv = typical * v
        volsum = float(v.sum())
        return float(pv.sum() / (volsum + 1e-9))

    def _detect_market_regime(self, trend_long_ok: bool, trend_short_ok: bool, vwap_long_ok: bool, vwap_short_ok: bool) -> Dict[str, Any]:
        min_need = max(self.TREND_SLOW, self.VWAP_WINDOW, self.RANGE_INSIDE_N)
        if len(self.close_buffer) < min_need:
            return {"regime": "NOISE", "vwap": 0.0, "atr_sma": 0.0, "inside_ratio": 0.0, "range_invalid": False}

        # usa rolling GPU si existe (misma lógica: requiere “suficientes puntos”)
        if self.atr is None:
            return {"regime": "NOISE", "vwap": 0.0, "atr_sma": 0.0, "inside_ratio": 0.0, "range_invalid": False}

        if getattr(self, "gpu", None) is not None:
            try:
                if int(self.gpu.atr_hist_len()) < 20:
                    return {"regime": "NOISE", "vwap": 0.0, "atr_sma": 0.0, "inside_ratio": 0.0, "range_invalid": False}
            except Exception:
                pass
        else:
            if len(self.recent_atr) < 20:
                return {"regime": "NOISE", "vwap": 0.0, "atr_sma": 0.0, "inside_ratio": 0.0, "range_invalid": False}

        atr = float(self.atr)

        # ============================================================
        # GPU HOOK (IMG1): atr_sma + vwap
        # ============================================================
        if getattr(self, "gpu", None) is not None:
            try:
                atr_sma = float(self.gpu.atr_mean(min_len=1) or 0.0)
            except Exception:
                atr_sma = float(np.mean(self.recent_atr)) if self.recent_atr else 0.0
        else:
            atr_sma = float(np.mean(self.recent_atr)) if self.recent_atr else 0.0

        vwap = float(self._calc_vwap_simple(self.VWAP_WINDOW))
        close = float(self.close_buffer[-1])

        if atr <= 0 or atr_sma <= 0 or vwap == 0.0:
            return {"regime": "NOISE", "vwap": vwap, "atr_sma": atr_sma, "inside_ratio": 0.0, "range_invalid": False}

        trend_ok = bool(trend_long_ok or trend_short_ok)
        vwap_ok = bool(vwap_long_ok or vwap_short_ok)
        atr_expanding = bool(atr > atr_sma)

        if (int(trend_ok) + int(vwap_ok) + int(atr_expanding)) >= 2:
            return {"regime": "TREND", "vwap": vwap, "atr_sma": atr_sma, "inside_ratio": 0.0, "range_invalid": False}

        trend_weak = not trend_ok
        atr_compressed = bool(atr < self.RANGE_ATR_COMP_RATIO * atr_sma)

        band = 1.2 * atr
        upper = vwap + band
        lower = vwap - band

        N = int(self.RANGE_INSIDE_N)
        closes = self.close_buffer[-N:]
        inside_count = sum(1 for x in closes if lower <= float(x) <= upper)
        inside_ratio = inside_count / max(1, len(closes))
        inside_ok = bool(inside_ratio >= self.RANGE_INSIDE_MIN)

        range_invalid = False
        if atr > self.RANGE_BREAK_ATR_RATIO * atr_sma:
            range_invalid = True
        if abs(close - vwap) > self.RANGE_BREAK_DIST_ATR * atr:
            range_invalid = True

        if not range_invalid:
            if (int(trend_weak) + int(atr_compressed) + int(inside_ok)) >= 2:
                return {"regime": "RANGE", "vwap": vwap, "atr_sma": atr_sma, "inside_ratio": inside_ratio, "range_invalid": False}

        return {"regime": "NOISE", "vwap": vwap, "atr_sma": atr_sma, "inside_ratio": inside_ratio, "range_invalid": range_invalid}

    # ============================================================
    # SCORING (NO SE CAMBIA)
    # ============================================================
    def _compute_signal_score(
        self,
        side: str,
        long_cross: bool,
        short_cross: bool,
        trend_long_ok: bool,
        trend_short_ok: bool,
        vwap_long_ok: bool,
        vwap_short_ok: bool,
        delta_long_ok: bool,
        delta_short_ok: bool,
        atr: Optional[float],
        atr_sma: Optional[float],
        volume: float,
        vol_sma: Optional[float],
    ) -> int:
        score = 0

        if side == "LONG":
            if long_cross:
                score += 1
            if trend_long_ok:
                score += 1
            if vwap_long_ok:
                score += 1
            if delta_long_ok:
                score += 1

        elif side == "SHORT":
            if short_cross:
                score += 1
            if trend_short_ok:
                score += 1
            if vwap_short_ok:
                score += 1
            if delta_short_ok:
                score += 1

        if atr is not None and atr_sma is not None:
            if atr_sma > 0 and 0.7 <= (atr / atr_sma) <= 2.0:
                score += 1

        if vol_sma is not None and vol_sma > 0:
            if 0.7 <= (volume / vol_sma) <= 2.5:
                score += 1

        return score

    # ---------- RANGE helpers ----------
    def _range_should_skip(self, price: float, vwap: float, atr: float, atr_sma: float, inside_ratio: float, range_invalid: bool) -> Optional[str]:
        if not self.RANGE_ENABLED:
            return "RANGE_DISABLED"
        if atr <= 0 or atr_sma <= 0 or vwap == 0.0:
            return "RANGE_NO_DATA"
        if abs(price - vwap) < self.RANGE_NO_TRADE_NEAR_ATR * atr:
            return "RANGE_NEAR_VWAP"
        if atr > atr_sma:
            return "RANGE_ATR_EXPANDING"
        if range_invalid:
            return "RANGE_INVALID"
        if inside_ratio < self.RANGE_INSIDE_MIN:
            return "RANGE_LOW_INSIDE"
        return None

    def _try_open_range_mean_reversion(self, price: float, vwap: float, atr: float, atr_sma: float, inside_ratio: float, range_invalid: bool, delta_long_ok: bool, delta_short_ok: bool, vol_target_override: Optional[float] = None):
        reason_skip = self._range_should_skip(price, vwap, atr, atr_sma, inside_ratio, range_invalid)
        if reason_skip is not None:
            self._p(f"[HYBRID][RANGE] Skip: {reason_skip}")
            return

        if price > vwap + self.RANGE_ENTRY_ATR * atr:
            if not delta_short_ok:
                self._p("[HYBRID][RANGE] Skip SHORT: DELTA blocks short")
                return
            sl = price + self.RANGE_STOP_ATR_MULT * atr
            self._open_position("SHORT", price, signal_score=1, reason="RANGE_VWAP", sl_override=sl, regime="RANGE", vol_target_override=vol_target_override)
            return

        if price < vwap - self.RANGE_ENTRY_ATR * atr:
            if not delta_long_ok:
                self._p("[HYBRID][RANGE] Skip LONG: DELTA blocks long")
                return
            sl = price - self.RANGE_STOP_ATR_MULT * atr
            self._open_position("LONG", price, signal_score=1, reason="RANGE_VWAP", sl_override=sl, regime="RANGE", vol_target_override=vol_target_override)
            return

    def _maybe_close_range_position(self, price: float, vwap: float, atr: float):
        if self.position_side is None or self.position_regime != "RANGE":
            return
        if abs(price - vwap) <= self.RANGE_TP_TO_VWAP_ATR * atr:
            self._close_position("RANGE_TP_TO_VWAP", price)
            return
        if self.position_entry_bar_index is not None:
            bars_in_trade = len(self.close_buffer) - self.position_entry_bar_index
            if bars_in_trade >= self.RANGE_MAX_HOLD_BARS:
                self._close_position("RANGE_TIME_STOP", price)
                return

    # ============================================================
    # OPEN/CLOSE
    # ============================================================
    def _open_position(
        self,
        side: str,
        price: float,
        signal_score: int,
        reason: str = "SIGNAL",
        sl_override: Optional[float] = None,
        regime: str = "TREND",
        vol_target_override: Optional[float] = None,
    ):

        if self.position_side is not None:
            self._p("[HYBRID] Ya hay posición abierta, no se abre otra.")
            return

        ### BEGIN PATCH 3A: ENTRY FREQUENCY GUARDS ###
        now_ts = time.time()

        # Cooldown global (post-exit)
        if now_ts < float(self.next_entry_allowed_ts or 0.0):
            return

        # Bloqueo de re-entry en mismo contexto
        if (
            self.last_exit_ts
            and self.last_exit_regime == regime
            and self.last_exit_side == side
            and (now_ts - (self.last_exit_ts / 1000.0)) < float(self.reentry_block_sec)
        ):
            return
        ### END PATCH 3A ###

        if not self.risk_manager.can_trade():
            self._p("[HYBRID] RiskManager bloquea nuevas entradas (max loss / DD / trades).")
            if self.event_bus and RUN_MODE != "PIPELINE":
                self.event_bus.emit("RISK_ALERT", {"symbol": self.symbol, "reason": "ENTRY_BLOCKED", "risk_state": self.risk_manager.as_dict()})
            return

        if self.atr is None or self.atr <= 0:
            self._p("[HYBRID] ATR no disponible, no se abre posición.")
            return

        bal = self.router.get_balance()
        if bal <= 0:
            self._p("[HYBRID] Balance cero, no se abre posición.")
            return

        # ============================================================
        # FASE 7.2: aplicar risk_mult al sizing
        # ============================================================
        base_vol_t = float(vol_target_override) if vol_target_override is not None else float(self.VOL_TARGET)
        base_vol_t *= float(self._risk_mult())

        if base_vol_t <= 0:
            self._p("[HYBRID] vol_target_override<=0 (bloqueado por meta-control / risk_mult).")
            return

        cash_to_risk = bal * base_vol_t
        qty = cash_to_risk / price
        if qty <= 0:
            self._p("[HYBRID] Qty calculada cero, no se abre posición.")
            return

        if sl_override is not None:
            sl = float(sl_override)
        else:
            sl = (price - self.ATR_STOP_MULT * self.atr) if side == "LONG" else (price + self.ATR_STOP_MULT * self.atr)

        atr_entry = float(self.atr)
        sl_initial = float(sl)
        risk_usdt = float(abs(price - sl_initial) * qty)

        fee_rate = float(self.fee_rate_est)
        fee_est_entry = float(fee_rate * (price * qty))
        fee_est_total = float(fee_rate * (price * qty) * 2.0)

        self._p(f"[HYBRID] ENTRY {side} @ {price:.4f} qty={qty:.4f} SL={sl_initial:.4f} | regime={regime} reason={reason} | VOL_T={base_vol_t:.4f}")

        ok = self.router.market_order(side, qty)
        if not ok:
            self._p("[HYBRID] La orden de entrada no fue confirmada, no se guarda estado.")
            return

        self.position_side = side
        self.position_qty = qty
        self.entry_price = price
        self.initial_atr = atr_entry
        self.current_sl = sl_initial
        self.best_price = price

        self.position_regime = regime
        self.position_entry_bar_index = len(self.close_buffer)

        self.atr_entry = atr_entry
        self.sl_initial = sl_initial
        self.risk_usdt = risk_usdt
        self.fee_est_entry = fee_est_entry
        self.fee_est_total = fee_est_total

        try:
            new_equity = self.router.client.sync_balance_from_exchange()
        except Exception:
            new_equity = self.router.get_balance()
        self.risk_manager.update_equity(new_equity)
        self.risk_manager.register_trade(event_type="ENTRY")

        snap = self._get_delta_snapshot()
        if self.logger is not None:
            self.logger.log_trade(
                type="ENTRY",
                timestamp=self.last_candle_ts,
                reason=reason,
                regime=regime,
                side=side,
                qty=qty,
                price=price,
                entry_price=price,
                pnl=0.0,
                pnl_pct=0.0,
                atr_entry=atr_entry,
                sl_initial=sl_initial,
                risk_usdt=risk_usdt,
                fee_rate=fee_rate,
                fee_est_entry=fee_est_entry,
                fee_est_total=fee_est_total,
                delta_snapshot=snap,
                risk_state=self.risk_manager.as_dict(),
            )

        if self.event_bus:
            self.event_bus.emit("TRADE_OPEN", {
                "symbol": self.symbol,
                "side": side,
                "price": price,
                "qty": qty,
                "score": signal_score,
                "risk_state": self.risk_manager.as_dict(),
                "delta_snapshot": snap,
                "regime": regime,
                "reason": reason,
                "atr_entry": atr_entry,
                "sl_initial": sl_initial,
                "risk_usdt": risk_usdt,
                "fee_est_total": fee_est_total,
                "vol_target_used": base_vol_t,
            })

    def _close_position(self, reason: str, price: float):
        if not self.position_side or self.position_qty <= 0:
            return

        close_side = "SELL" if self.position_side == "LONG" else "BUY"

        pnl_abs = 0.0
        pnl_pct = 0.0
        if self.entry_price is not None:
            direction = 1.0 if self.position_side == "LONG" else -1.0
            pnl_abs = (price - self.entry_price) * self.position_qty * direction
            equity_before = self.risk_manager.equity or 0.0
            if equity_before > 0:
                pnl_pct = pnl_abs / equity_before

        self._p(f"[HYBRID] EXIT {self.position_side} reason={reason} @ {price:.4f} qty={self.position_qty:.4f} PnL={pnl_abs:.4f} ({pnl_pct:.4%})")

        if pnl_abs > 0:
            self._p(f"[HYBRID] RESULTADO: GANANCIA +{pnl_abs:.6f} USDT ({pnl_pct:.4%})")
        elif pnl_abs < 0:
            self._p(f"[HYBRID] RESULTADO: PÉRDIDA -{abs(pnl_abs):.6f} USDT ({pnl_pct:.4%})")
        else:
            self._p("[HYBRID] RESULTADO: NEUTRO 0.000000 USDT (0.0000%)")

        self.router.market_order(close_side, self.position_qty)

        try:
            new_equity = self.router.client.sync_balance_from_exchange()
        except Exception:
            new_equity = self.router.get_balance()
        self.risk_manager.update_equity(new_equity)
        self.risk_manager.register_trade(event_type="EXIT")

        fee_rate = float(self.fee_rate_est)
        fee_est_exit = float(fee_rate * (price * self.position_qty))
        fee_est_entry = float(self.fee_est_entry or (fee_rate * ((self.entry_price or 0.0) * self.position_qty)))
        fee_est_total = float(fee_est_entry + fee_est_exit)

        snap = self._get_delta_snapshot()
        if self.logger is not None:
            self.logger.log_trade(
                type="EXIT",
                timestamp=self.last_candle_ts,
                reason=reason,
                regime=self.position_regime,
                side=self.position_side,
                qty=self.position_qty,
                price=price,
                entry_price=self.entry_price or 0.0,
                pnl=pnl_abs,
                pnl_pct=pnl_pct,
                atr_entry=self.atr_entry,
                sl_initial=self.sl_initial,
                risk_usdt=self.risk_usdt,
                fee_rate=fee_rate,
                fee_est_entry=fee_est_entry,
                fee_est_exit=fee_est_exit,
                fee_est_total=fee_est_total,
                delta_snapshot=snap,
                risk_state=self.risk_manager.as_dict(),
            )

        if self.event_bus:
            self.event_bus.emit("TRADE_CLOSE", {
                "symbol": self.symbol,
                "side": self.position_side,
                "price": price,
                "qty": self.position_qty,
                "reason": reason,
                "pnl_abs": pnl_abs,
                "pnl_pct": pnl_pct,
                "risk_state": self.risk_manager.as_dict(),
                "regime": self.position_regime,
                "fee_est_total": fee_est_total,
            })

        # ============================================================
        # PATCH 3B (FIX): cooldown post-exit + pnl_net_est BLINDADO
        # ============================================================
        fee_est_total = float(fee_est_total or 0.0)
        pnl_net_est = float(pnl_abs - fee_est_total)

        self.last_exit_ts = int(self.last_candle_ts or 0)
        self.last_exit_regime = self.position_regime
        self.last_exit_side = self.position_side

        now_ts = time.time()
        if pnl_net_est < 0:
            self.next_entry_allowed_ts = now_ts + float(self.cooldown_after_loss_sec)
        else:
            self.next_entry_allowed_ts = now_ts + float(self.cooldown_after_win_sec)

        if RUN_MODE != "PIPELINE":
            pnl_net_est = float(pnl_abs - fee_est_total)
            self.last_exit_ts = int(self.last_candle_ts or 0)
            self.last_exit_regime = self.position_regime
            self.last_exit_side = self.position_side

            now_ts = time.time()
            if pnl_net_est < 0:
                self.next_entry_allowed_ts = now_ts + float(self.cooldown_after_loss_sec)
            else:
                self.next_entry_allowed_ts = now_ts + float(self.cooldown_after_win_sec)
        # ============================================================

        self.position_side = None
        self.position_qty = 0.0
        self.entry_price = None
        self.initial_atr = None
        self.current_sl = None
        self.best_price = None
        self.position_regime = None
        self.position_entry_bar_index = None

        self.atr_entry = None
        self.sl_initial = None
        self.risk_usdt = None
        self.fee_est_entry = None
        self.fee_est_total = None

    def _update_trailing(self, close_price: float):
        if not self.position_side or self.position_qty <= 0:
            return
        if self.position_regime == "RANGE":
            return
        if self.position_side == "LONG":
            if self.best_price is None or close_price > self.best_price:
                self.best_price = close_price
                self.current_sl = self.best_price - self.ATR_TRAIL_MULT * (self.initial_atr or self.atr or 0)
            if self.current_sl is not None and close_price <= self.current_sl:
                self._close_position("TRAIL_SL_LONG", close_price)
        elif self.position_side == "SHORT":
            if self.best_price is None or close_price < self.best_price:
                self.best_price = close_price
                self.current_sl = self.best_price + self.ATR_TRAIL_MULT * (self.initial_atr or self.atr or 0)
            if self.current_sl is not None and close_price >= self.current_sl:
                self._close_position("TRAIL_SL_SHORT", close_price)

    # ===============================
    # ENTRY DIAGNOSTICS (print helper)
    # ===============================
    def _print_entry_diagnostics(self) -> None:
        if RUN_MODE == "PIPELINE" and not PIPELINE_VERBOSE_DIAGNOSTICS:
            return
        d = getattr(self, "_entry_diag", None)
        if not isinstance(d, dict):
            return
        try:
            print("\nENTRY DIAGNOSTICS")
            for k in ["candles", "trend_ok", "vwap_ok", "delta_ok", "score_ok", "risk_ok", "all_ok"]:
                print(f"{k:12}: {int(d.get(k, 0))}")
            print()
        except Exception:
            pass

    # ============================================================
    # LOOP POR VELA (único on_bar; evita overrides y NameError)
    # ============================================================
    def on_bar(self, candle: dict) -> None:
        # 0) diagnostics
        if hasattr(self, "_entry_diag") and isinstance(self._entry_diag, dict):
            self._entry_diag["candles"] = int(self._entry_diag.get("candles", 0)) + 1
            if (not getattr(self, "_printed_entry_diag", False)) and int(self._entry_diag["candles"]) > 1000:
                self._printed_entry_diag = True
                self._print_entry_diagnostics()

        # 1) parse candle
        o = float(candle.get("open", 0.0))
        h = float(candle.get("high", 0.0))
        l = float(candle.get("low", 0.0))
        c = float(candle.get("close", 0.0))
        v = float(candle.get("volume", 0.0))
        ts = int(candle.get("timestamp", 0) or candle.get("timestamp_ms", 0) or 0)

        self.last_candle_ts = ts

        # 2) indicadores base
        self._append_ohlcv(o, h, l, c, v)
        self._update_atr(h, l, c)
        long_cross, short_cross = self._update_ema(c)

        if self.atr is None:
            return

        trend_long_ok, trend_short_ok, vwap_long_ok, vwap_short_ok = self._filters_trend_vwap()

        snap = self._get_delta_snapshot()
        delta_long_ok = self._delta_allows_long(snap)
        delta_short_ok = self._delta_allows_short(snap)

        # ============================================================
        # GPU HOOK (IMG1): rolling means (atr_sma / vol_sma)
        # ============================================================
        if getattr(self, "gpu", None) is not None:
            try:
                atr_sma = self.gpu.atr_mean(min_len=10)
            except Exception:
                atr_sma = float(np.mean(self.recent_atr)) if len(self.recent_atr) >= 10 else None
            try:
                vol_sma = self.gpu.vol_mean(min_len=30)
            except Exception:
                vol_sma = float(np.mean(self.recent_volume)) if len(self.recent_volume) >= 30 else None
        else:
            atr_sma = float(np.mean(self.recent_atr)) if len(self.recent_atr) >= 10 else None
            vol_sma = float(np.mean(self.recent_volume)) if len(self.recent_volume) >= 30 else None

        # 3) régimen
        reg = self._detect_market_regime(
            trend_long_ok=trend_long_ok,
            trend_short_ok=trend_short_ok,
            vwap_long_ok=vwap_long_ok,
            vwap_short_ok=vwap_short_ok,
        )
        regime = str(reg.get("regime") or "NOISE").upper().strip()

        self.last_regime = regime
        self.last_regime_ts = ts

        vwap_val = float(reg.get("vwap", 0.0) or 0.0)
        atr_sma_val = float(reg.get("atr_sma", 0.0) or 0.0)
        inside_ratio = float(reg.get("inside_ratio", 0.0) or 0.0)
        range_invalid = bool(reg.get("range_invalid", False))

        candle_enriched = dict(candle)
        candle_enriched["regime"] = regime
        candle_enriched["vwap"] = vwap_val
        candle_enriched["atr"] = float(self.atr) if self.atr is not None else None
        candle_enriched["atr_sma"] = atr_sma_val
        candle_enriched["inside_ratio"] = inside_ratio
        candle_enriched["range_invalid"] = range_invalid

        if self.logger is not None and not getattr(self, "_silent", False):
            try:
                self.logger.log_bar(candle=candle_enriched)
            except Exception:
                pass

        # 4) si hay posición, gestionar salida / trailing / delta-flip tighten
        if self.position_side:
            # RANGE management
            if self.position_regime == "RANGE":
                self._maybe_close_range_position(price=c, vwap=vwap_val, atr=float(self.atr))
                if self.position_side == "LONG" and self.current_sl is not None and c <= self.current_sl:
                    self._close_position("RANGE_HARD_SL_LONG", c)
                    return
                if self.position_side == "SHORT" and self.current_sl is not None and c >= self.current_sl:
                    self._close_position("RANGE_HARD_SL_SHORT", c)
                    return
                return

            # TREND management
            self._update_trailing(c)

            # PATCH A: DELTA_FLIP -> tighten SL (NO EXIT)
            try:
                TIGHTEN_K_ATR = 0.25
                if self.entry_price is not None and self.initial_atr:
                    if self.position_side == "LONG" and delta_short_ok:
                        self.last_delta_flip_ts = ts
                        self.last_delta_flip_from = "LONG"
                        self.last_delta_flip_to = "SHORT"
                        if self.current_sl is not None:
                            tightened_sl = float(self.entry_price) + float(TIGHTEN_K_ATR) * float(self.initial_atr)
                            self.current_sl = max(float(self.current_sl), float(tightened_sl))

                    if self.position_side == "SHORT" and delta_long_ok:
                        self.last_delta_flip_ts = ts
                        self.last_delta_flip_from = "SHORT"
                        self.last_delta_flip_to = "LONG"
                        if self.current_sl is not None:
                            tightened_sl = float(self.entry_price) - float(TIGHTEN_K_ATR) * float(self.initial_atr)
                            self.current_sl = min(float(self.current_sl), float(tightened_sl))
            except Exception:
                pass

            # HARD SL fallback
            if self.position_side == "LONG" and self.current_sl is not None and c <= self.current_sl:
                self._close_position("HARD_SL_LONG", c)
                return
            if self.position_side == "SHORT" and self.current_sl is not None and c >= self.current_sl:
                self._close_position("HARD_SL_SHORT", c)
                return

            return  # con posición, no abrir otra

        # 5) sin posición: update equity (solo live)
        if not getattr(self, "_silent", False):
            try:
                self.router.client.sync_balance_from_exchange()
                self.risk_manager.update_equity(self.router.client.balance)
            except Exception:
                pass

        # 6) META-CONTROL (FASE 5)
        effective_vol_target = float(self.VOL_TARGET)

        if regime in ("TREND", "RANGE") and getattr(self, "META_CONTROL_ENABLED", False):
            allow_mc, vol_mult, why_mc = self._meta_decide(regime)
            if not allow_mc:
                dt_local = self._to_local_dt(ts)
                hour_local = int(dt_local.hour)
                self._log_skip(ts_ms=ts, regime=regime, hour_local=hour_local, reason=f"META_{why_mc}", price=c, snap=snap)
                return

            effective_vol_target *= float(vol_mult)
            effective_vol_target *= float(self._regime_weight(regime))
            effective_vol_target = max(
                0.0,
                min(effective_vol_target, float(self.VOL_TARGET) * float(self.META_VOL_MULT_MAX)),
            )

        # 7) HOURLY GATING
        if regime in ("TREND", "RANGE") and getattr(self, "HOURLY_GATING_ENABLED", False):
            allowed, hour_local, why = self._hourly_allows_entry(regime=regime, ts_ms=ts)
            if not allowed:
                self._log_skip(ts_ms=ts, regime=regime, hour_local=hour_local, reason=why, price=c, snap=snap)
                return

        # 8) diagnostics counters
        if hasattr(self, "_entry_diag") and isinstance(self._entry_diag, dict):
            trend_ok_any = bool(trend_long_ok or trend_short_ok)
            vwap_ok_any = bool(vwap_long_ok or vwap_short_ok)
            delta_ok_any = bool(delta_long_ok or delta_short_ok)
            if trend_ok_any:
                self._entry_diag["trend_ok"] = int(self._entry_diag.get("trend_ok", 0)) + 1
            if vwap_ok_any:
                self._entry_diag["vwap_ok"] = int(self._entry_diag.get("vwap_ok", 0)) + 1
            if delta_ok_any:
                self._entry_diag["delta_ok"] = int(self._entry_diag.get("delta_ok", 0)) + 1

        # 9) entries
        eff_min_long, eff_min_short = self._effective_min_scores()

        if regime == "TREND":
            score_long = self._compute_signal_score(
                side="LONG",
                long_cross=long_cross,
                short_cross=short_cross,
                trend_long_ok=trend_long_ok,
                trend_short_ok=trend_short_ok,
                vwap_long_ok=vwap_long_ok,
                vwap_short_ok=vwap_short_ok,
                delta_long_ok=delta_long_ok,
                delta_short_ok=delta_short_ok,
                atr=self.atr,
                atr_sma=atr_sma,
                volume=v,
                vol_sma=vol_sma,
            )
            score_short = self._compute_signal_score(
                side="SHORT",
                long_cross=long_cross,
                short_cross=short_cross,
                trend_long_ok=trend_long_ok,
                trend_short_ok=trend_short_ok,
                vwap_long_ok=vwap_long_ok,
                vwap_short_ok=vwap_short_ok,
                delta_long_ok=delta_long_ok,
                delta_short_ok=delta_short_ok,
                atr=self.atr,
                atr_sma=atr_sma,
                volume=v,
                vol_sma=vol_sma,
            )

            score_ok_any = bool(score_long >= eff_min_long or score_short >= eff_min_short)
            risk_ok = bool(self.risk_manager.can_trade())

            if hasattr(self, "_entry_diag") and isinstance(self._entry_diag, dict):
                if score_ok_any:
                    self._entry_diag["score_ok"] = int(self._entry_diag.get("score_ok", 0)) + 1
                if risk_ok:
                    self._entry_diag["risk_ok"] = int(self._entry_diag.get("risk_ok", 0)) + 1
                if score_ok_any and risk_ok and (trend_long_ok or trend_short_ok) and (vwap_long_ok or vwap_short_ok) and (delta_long_ok or delta_short_ok):
                    self._entry_diag["all_ok"] = int(self._entry_diag.get("all_ok", 0)) + 1

            if score_long >= eff_min_long and self.risk_manager.can_trade():
                self._open_position("LONG", c, signal_score=score_long, reason="TREND_SIGNAL", regime="TREND", vol_target_override=effective_vol_target)
                return

            if score_short >= eff_min_short and self.risk_manager.can_trade():
                self._open_position("SHORT", c, signal_score=score_short, reason="TREND_SIGNAL", regime="TREND", vol_target_override=effective_vol_target)
                return

            return

        if regime == "RANGE":
            try:
                reason_skip = self._range_should_skip(c, vwap_val, float(self.atr), atr_sma_val, inside_ratio, range_invalid)
                if hasattr(self, "_entry_diag") and isinstance(self._entry_diag, dict):
                    if reason_skip is None:
                        self._entry_diag["score_ok"] = int(self._entry_diag.get("score_ok", 0)) + 1
                    if self.risk_manager.can_trade():
                        self._entry_diag["risk_ok"] = int(self._entry_diag.get("risk_ok", 0)) + 1
                    if reason_skip is None and self.risk_manager.can_trade():
                        self._entry_diag["all_ok"] = int(self._entry_diag.get("all_ok", 0)) + 1
            except Exception:
                pass

            self._try_open_range_mean_reversion(
                price=c,
                vwap=vwap_val,
                atr=float(self.atr),
                atr_sma=atr_sma_val,
                inside_ratio=inside_ratio,
                range_invalid=range_invalid,
                delta_long_ok=delta_long_ok,
                delta_short_ok=delta_short_ok,
                vol_target_override=effective_vol_target,
            )
            return

        return
