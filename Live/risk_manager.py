# Live/risk_manager.py
# ===============================================================
#   RiskManager PRO — Daily Risk Limits + Soft Limits + Cooldowns
#   Compatible con HybridScalperPRO (update_equity / can_trade / register_trade / as_dict)
# ===============================================================

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class RiskDecision:
    allow: bool
    reason: str
    mode: str  # "NORMAL" | "CONSERVATIVE" | "HARD_STOP"
    risk_mult: float  # 1.0 normal, <1.0 para recortar riesgo (allocator puede usarlo)
    cooldown_sec: int = 0


class RiskManager:
    """
    Objetivo:
      - Hard stops (pérdida diaria / DD diario)
      - Soft limits (max trades/día, cooldowns, rachas de pérdidas)
      - Reset diario automático (sin requerir cron externo)
      - Mantener compatibilidad con el Hybrid actual.
    """

    def __init__(
        self,
        max_loss_pct: float = 0.03,
        max_dd_pct: float = 0.04,
        max_trades: int = 600,
        starting_equity: float = 1000.0,

        # --- Soft-limit knobs (institucional / robusto) ---
        soft_trades_after: Optional[int] = None,   # si None -> usa max_trades como gatillo
        conservative_risk_mult: float = 0.50,      # si entra en conservador, sugerimos recortar riesgo
        max_consecutive_losses: int = 3,           # racha de pérdidas -> conservador/cooldown
        cooldown_on_loss_sec: int = 0,             # cooldown tras un loss (opcional)
        cooldown_on_streak_sec: int = 600,         # cooldown si se rompe max_consecutive_losses
        timezone_name: str = "America/Argentina/Buenos_Aires",
    ):
        # config base
        self.max_loss_pct = float(max_loss_pct)
        self.max_dd_pct = float(max_dd_pct)
        self.max_trades = int(max_trades)

        self.starting_equity = float(starting_equity)
        self.equity = float(starting_equity)

        # tracking de equity extrema (para DD)
        self.max_equity = float(starting_equity)
        self.min_equity = float(starting_equity)

        # daily tracking
        self.timezone_name = str(timezone_name)
        self.day_start_equity = float(starting_equity)
        self.day_max_equity = float(starting_equity)
        self.day_min_equity = float(starting_equity)
        self.day_key = self._current_day_key()

        # trades
        self.trades_today = 0          # cuenta por EXIT (trade completo)
        self.entries_today = 0         # cuenta por ENTRY (debug)
        self.exits_today = 0           # cuenta por EXIT (debug)

        # soft-limit mode
        self.conservative_mode = False
        self.conservative_reason = ""
        self.risk_mult = 1.0

        # performance micro (para rachas)
        self.consecutive_losses = 0
        self.last_pnl_abs: Optional[float] = None
        self.last_event_ts: float = 0.0

        # cooldown
        self.cooldown_until_ts: float = 0.0

        # knobs
        self.soft_trades_after = int(soft_trades_after) if soft_trades_after is not None else int(self.max_trades)
        self.conservative_risk_mult = float(conservative_risk_mult)
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.cooldown_on_loss_sec = int(cooldown_on_loss_sec)
        self.cooldown_on_streak_sec = int(cooldown_on_streak_sec)

    # -----------------------------------------------------------
    # Internals: day key
    # -----------------------------------------------------------
    def _current_day_key(self) -> str:
        """
        Day key local-ish: usamos hora local por TZ si está disponible (zoneinfo),
        pero mantenemos fallback estable si no hay zoneinfo.
        """
        try:
            from datetime import datetime, timezone
            from zoneinfo import ZoneInfo

            dt = datetime.now(timezone.utc).astimezone(ZoneInfo(self.timezone_name))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            # fallback: UTC date (estable)
            return time.strftime("%Y-%m-%d", time.gmtime())

    def _maybe_reset_daily(self):
        k = self._current_day_key()
        if k == self.day_key:
            return

        # reset daily
        self.day_key = k
        self.day_start_equity = float(self.equity)
        self.day_max_equity = float(self.equity)
        self.day_min_equity = float(self.equity)

        self.trades_today = 0
        self.entries_today = 0
        self.exits_today = 0

        self.conservative_mode = False
        self.conservative_reason = ""
        self.risk_mult = 1.0

        self.consecutive_losses = 0
        self.last_pnl_abs = None

        self.cooldown_until_ts = 0.0

        print(f"[RISK] Daily reset ({self.day_key}). day_start_equity={self.day_start_equity:.4f}")

    # -----------------------------------------------------------
    def update_equity(self, new_equity: float):
        self._maybe_reset_daily()

        if new_equity is None:
            return

        self.equity = float(new_equity)
        self.max_equity = max(self.max_equity, self.equity)
        self.min_equity = min(self.min_equity, self.equity)

        # daily extremes (DD diario)
        self.day_max_equity = max(self.day_max_equity, self.equity)
        self.day_min_equity = min(self.day_min_equity, self.equity)

    # -----------------------------------------------------------
    def set_conservative(self, reason: str, risk_mult: Optional[float] = None):
        self.conservative_mode = True
        self.conservative_reason = str(reason or "CONSERVATIVE")
        self.risk_mult = float(risk_mult) if risk_mult is not None else float(self.conservative_risk_mult)
        self.risk_mult = max(0.0, min(1.0, self.risk_mult))

    def clear_conservative(self):
        self.conservative_mode = False
        self.conservative_reason = ""
        self.risk_mult = 1.0

    # -----------------------------------------------------------
    def set_cooldown(self, seconds: int, reason: str = "COOLDOWN"):
        seconds = int(max(0, seconds))
        if seconds <= 0:
            return
        until = time.time() + seconds
        self.cooldown_until_ts = max(self.cooldown_until_ts, until)
        # si hay cooldown, entramos conservador (no hard stop)
        self.set_conservative(reason=reason, risk_mult=self.conservative_risk_mult)
        print(f"[RISK] Cooldown {seconds}s set. reason={reason}")

    # -----------------------------------------------------------
    def register_trade(self, event_type: str = "EXIT", pnl_abs: Optional[float] = None):
        """
        Compat:
          - Hybrid llama register_trade("ENTRY") y register_trade("EXIT")
        Opcional:
          - pnl_abs (si lo pasás desde executor o logger) ayuda a rachas.
        """
        self._maybe_reset_daily()
        self.last_event_ts = time.time()

        et = str(event_type or "").upper().strip()

        if et == "ENTRY":
            self.entries_today += 1
            return

        if et == "EXIT":
            self.exits_today += 1
            self.trades_today += 1

            # rachas con pnl si está disponible
            if pnl_abs is not None:
                try:
                    p = float(pnl_abs)
                except Exception:
                    p = None
                if p is not None:
                    self.last_pnl_abs = p
                    if p < 0:
                        self.consecutive_losses += 1
                        if self.cooldown_on_loss_sec > 0:
                            self.set_cooldown(self.cooldown_on_loss_sec, reason="COOLDOWN_ON_LOSS")
                    elif p > 0:
                        self.consecutive_losses = 0

            # soft: si excede trades, conservador (pero no hard stop)
            if self.trades_today >= int(self.soft_trades_after):
                self.set_conservative(reason="SOFT_MAX_TRADES", risk_mult=self.conservative_risk_mult)

            # soft: racha de pérdidas -> cooldown más fuerte
            if self.consecutive_losses >= int(self.max_consecutive_losses):
                self.set_cooldown(self.cooldown_on_streak_sec, reason="LOSS_STREAK")

            return

        # otros eventos: no rompas
        return

    # -----------------------------------------------------------
    def _hard_stop_check(self) -> Optional[str]:
        """
        Hard stops basados en métricas diarias (más realista para live):
          - DD desde day_max_equity
          - Loss desde day_start_equity
        """
        if self.day_max_equity > 0:
            dd = (self.equity - self.day_max_equity) / self.day_max_equity
            if dd <= -float(self.max_dd_pct):
                return "HARD_STOP_DD"

        if self.day_start_equity > 0:
            loss_pct = (self.equity - self.day_start_equity) / self.day_start_equity
            if loss_pct <= -float(self.max_loss_pct):
                return "HARD_STOP_DAILY_LOSS"

        return None

    # -----------------------------------------------------------
    def decision(self) -> RiskDecision:
        """
        API rica (para allocator / hybrid si querés usarla).
        """
        self._maybe_reset_daily()

        # cooldown gate
        now = time.time()
        if now < self.cooldown_until_ts:
            remaining = int(self.cooldown_until_ts - now)
            return RiskDecision(
                allow=False,
                reason=f"COOLDOWN_{remaining}s",
                mode="CONSERVATIVE",
                risk_mult=float(self.risk_mult),
                cooldown_sec=remaining,
            )

        # hard stops
        hs = self._hard_stop_check()
        if hs is not None:
            return RiskDecision(
                allow=False,
                reason=hs,
                mode="HARD_STOP",
                risk_mult=0.0,
                cooldown_sec=0,
            )

        # soft mode
        if self.conservative_mode:
            return RiskDecision(
                allow=True,
                reason=self.conservative_reason or "CONSERVATIVE",
                mode="CONSERVATIVE",
                risk_mult=float(self.risk_mult),
                cooldown_sec=0,
            )

        return RiskDecision(
            allow=True,
            reason="OK",
            mode="NORMAL",
            risk_mult=1.0,
            cooldown_sec=0,
        )

    # -----------------------------------------------------------
    def can_trade(self) -> bool:
        """
        Compat con Hybrid:
          - False solo si hay bloqueo real (hard stop o cooldown).
          - True si está en modo conservador (pero Hybrid debe endurecer señales).
        """
        d = self.decision()
        if not d.allow:
            if d.mode == "HARD_STOP":
                print(f"[RISK] HARD STOP – {d.reason}")
            else:
                print(f"[RISK] BLOCK – {d.reason}")
            return False

        if d.mode == "CONSERVATIVE":
            # permitimos operar, pero avisamos.
            print(f"[RISK] CONSERVATIVE – {d.reason} | risk_mult={d.risk_mult:.2f}")

        return True

    # -----------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        self._maybe_reset_daily()
        hs = self._hard_stop_check()
        return {
            # equity
            "equity": float(self.equity),
            "max_equity": float(self.max_equity),
            "min_equity": float(self.min_equity),

            # daily
            "day_key": self.day_key,
            "day_start_equity": float(self.day_start_equity),
            "day_max_equity": float(self.day_max_equity),
            "day_min_equity": float(self.day_min_equity),

            # limits
            "max_loss_pct": float(self.max_loss_pct),
            "max_dd_pct": float(self.max_dd_pct),
            "max_trades": int(self.max_trades),
            "soft_trades_after": int(self.soft_trades_after),

            # trades
            "trades_today": int(self.trades_today),
            "entries_today": int(self.entries_today),
            "exits_today": int(self.exits_today),

            # mode
            "conservative_mode": bool(self.conservative_mode),
            "conservative_reason": str(self.conservative_reason),
            "risk_mult": float(self.risk_mult),

            # streaks/cooldown
            "consecutive_losses": int(self.consecutive_losses),
            "cooldown_until_ts": float(self.cooldown_until_ts),
            "hard_stop_reason": hs,
            "last_pnl_abs": (float(self.last_pnl_abs) if self.last_pnl_abs is not None else None),
        }
