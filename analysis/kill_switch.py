from __future__ import annotations

from typing import Any, Dict, List, Union
import time


def _now_ts() -> int:
    return int(time.time())


def _compute_max_dd(equity: List[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for x in equity:
        peak = max(peak, x)
        max_dd = max(max_dd, peak - x)
    return float(max_dd)


def _to_returns(series: Union[List[float], List[Dict[str, Any]]]) -> List[float]:
    if not series:
        return []
    if isinstance(series[0], dict):
        return [float(t.get("pnl_r", 0.0)) for t in series]  # type: ignore[index]
    return [float(x) for x in series]  # type: ignore[return-value]


class RollingDDKillSwitch:
    """
    Kill-switch v1:
      - DD rolling por trades cerrados (pnl_r acumulado)
      - Determinista, simple
    """

    def __init__(
        self,
        *,
        window_trades: int = 20,
        dd_limit_r: float = 6.0,
        action: str = "halt",  # "halt" | "fallback"
        cooldown_sec: int = 1800,
    ):
        self.window_trades = int(window_trades)
        self.dd_limit_r = float(dd_limit_r)
        self.action = str(action)
        self.cooldown_sec = int(cooldown_sec)

        self._last_trigger_ts: int = 0
        self._active: bool = False

    def evaluate(self, closed_trades: Union[List[float], List[Dict[str, Any]]]) -> Dict[str, Any]:
        now = _now_ts()

        if self._active and (now - self._last_trigger_ts) < self.cooldown_sec:
            return {
                "triggered": True,
                "reason": "cooldown",
                "action": self.action,
                "ts": now,
            }

        returns = _to_returns(closed_trades)
        if len(returns) < self.window_trades:
            return {"triggered": False, "reason": "insufficient_trades", "ts": now, "n": len(returns)}

        recent = returns[-self.window_trades:]
        equity = []
        s = 0.0
        for r in recent:
            s += r
            equity.append(s)

        dd = _compute_max_dd(equity)

        if dd >= self.dd_limit_r:
            self._active = True
            self._last_trigger_ts = now
            return {
                "triggered": True,
                "reason": "rolling_dd_exceeded",
                "dd_r": dd,
                "limit_r": self.dd_limit_r,
                "window_trades": self.window_trades,
                "action": self.action,
                "ts": now,
            }

        return {
            "triggered": False,
            "dd_r": dd,
            "limit_r": self.dd_limit_r,
            "window_trades": self.window_trades,
            "ts": now,
        }


def enrich_with_telemetry(status: Dict[str, Any], selector_meta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(status)
    picked = selector_meta.get("picked_item") or {}
    out.update({
        "selector": selector_meta.get("selector"),
        "top_k_path": selector_meta.get("top_k_path"),
        "regime": selector_meta.get("regime") or picked.get("regime"),
        "window": picked.get("window"),
        "seed": picked.get("seed"),
        "score": picked.get("score") or selector_meta.get("picked_rank"),
    })
    return out
