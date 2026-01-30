#!/usr/bin/env python3
import inspect
import json
import os
import sys
from typing import Any, Dict, List

from backtest.backtest_runner import _expand_strategy_kwargs
from backtest.run_backtest import _merge_strategy_kwargs
from Live.hybrid_scalper_pro import HybridScalperPRO
from Live.logger_pro import TradeLogger
from Live.order_manager_shadow import ShadowOrderManager
from Live.risk_manager import RiskManager


FREQUENCY_KEYS = [
    "delta_rolling_sec",
    "delta_threshold",
    "hour_start",
    "hour_end",
    "use_time_filter",
    "rr_min",
    "cooldown_sec",
    "max_trades_day",
]


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_strategy(cfg: Dict[str, Any]) -> HybridScalperPRO:
    kwargs = _merge_strategy_kwargs(cfg)
    kwargs = _expand_strategy_kwargs(kwargs)
    params = inspect.signature(HybridScalperPRO.__init__).parameters
    accepted = {k for k in params if k != "self"}
    init_kwargs = {k: v for k, v in (kwargs or {}).items() if k in accepted}
    extra_kwargs = {k: v for k, v in (kwargs or {}).items() if k not in accepted}

    symbol = str(cfg.get("symbol") or "TEST").upper()
    logger = TradeLogger(symbol)
    router = ShadowOrderManager(symbol, cfg)
    risk_manager = RiskManager(
        max_loss_pct=cfg.get("max_loss", 0.03),
        max_dd_pct=cfg.get("max_dd", 0.04),
        max_trades=cfg.get("max_trades", 12),
        starting_equity=1000.0,
    )
    strategy = HybridScalperPRO(
        symbol=symbol,
        router=router,
        delta_router=None,
        risk_manager=risk_manager,
        event_bus=None,
        logger=logger,
        **init_kwargs,
    )
    for key, value in extra_kwargs.items():
        setattr(strategy, key, value)
    if hasattr(strategy, "apply_param_overrides"):
        strategy.apply_param_overrides(kwargs)
    return strategy


def _match_attrs(strategy: HybridScalperPRO, key: str) -> List[str]:
    key_lower = key.lower()
    key_compact = key_lower.replace("_", "")
    matches = []
    for attr in dir(strategy):
        attr_lower = attr.lower()
        if key_lower in attr_lower or key_compact in attr_lower:
            matches.append(attr)
    return sorted(matches)


def _resolve_value(strategy: HybridScalperPRO, key: str) -> Dict[str, Any]:
    candidates = [
        key,
        key.upper(),
    ]
    if key == "cooldown_sec":
        candidates.extend(
            [
                "cooldown_after_loss_sec",
                "cooldown_after_win_sec",
                "reentry_block_sec",
                "COOLDOWN_SEC",
            ]
        )
    if key == "max_trades_day":
        candidates.extend(["risk_max_trades", "MAX_TRADES_DAY"])
    for attr in candidates:
        if hasattr(strategy, attr):
            return {"attr": attr, "value": getattr(strategy, attr)}
    if key == "max_trades_day":
        rm = getattr(strategy, "risk_manager", None)
        if rm is not None and hasattr(rm, "max_trades"):
            return {"attr": "risk_manager.max_trades", "value": getattr(rm, "max_trades")}
    return {"attr": "<missing>", "value": None}


def main() -> int:
    if os.getenv("PIPELINE_VERBOSE_DIAGNOSTICS", "").strip().lower() not in ("1", "true", "yes", "on"):
        print("[DIAG] PIPELINE_VERBOSE_DIAGNOSTICS is not enabled; exiting.")
        return 0
    if len(sys.argv) < 2:
        print("Usage: python tools/diagnose_frequency_params.py <config.json>")
        return 1

    cfg = _load_config(sys.argv[1])
    strategy = _build_strategy(cfg)

    print("[DIAG] Frequency attribute matches and resolved values:")
    for key in FREQUENCY_KEYS:
        matches = _match_attrs(strategy, key)
        resolved = _resolve_value(strategy, key)
        print(f"- {key}: matches={matches}")
        print(f"  -> {resolved['attr']} = {resolved['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
