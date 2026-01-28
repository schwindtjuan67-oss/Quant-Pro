from __future__ import annotations

import os


def is_truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    raw = str(raw).strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def disable_soft_max_trades_enabled() -> bool:
    run_mode = os.getenv("RUN_MODE", "LIVE").upper().strip()
    if run_mode != "PIPELINE":
        return False
    return is_truthy_env("PIPELINE_DISABLE_SOFT_MAX_TRADES") or is_truthy_env("RISK_DISABLE_SOFT_MAX_TRADES")
