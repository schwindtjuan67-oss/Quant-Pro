#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from io import StringIO

from Live.risk_manager import RiskManager


def _run_case(run_mode: str, disable_flag: bool, exits: int = 3) -> tuple[RiskManager, str]:
    os.environ["RUN_MODE"] = run_mode
    if disable_flag:
        os.environ["PIPELINE_DISABLE_SOFT_MAX_TRADES"] = "1"
    else:
        os.environ.pop("PIPELINE_DISABLE_SOFT_MAX_TRADES", None)
    os.environ.pop("RISK_DISABLE_SOFT_MAX_TRADES", None)

    rm = RiskManager(
        max_trades=5,
        soft_trades_after=1,
        conservative_risk_mult=0.50,
        max_consecutive_losses=999,
        cooldown_on_loss_sec=0,
        cooldown_on_streak_sec=0,
    )

    buf = StringIO()
    with redirect_stdout(buf):
        for _ in range(exits):
            rm.register_trade("EXIT")
    return rm, buf.getvalue()


def main() -> int:
    failures: list[str] = []
    passes: list[str] = []

    rm_normal, out_normal = _run_case("PIPELINE", False, exits=3)
    if not rm_normal.conservative_mode:
        failures.append("expected conservative_mode=True without disable flag")
    if abs(rm_normal.risk_mult - 0.5) > 1e-9:
        failures.append(f"expected risk_mult=0.5 without disable flag, got {rm_normal.risk_mult}")
    if rm_normal.conservative_reason != "SOFT_MAX_TRADES":
        failures.append(f"expected conservative_reason=SOFT_MAX_TRADES, got {rm_normal.conservative_reason!r}")
    if "SOFT_MAX_TRADES disabled" in out_normal:
        failures.append("did not expect disabled log without flag")
    if not failures:
        passes.append("CASE A ok: conservative_mode + risk_mult applied without disable flag")

    rm_disabled, out_disabled = _run_case("PIPELINE", True, exits=3)
    if rm_disabled.conservative_mode:
        failures.append("expected conservative_mode=False with disable flag")
    if abs(rm_disabled.risk_mult - 1.0) > 1e-9:
        failures.append(f"expected risk_mult=1.0 with disable flag, got {rm_disabled.risk_mult}")
    disabled_count = out_disabled.count("[RISK] SOFT_MAX_TRADES disabled (PIPELINE)")
    if disabled_count != 1:
        failures.append(f"expected disabled log exactly once with flag, got {disabled_count}")
    if "SOFT_MAX_TRADES" in out_disabled and disabled_count != 1:
        failures.append("unexpected extra SOFT_MAX_TRADES logs with disable flag")
    if not failures:
        passes.append("CASE B ok: soft-max trades disabled, log once despite 3 checks")

    if failures:
        print("FAIL: selfcheck_risk_soft_max_trades")
        for msg in failures:
            print(f" - {msg}")
        return 1

    print("PASS: selfcheck_risk_soft_max_trades")
    for msg in passes:
        print(f" - {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
