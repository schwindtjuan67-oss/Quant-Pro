#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from io import StringIO

from Live.risk_manager import RiskManager


def _run_case(run_mode: str, disable_flag: bool, exits: int = 5) -> tuple[RiskManager, str]:
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
    passes: list[str] = []

    def _assert(condition: bool, message: str) -> None:
        assert condition, message

    try:
        rm_normal, out_normal = _run_case("PIPELINE", False)
        _assert(rm_normal.conservative_mode, "expected conservative_mode=True without disable flag")
        _assert(
            abs(rm_normal.risk_mult - rm_normal.conservative_risk_mult) < 1e-9,
            f"expected risk_mult={rm_normal.conservative_risk_mult} without disable flag, got {rm_normal.risk_mult}",
        )
        _assert(
            rm_normal.conservative_reason == "SOFT_MAX_TRADES",
            f"expected conservative_reason=SOFT_MAX_TRADES, got {rm_normal.conservative_reason!r}",
        )
        _assert(
            "SOFT_MAX_TRADES disabled (PIPELINE)" not in out_normal,
            "did not expect disabled log without flag",
        )
        passes.append("CASE A ok: conservative_mode + risk_mult applied without disable flag")

        rm_disabled, out_disabled = _run_case("PIPELINE", True)
        _assert(not rm_disabled.conservative_mode, "expected conservative_mode=False with disable flag")
        _assert(
            abs(rm_disabled.risk_mult - 1.0) < 1e-9,
            f"expected risk_mult=1.0 with disable flag, got {rm_disabled.risk_mult}",
        )
        _assert(
            abs(rm_disabled.risk_mult - rm_disabled.conservative_risk_mult) > 1e-9,
            "expected risk_mult not to drop to conservative_risk_mult with disable flag",
        )
        disabled_count = out_disabled.count("SOFT_MAX_TRADES disabled (PIPELINE)")
        _assert(
            disabled_count == 1,
            f"expected disabled log exactly once with flag, got {disabled_count}",
        )
        passes.append("CASE B ok: soft-max trades disabled, log once despite multiple checks")
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1

    print("PASS: selfcheck_risk_soft_max_trades")
    for msg in passes:
        print(f" - {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
