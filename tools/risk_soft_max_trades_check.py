import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Live.risk_manager import RiskManager


def _run_case(disable_flag: bool):
    os.environ["RUN_MODE"] = "PIPELINE"
    if disable_flag:
        os.environ["PIPELINE_DISABLE_SOFT_MAX_TRADES"] = "1"
    else:
        os.environ.pop("PIPELINE_DISABLE_SOFT_MAX_TRADES", None)
        os.environ.pop("RISK_DISABLE_SOFT_MAX_TRADES", None)

    rm = RiskManager(max_trades=2, soft_trades_after=2, starting_equity=1000.0)
    rm.register_trade("EXIT", pnl_abs=1.0)
    rm.register_trade("EXIT", pnl_abs=1.0)
    return rm


def main() -> int:
    rm_disabled = _run_case(True)
    rm_enabled = _run_case(False)

    if rm_disabled.risk_mult != 1.0 or rm_disabled.conservative_mode:
        print("[CHECK] Failed: disable flag should keep risk_mult=1.0 and conservative_mode=False")
        return 1

    if rm_enabled.risk_mult >= 1.0 or not rm_enabled.conservative_mode:
        print("[CHECK] Failed: without flag should enable conservative mode and reduce risk_mult")
        return 1

    print("[CHECK] OK: soft max trades toggle behaves as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
