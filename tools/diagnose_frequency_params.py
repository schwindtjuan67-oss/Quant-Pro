import os


def _verbose_enabled() -> bool:
    return os.getenv("PIPELINE_VERBOSE_DIAGNOSTICS", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
        "on",
        "ON",
    )


def _build_strategy_instance():
    from Live.hybrid_scalper_pro import HybridScalperPRO

    class Dummy:
        pass

    class DummyRiskManager:
        def __init__(self):
            self.max_loss_pct = 0.0
            self.max_dd_pct = 0.0
            self.max_trades = 0

    return HybridScalperPRO(
        symbol="DIAG",
        router=Dummy(),
        delta_router=None,
        risk_manager=DummyRiskManager(),
        event_bus=None,
        logger=None,
    )


def main() -> None:
    if not _verbose_enabled():
        return

    from Live.hybrid_scalper_pro import HybridScalperPRO

    strategy = _build_strategy_instance()
    keys = [
        "delta_rolling_sec",
        "delta_threshold",
        "hour_start",
        "hour_end",
        "use_time_filter",
        "rr_min",
        "cooldown_sec",
        "max_trades_day",
    ]

    print("[DIAG] HybridScalperPRO frequency param attribute check")
    for key in keys:
        lower = key
        upper = key.upper()
        class_lower = hasattr(HybridScalperPRO, lower)
        class_upper = hasattr(HybridScalperPRO, upper)
        inst_lower = hasattr(strategy, lower)
        inst_upper = hasattr(strategy, upper)
        lower_value = getattr(strategy, lower, None) if inst_lower else None
        upper_value = getattr(strategy, upper, None) if inst_upper else None
        print(
            "[DIAG] key="
            f"{key} class_lower={class_lower} class_upper={class_upper} "
            f"inst_lower={inst_lower} inst_upper={inst_upper} "
            f"lower_value={lower_value} upper_value={upper_value}"
        )


if __name__ == "__main__":
    main()
