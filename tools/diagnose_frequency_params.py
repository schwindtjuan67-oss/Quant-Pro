import argparse
import json
import os
import sys

from Live.hybrid_scalper_pro import HybridScalperPRO
from Live.order_manager_shadow import ShadowOrderManager
from Live.risk_manager import RiskManager
from Live.logger_pro import TradeLogger


FREQ_KEYS = (
    "delta_rolling_sec",
    "delta_threshold",
    "hour_start",
    "hour_end",
    "use_time_filter",
    "rr_min",
)
FREQ_ATTR_MAP = {
    "delta_rolling_sec": ("delta_rolling_sec", "DELTA_ROLLING_SEC"),
    "delta_threshold": ("delta_threshold", "DELTA_THRESHOLD"),
    "hour_start": ("hour_start", "HOUR_START"),
    "hour_end": ("hour_end", "HOUR_END"),
    "use_time_filter": ("use_time_filter", "USE_TIME_FILTER"),
    "rr_min": ("rr_min", "RR_MIN"),
}


def _is_verbose() -> bool:
    return os.getenv("PIPELINE_VERBOSE_DIAGNOSTICS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _vprint(message: str) -> None:
    if _is_verbose():
        print(message)


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _merge_strategy_kwargs(cfg: dict) -> dict:
    merged = {}
    if not isinstance(cfg, dict):
        return merged
    strategy = cfg.get("strategy")
    if isinstance(strategy, dict) and isinstance(strategy.get("kwargs"), dict):
        merged.update(strategy.get("kwargs") or {})
    if isinstance(cfg.get("strategy_kwargs"), dict):
        merged.update(cfg.get("strategy_kwargs") or {})
    if isinstance(cfg.get("params"), dict):
        merged.update(cfg.get("params") or {})
    if isinstance(cfg.get("strategy_params"), dict):
        merged.update(cfg.get("strategy_params") or {})
    if isinstance(strategy, dict) and isinstance(strategy.get("params"), dict):
        merged.update(strategy.get("params") or {})
    return merged


def _resolve_attr(obj: object, key: str) -> tuple:
    for attr in FREQ_ATTR_MAP.get(key, (key, key.upper())):
        if hasattr(obj, attr):
            try:
                return attr, getattr(obj, attr)
            except Exception:
                return attr, None
    return "<missing>", None


def _apply_params(strategy: HybridScalperPRO, params: dict) -> None:
    if not isinstance(params, dict):
        return
    if hasattr(strategy, "apply_param_overrides"):
        strategy.apply_param_overrides(params)
        return
    for key, value in params.items():
        attr, _value = _resolve_attr(strategy, key)
        if attr != "<missing>":
            setattr(strategy, attr, value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose frequency params on HybridScalperPRO.")
    parser.add_argument("config", nargs="?", help="Optional path to config JSON.")
    args = parser.parse_args()

    if not _is_verbose():
        return 0

    cfg = {}
    if args.config:
        cfg = _load_config(args.config)

    _vprint("[DIAG] HybridScalperPRO attribute presence")
    for key in FREQ_KEYS:
        _vprint(
            f"[DIAG] {key}: has_lower={hasattr(HybridScalperPRO, key)} "
            f"has_upper={hasattr(HybridScalperPRO, key.upper())}"
        )

    merged_kwargs = _merge_strategy_kwargs(cfg)
    _vprint("[DIAG] merged_kwargs=" + json.dumps(merged_kwargs, ensure_ascii=False, sort_keys=True))

    symbol = str(cfg.get("symbol") or "DIAG")
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
    )

    if merged_kwargs:
        _apply_params(strategy, merged_kwargs)

    _vprint("[DIAG] instance values")
    for key in FREQ_KEYS:
        attr, value = _resolve_attr(strategy, key)
        _vprint(f"[DIAG] {key}: {attr}={value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
