from analysis.config_selector import ConfigSelector
from analysis.selector_telemetry import emit_selector_telemetry

selector = ConfigSelector()

params, meta = selector.select_from_robust(
    robust_path="results/grid_refine/best_config_robust.json",
    min_score=0.0,
)

emit_selector_telemetry(
    path="logs/telemetry_selector.json",
    symbol="BTCUSDT",
    params=params,
    selector_meta=meta,
)

print("OK selector + telemetry")
