from analysis.config_selector import ConfigSelector
from analysis.kill_switch import RollingDDKillSwitch, enrich_with_telemetry
from analysis.selector_telemetry import emit_selector_telemetry

# 1) selector
selector = ConfigSelector()
params, meta = selector.select_from_top_k(
    top_k_path="logs/top_k_library.json",
    symbol="SOLUSDT",
    regime="TREND",
)

print("params:", params)
print("meta:", meta)

emit_selector_telemetry(
    path="logs/telemetry_selector.json",
    symbol="SOLUSDT",
    params=params or {},
    selector_meta=meta,
    event="selector_pick",
)

# 2) kill switch (simulación)
closed_trades = [{"pnl_r": -1.2}, {"pnl_r": -0.9}, {"pnl_r": -2.1}, {"pnl_r": -1.0}, {"pnl_r": -1.1}]
kill = RollingDDKillSwitch(window_trades=5, dd_limit_r=5.0, action="halt")
status = kill.evaluate(closed_trades)

if status["triggered"]:
    payload = enrich_with_telemetry(status, meta)
    emit_selector_telemetry(
        path="logs/telemetry_selector.json",
        symbol="SOLUSDT",
        params=params or {},
        selector_meta=payload,
        event="kill_switch",
    )

print("kill:", status)
