# analysis/simulate_hourly_kill.py

import os
import json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

SYMBOL = "SOLUSDT"
TRADES_PATH = os.path.join(LOGS, f"{SYMBOL}_shadow_trades_v2.csv")
OUT_PATH = os.path.join(LOGS, f"{SYMBOL}_hourly_regime_flags.json")

# -----------------------------
# Thresholds (ajustables)
# -----------------------------
MIN_TRADES_PER_HOUR = 5
KILL_EXPECTANCY_R_MAX = -0.02
KILL_WINRATE_MAX = 0.45

TIMEZONE = "America/Argentina/Buenos_Aires"

# -----------------------------
# Load trades
# -----------------------------
df = pd.read_csv(TRADES_PATH)

df = df[df["type"] == "EXIT"].copy()
df["dt_local"] = pd.to_datetime(df["ts_local_iso"], errors="coerce")
df = df.dropna(subset=["dt_local"])

df["hour"] = df["dt_local"].dt.hour
df["regime"] = df["regime"].fillna("UNKNOWN")

# R multiple
df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
df["risk_usdt"] = pd.to_numeric(df["risk_usdt"], errors="coerce")
df["pnl_r"] = df["pnl"] / df["risk_usdt"]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pnl_r"])

# -----------------------------
# Analysis
# -----------------------------
kill_hours = {"TREND": [], "RANGE": []}

for regime in ["TREND", "RANGE"]:
    rg = df[df["regime"] == regime]

    if rg.empty:
        continue

    g = rg.groupby("hour")

    stats = pd.DataFrame({
        "trades": g.size(),
        "expectancy_r": g["pnl_r"].mean(),
        "winrate": g["pnl"].apply(lambda s: (s > 0).mean()),
    })

    for hour, row in stats.iterrows():
        if row["trades"] < MIN_TRADES_PER_HOUR:
            continue

        if (
            row["expectancy_r"] <= KILL_EXPECTANCY_R_MAX
            or row["winrate"] <= KILL_WINRATE_MAX
        ):
            kill_hours[regime].append(int(hour))

# -----------------------------
# Output JSON (bot-readable)
# -----------------------------
payload = {
    "timezone": TIMEZONE,
    "regimes": {
        "TREND": {"kill_hours": sorted(set(kill_hours["TREND"]))},
        "RANGE": {"kill_hours": sorted(set(kill_hours["RANGE"]))},
    }
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print("✅ Simulación horaria completada")
print(json.dumps(payload, indent=2))
print(f"📁 Guardado en: {OUT_PATH}")
