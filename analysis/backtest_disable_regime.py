# analysis/backtest_disable_regime.py
import pandas as pd
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

PATH = os.path.join(LOGS, "SOLUSDT_shadow_trades_v2.csv")

df = pd.read_csv(PATH)
df = df[df["type"] == "EXIT"].copy()

df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
df["risk_usdt"] = pd.to_numeric(df["risk_usdt"], errors="coerce")
df["pnl_r"] = df["pnl"] / df["risk_usdt"]

base = df["pnl_r"].sum()

no_range = df[df["regime"] != "RANGE"]["pnl_r"].sum()
no_trend = df[df["regime"] != "TREND"]["pnl_r"].sum()

print("🧪 BACKTEST REGIME TOGGLE")
print(f"TOTAL R: {base:.2f}")
print(f"SIN RANGE: {no_range:.2f}  Δ {(no_range-base):+.2f}")
print(f"SIN TREND: {no_trend:.2f}  Δ {(no_trend-base):+.2f}")
