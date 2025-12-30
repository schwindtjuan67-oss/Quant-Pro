# analysis/regime_metrics.py
import pandas as pd

df = pd.read_csv("logs/trades_SOLUSDT.csv")

exits = df[df["type"] == "EXIT"].copy()

exits["R"] = exits["pnl"] / exits["risk_usdt"]

group = exits.groupby("regime").agg(
    trades=("R", "count"),
    expectancy=("R", "mean"),
    total_R=("R", "sum"),
    pnl_usdt=("pnl", "sum"),
    fees=("fee_est_total", "sum"),
)

print(group)
