import pandas as pd
import numpy as np

# ===============================
# CONFIG
# ===============================
CSV_PATH = "SOLUSDT_shadow_trades_v4.csv"

# ===============================
# LOAD
# ===============================
df = pd.read_csv(CSV_PATH)

# Normalizamos
df.columns = [c.strip() for c in df.columns]
df = df[df["type"] == "EXIT"].copy()

# Asegurar tipos numéricos
NUM_COLS = [
    "pnl", "pnl_pct", "pnl_net_est", "fee_est",
    "pnl_r", "risk_usdt", "mfe", "mae"
]
for c in NUM_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ===============================
# 1️⃣ ¿PNL_R ES NETO O BRUTO?
# ===============================
df["_r_from_net"] = df["pnl_net_est"] / df["risk_usdt"]
df["_r_diff"] = df["pnl_r"] - df["_r_from_net"]

r_diff_mean = df["_r_diff"].abs().mean()

print("\n" + "="*60)
print("CHECK pnl_r vs pnl_net_est")
print(f"Mean |pnl_r - pnl_net_est/risk| = {r_diff_mean:.6f}")

if r_diff_mean < 0.01:
    print("✅ pnl_r ES NETO (fees ya descontadas)")
    R_COL = "pnl_r"
else:
    print("⚠️ pnl_r parece BRUTO → usamos pnl_net_est / risk")
    df["pnl_r_net"] = df["_r_from_net"]
    R_COL = "pnl_r_net"

# ===============================
# HELPERS
# ===============================
def expectancy(x):
    return x.mean()

def summarize(group, r_col):
    return pd.Series({
        "trades": len(group),
        "winrate_%": (group[r_col] > 0).mean() * 100,
        "avg_R": group[r_col].mean(),
        "expectancy": expectancy(group[r_col]),
        "avg_fee_%": (group["fee_est"] / group["risk_usdt"]).mean() * 100
    })

# ===============================
# 2️⃣ EXPECTANCY POR REGIMEN
# ===============================
regime_stats = (
    df.groupby("regime")
      .apply(summarize, R_COL)
      .sort_values("expectancy", ascending=False)
)

print("\n" + "="*60)
print("EXPECTANCY POR REGIMEN")
print(regime_stats.round(4))

# ===============================
# 3️⃣ EXPECTANCY POR TIPO DE SALIDA
# ===============================
exit_stats = (
    df.groupby("reason")
      .apply(summarize, R_COL)
      .sort_values("expectancy", ascending=False)
)

print("\n" + "="*60)
print("EXPECTANCY POR EXIT REASON")
print(exit_stats.round(4))

# ===============================
# 4️⃣ NOISE CHECK
# ===============================
if "NOISE" in df["regime"].unique():
    noise = df[df["regime"] == "NOISE"]
    print("\n" + "="*60)
    print("NOISE REGIME CHECK")
    print(f"Trades NOISE: {len(noise)}")
    print(f"Expectancy NOISE: {noise[R_COL].mean():.4f}")

# ===============================
# 5️⃣ MFE / MAE DIAGNOSTICO RAPIDO
# ===============================
df["_mfe_r"] = df["mfe"] / df["risk_usdt"]
df["_mae_r"] = df["mae"] / df["risk_usdt"]

print("\n" + "="*60)
print("MFE / MAE QUICK CHECK")
print(df[["_mfe_r", "_mae_r", R_COL]].describe().round(4))

# ===============================
# 6️⃣ TOP PROBLEMAS
# ===============================
print("\n" + "="*60)
print("EXIT REASONS NEGATIVOS IMPORTANTES")
print(exit_stats[exit_stats["expectancy"] < 0].round(4))
