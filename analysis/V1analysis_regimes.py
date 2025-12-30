import pandas as pd
from pathlib import Path

LOG_DIR = Path("../logs")
BARS_FILE = LOG_DIR / "SOLUSDT_shadow_bars.csv"
TRADES_FILE = LOG_DIR / "SOLUSDT_shadow_trades.csv"

def load_data():
    bars = pd.read_csv(BARS_FILE)
    trades = pd.read_csv(TRADES_FILE)

    trades = trades[trades["type"] == "EXIT"].copy()
    return bars, trades

def compute_expectancy(df):
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    winrate = len(wins) / len(df) if len(df) > 0 else 0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

    expectancy = winrate * avg_win + (1 - winrate) * avg_loss
    return winrate, expectancy

def analyze_by_regime(bars, trades):
    results = []

    for regime, t in trades.groupby("regime"):
        if len(t) < 5:
            continue  # no tomamos decisiones con muestra chica

        winrate, expectancy = compute_expectancy(t)

        mae = t["mae"].mean()
        mfe = t["mfe"].mean()
        hold = t["holding_time_sec"].mean()

        results.append({
            "regime": regime,
            "trades": len(t),
            "winrate": round(winrate, 3),
            "expectancy": round(expectancy, 5),
            "avg_mae": round(mae, 5),
            "avg_mfe": round(mfe, 5),
            "avg_hold_sec": round(hold, 1),
        })

    return pd.DataFrame(results)

def verdict(row):
    if row["expectancy"] <= 0:
        return "KILL"
    if row["avg_mae"] > row["avg_mfe"]:
        return "KILL"
    if row["expectancy"] > 0:
        return "PROMOTE"
    return "HOLD"

def main():
    bars, trades = load_data()
    summary = analyze_by_regime(bars, trades)

    if summary.empty:
        print("No hay suficientes trades para análisis.")
        return

    summary["decision"] = summary.apply(verdict, axis=1)

    print("\n=== REGIME ANALYSIS ===\n")
    print(summary.to_string(index=False))

    summary.to_csv("analysis_regime_summary.csv", index=False)
    print("\nSaved → analysis_regime_summary.csv")

if __name__ == "__main__":
    main()
