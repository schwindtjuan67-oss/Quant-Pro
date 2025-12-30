import os
import json
import pandas as pd
from datetime import datetime
from analysis_regimes import regime_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

# --- CONFIG ---
SYMBOL = "SOLUSDT"
MIN_TRADES = 5
KILL_EXPECTANCY = -0.02
RISK_EXPECTANCY = 0.01
PROMOTE_EXPECTANCY = 0.05

OUT_FILE = os.path.join(LOGS, f"{SYMBOL}_hourly_regime_flags.json")
EDGE_EXPORT = os.path.join(LOGS, f"{SYMBOL}_hourly_edge_daily.csv")


def build_flags():
    df = pd.read_csv(os.path.join(LOGS, f"{SYMBOL}_shadow_trades_v2.csv"))
    df = df[df["type"] == "EXIT"].copy()

    df["hour_local"] = pd.to_numeric(df["hour_local"], errors="coerce")
    df["pnl_r"] = df["pnl"] / df["risk_usdt"]

    flags = {
        "timezone": "America/Argentina/Buenos_Aires",
        "generated_at": datetime.utcnow().isoformat(),
        "regimes": {
            "TREND": {"kill_hours": [], "risk_hours": [], "promote_hours": []},
            "RANGE": {"kill_hours": [], "risk_hours": [], "promote_hours": []},
        },
    }

    export_rows = []

    for regime in ["TREND", "RANGE"]:
        df_rg = df[df["regime"] == regime]

        for hour, g in df_rg.groupby("hour_local"):
            if len(g) < MIN_TRADES:
                continue

            m = regime_metrics(g)
            exp = m["expectancy_r"]

            export_rows.append({
                "regime": regime,
                "hour": int(hour),
                "trades": m["trades"],
                "expectancy_r": exp,
                "sum_r": m["sum_r"],
                "pf": m["pf"],
            })

            if exp <= KILL_EXPECTANCY:
                flags["regimes"][regime]["kill_hours"].append(int(hour))
            elif exp <= RISK_EXPECTANCY:
                flags["regimes"][regime]["risk_hours"].append(int(hour))
            elif exp >= PROMOTE_EXPECTANCY:
                flags["regimes"][regime]["promote_hours"].append(int(hour))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2)

    pd.DataFrame(export_rows).to_csv(EDGE_EXPORT, index=False)

    print(f"[OK] Flags generados → {OUT_FILE}")
    print(f"[OK] Edge export → {EDGE_EXPORT}")


if __name__ == "__main__":
    build_flags()
