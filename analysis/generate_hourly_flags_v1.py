# analysis/generate_hourly_flags_v1.py
import os, json
import pandas as pd
import numpy as np

from analysis_regimes import load_rules, compute_decay_weights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT, "logs")
RULES_PATH = os.path.join(ROOT, "analysis", "regime_rules.yaml")


def build_flags(symbol: str) -> dict:
    rules = load_rules(RULES_PATH)
    tz = rules.get("timezone", "America/Argentina/Buenos_Aires")

    decay = rules.get("decay", {}) or {}
    enabled = bool(decay.get("enabled", True))
    half_life_hours = float(decay.get("half_life_hours", 72.0))

    trades_path = os.path.join(LOGS_DIR, f"{symbol}_shadow_trades_v2.csv")
    if not os.path.exists(trades_path):
        raise FileNotFoundError(trades_path)

    df = pd.read_csv(trades_path)

    # EXIT only
    df = df[df.get("type", "") == "EXIT"].copy()
    if df.empty:
        return {"timezone": tz, "regimes": {"TREND": {"kill_hours": []}, "RANGE": {"kill_hours": []}}, "meta": {"reason": "no_exit_trades"}}

    # dt_local
    if "ts_local_iso" in df.columns:
        df["dt_local"] = pd.to_datetime(df["ts_local_iso"], errors="coerce")
    elif "timestamp" in df.columns:
        df["dt_local"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        df["dt_local"] = pd.NaT
    df = df.dropna(subset=["dt_local"]).sort_values("dt_local")

    # hour_local
    if "hour_local" not in df.columns:
        df["hour_local"] = df["dt_local"].dt.hour

    # pnl_r
    df["pnl"] = pd.to_numeric(df.get("pnl", 0.0), errors="coerce").fillna(0.0)
    df["risk_usdt"] = pd.to_numeric(df.get("risk_usdt", np.nan), errors="coerce")
    df["pnl_r"] = df["pnl"] / df["risk_usdt"]
    df["pnl_r"] = df["pnl_r"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # regime
    df["regime"] = df.get("regime", "UNKNOWN").fillna("UNKNOWN")

    # decay weights
    if enabled:
        w = compute_decay_weights(df["dt_local"], half_life_hours=half_life_hours)
    else:
        w = pd.Series(np.ones(len(df)), index=df.index, dtype=float)

    df["w"] = w

    # === edge por hora: sum_r decayed
    out = {
        "timezone": tz,
        "regimes": {
            "TREND": {"kill_hours": []},
            "RANGE": {"kill_hours": []},
        },
        "meta": {
            "decay_enabled": enabled,
            "half_life_hours": half_life_hours,
            "note": "kill_hours = horas con sum_r_decayed < 0 (por régimen)",
        },
    }

    for rg in ["TREND", "RANGE"]:
        d = df[df["regime"] == rg].copy()
        if d.empty:
            continue

        g = d.groupby("hour_local")
        hourly_sum_r = g.apply(lambda x: float((x["pnl_r"] * x["w"]).sum()))
        hourly_eff = g["w"].sum()

        # criterio kill simple y robusto para empezar:
        # - si sum_r_decayed < 0 y hay masa efectiva mínima (>= 1.5)
        kill = []
        for hour, sr in hourly_sum_r.items():
            eff = float(hourly_eff.loc[hour])
            if eff >= 1.5 and sr < 0.0:
                kill.append(int(hour))

        out["regimes"][rg]["kill_hours"] = sorted(list(set(kill)))

    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SOLUSDT")
    args = p.parse_args()

    symbol = args.symbol.strip().upper()
    flags = build_flags(symbol)

    path = os.path.join(LOGS_DIR, f"{symbol}_hourly_regime_flags.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(flags, f, ensure_ascii=False, indent=2)

    print(f"[OK] flags written -> {path}")


if __name__ == "__main__":
    main()
