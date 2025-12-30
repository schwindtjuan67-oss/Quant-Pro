import os
import json
import pandas as pd
import numpy as np

CSV_PATH = os.getenv("PIPELINE_TRADES_CSV", "results/pipeline_trades.csv")
OUT_CSV  = os.getenv("PIPELINE_CANDIDATES_CSV", "results/pipeline_candidates_for_shadow.csv")
OUT_JSON = os.getenv("PIPELINE_CANDIDATES_JSON", "results/pipeline_candidates_for_shadow.json")

# ======================
# Config (tunables)
# ======================
MIN_TRADES = int(os.getenv("PIPELINE_MIN_TRADES", "300"))
MIN_R_OBS  = int(os.getenv("PIPELINE_MIN_R_OBS", "200"))

TH_EXPECTANCY = float(os.getenv("PIPELINE_TH_EXPECTANCY", "0.05"))
TH_SORTINO    = float(os.getenv("PIPELINE_TH_SORTINO", "1.50"))
TH_PF         = float(os.getenv("PIPELINE_TH_PF", "1.30"))
TH_DD         = float(os.getenv("PIPELINE_TH_DD", "-0.20"))
TH_WINRATE    = float(os.getenv("PIPELINE_TH_WINRATE", "0.40"))
TH_WORST5     = float(os.getenv("PIPELINE_TH_WORST5", "-1.50"))

TOP_N_JSON = int(os.getenv("PIPELINE_TOP_N_JSON", "50"))
EPS = 1e-12

# ======================
# helpers
# ======================
def _safe_json_loads(x):
    if not isinstance(x, str):
        return {}
    s = x.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

def _stable_params_key(params: dict) -> str:
    try:
        return json.dumps(params or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(params)

def sortino(r: np.ndarray) -> float:
    if r.size == 0:
        return np.nan
    downside = r[r < 0]
    if downside.size == 0:
        return np.inf
    dstd = np.std(downside, ddof=1) if downside.size > 1 else 0.0
    if dstd <= 0:
        return np.inf if r.mean() > 0 else -np.inf
    return r.mean() / dstd

def max_drawdown(eq: pd.Series) -> float:
    eq = pd.to_numeric(eq, errors="coerce").dropna()
    if eq.empty:
        return np.nan
    roll = eq.cummax()
    dd = (eq - roll) / roll.replace(0, np.nan)
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    return float(dd.min()) if not dd.empty else np.nan

def profit_factor(pnl_net: pd.Series) -> float:
    x = pd.to_numeric(pnl_net, errors="coerce").dropna()
    if x.empty:
        return np.nan
    pos = x[x > 0].sum()
    neg = x[x < 0].sum()
    if abs(neg) < EPS:
        return np.inf if pos > 0 else np.nan
    return pos / abs(neg)

# ======================
# load
# ======================
if not os.path.exists(CSV_PATH):
    raise SystemExit(f"[analyze_pipeline] Missing CSV: {CSV_PATH}")

df_all = pd.read_csv(CSV_PATH)

if "type" not in df_all.columns:
    raise SystemExit("[analyze_pipeline] CSV missing column: type")

# Parse meta once
df_all["_meta"] = df_all.get("meta_json", "").apply(_safe_json_loads)

# ----------------------
# 🔑 BUILD ENTRY PARAMS MAP
# ----------------------
entry_params_by_trade = {}

entries = df_all[df_all["type"].astype(str).str.upper() == "ENTRY"]
for _, row in entries.iterrows():
    tid = row.get("trade_id")
    meta = row["_meta"]
    if isinstance(meta, dict):
        p = meta.get("params")
        if isinstance(p, dict) and p:
            entry_params_by_trade[tid] = p

# ----------------------
# EXIT only (as before)
# ----------------------
df = df_all[df_all["type"].astype(str).str.upper() == "EXIT"].copy()

if df.empty:
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    pd.DataFrame([]).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    print("Saved 0 candidates")
    raise SystemExit(0)

# Numeric coercion
for col in ("pnl_r", "pnl_net_est", "equity_after"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------
# 🔑 PARAM EXTRACTION WITH ENTRY FALLBACK
# ----------------------
def _extract_params_with_fallback(row) -> dict:
    meta = row["_meta"]
    if isinstance(meta, dict):
        p = meta.get("params")
        if isinstance(p, dict) and p:
            return p

    # fallback to ENTRY params
    tid = row.get("trade_id")
    return entry_params_by_trade.get(tid, {})

df["_params"] = df.apply(_extract_params_with_fallback, axis=1)
df["_group"]  = df["_params"].apply(_stable_params_key)

if df["_group"].nunique() <= 1 and df["_group"].iloc[0] == "{}":
    print("[WARN] Todos los trades quedaron agrupados en '{}'. Revisa inyección de params.")

# ======================
# ANALYSIS
# ======================
results = []

for g, d in df.groupby("_group"):
    trades = len(d)
    if trades < MIN_TRADES:
        continue

    r = d["pnl_r"].dropna().to_numpy(dtype=float)
    if r.size < MIN_R_OBS:
        continue

    expectancy = r.mean()
    s = sortino(r)
    pf = profit_factor(d["pnl_net_est"])
    dd = max_drawdown(d["equity_after"])
    winrate = (r > 0).mean()
    worst_5 = np.quantile(r, 0.05)

    passed = (
        expectancy > TH_EXPECTANCY and
        s >= TH_SORTINO and
        pf >= TH_PF and
        np.isfinite(dd) and dd > TH_DD and
        winrate >= TH_WINRATE and
        worst_5 > TH_WORST5
    )

    params_dict = d["_params"].iloc[0] if isinstance(d["_params"].iloc[0], dict) else {}

    results.append({
        "params": params_dict,
        "trades": trades,
        "expectancy_r": expectancy,
        "sortino": s,
        "profit_factor": pf,
        "max_dd": dd,
        "winrate": winrate,
        "worst_5pct_r": worst_5,
        "PASS_SHADOW": bool(passed),
        "_params_dict": params_dict,
    })

out = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)

if out.empty:
    out.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    print("Saved 0 candidates")
    raise SystemExit(0)

out = out.sort_values(
    by=["PASS_SHADOW", "expectancy_r", "sortino", "profit_factor"],
    ascending=[False, False, False, False],
)

out.drop(columns=["_params_dict"], errors="ignore").to_csv(OUT_CSV, index=False)

payload = []
for _, row in out[out["PASS_SHADOW"]].head(TOP_N_JSON).iterrows():
    payload.append({
        "params": row["_params_dict"],
        "metrics": {
            "trades": int(row["trades"]),
            "expectancy_r": float(row["expectancy_r"]),
            "sortino": float(row["sortino"]),
            "profit_factor": float(row["profit_factor"]),
            "max_dd": float(row["max_dd"]),
            "winrate": float(row["winrate"]),
            "worst_5pct_r": float(row["worst_5pct_r"]),
        }
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"Saved {len(out)} candidates -> {OUT_CSV}")
print(f"Saved {len(payload)} PASS_SHADOW -> {OUT_JSON}")



