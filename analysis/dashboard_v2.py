# analysis/dashboard_v2.py
import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from analysis_regimes import (
    load_rules,
    regime_metrics,
    regime_metrics_decayed,
    decide_status,
    hourly_metrics,
    hourly_status_map,
)

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Quant Shadow Dashboard V2", layout="wide")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT, "logs")
RULES_PATH = os.path.join(ROOT, "analysis", "regime_rules.yaml")

rules = load_rules(RULES_PATH)
DEFAULT_TF = rules.get("default_timeframe", "5m")
ROLL_N = int(rules.get("rolling", {}).get("window_trades", 50))

TZ_NAME = rules.get("timezone", "America/Argentina/Buenos_Aires")
H_MIN_TRADES = int(rules.get("hourly", {}).get("min_trades_per_hour", 5))
EXPORT_NAME = rules.get("hourly", {}).get("export_filename", "hourly_regime_flags.json")

# ✅ FASE 6 — Memory Decay
DECAY_CFG = rules.get("decay", {}) or {}
DECAY_ENABLED_CFG = bool(DECAY_CFG.get("enabled", False))
DECAY_HALF_LIFE = float(DECAY_CFG.get("half_life_hours", 72.0))
DECAY_MIN_EFF = float(DECAY_CFG.get("min_eff_trades", 10.0))

st.title("📊 Quant Shadow Dashboard V2 — Regimes, Hourly Edge & Kill/Promote")

# ============================================================
# Sidebar
# ============================================================
symbol = st.sidebar.text_input("Symbol", value="SOLUSDT").strip().upper()
tf = st.sidebar.selectbox(
    "Timeframe (view)", ["1m", "5m"], index=(1 if DEFAULT_TF == "5m" else 0)
)

st.sidebar.markdown("### 🧠 FASE 6 — Memory Decay")
use_decay = st.sidebar.checkbox(
    "Use decayed metrics (ESS)",
    value=DECAY_ENABLED_CFG,
    help="Aplica ponderación por recencia (exponential decay) y usa ESS (effective sample size)."
)

if use_decay:
    st.sidebar.caption(f"half-life={DECAY_HALF_LIFE:.1f}h | min_eff_trades={DECAY_MIN_EFF:.1f}")

trades_v2 = os.path.join(LOGS_DIR, f"{symbol}_shadow_trades_v2.csv")
bars_v2 = os.path.join(LOGS_DIR, f"{symbol}_shadow_bars_v2.csv")

if not os.path.exists(trades_v2) or not os.path.exists(bars_v2):
    st.warning(
        f"No encuentro logs v2 para {symbol}.\n\n"
        f"Esperados:\n- {trades_v2}\n- {bars_v2}\n\n"
        "Corré el bot con logger v2."
    )
    st.stop()

# ============================================================
# Loaders
# ============================================================
@st.cache_data(ttl=2)
def load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Preferimos local ISO si existe
    if "ts_local_iso" in df.columns:
        df["dt_local"] = pd.to_datetime(df["ts_local_iso"], errors="coerce")
    elif "timestamp" in df.columns:
        df["dt_local"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        df["dt_local"] = pd.NaT

    df = df.dropna(subset=["dt_local"]).sort_values("dt_local").set_index("dt_local")

    for col in ["open", "high", "low", "close", "volume", "regime", "atr", "atr_sma", "vwap", "inside_ratio", "range_invalid"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


@st.cache_data(ttl=2)
def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "ts_local_iso" in df.columns:
        df["dt_local"] = pd.to_datetime(df["ts_local_iso"], errors="coerce")
    elif "timestamp" in df.columns:
        df["dt_local"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        df["dt_local"] = pd.NaT

    df = df.dropna(subset=["dt_local"]).sort_values("dt_local")

    # si faltan, derivarlos
    if "hour_local" not in df.columns:
        df["hour_local"] = df["dt_local"].dt.hour
    if "weekday_local" not in df.columns:
        df["weekday_local"] = df["dt_local"].dt.day_name().str.slice(0, 3)

    return df


bars = load_bars(bars_v2)
trades = load_trades(trades_v2)

# ============================================================
# Bars view (solo visual)
# ============================================================
if tf == "5m":
    bars_view = bars.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "atr": "last",
        "atr_sma": "last",
        "vwap": "last",
        "regime": "last",
        "inside_ratio": "last",
        "range_invalid": "last",
    }).dropna(subset=["close"])
else:
    bars_view = bars.copy()

# ============================================================
# Trades (EXIT / ENTRY)
# ============================================================
exits = trades[trades.get("type", "") == "EXIT"].copy()
entries = trades[trades.get("type", "") == "ENTRY"].copy()

# coerce numerics
for col in ["pnl", "pnl_pct", "risk_usdt", "qty", "entry_price", "exit_price"]:
    if col in exits.columns:
        exits[col] = pd.to_numeric(exits[col], errors="coerce")

# pnl_r
if "pnl" in exits.columns and "risk_usdt" in exits.columns:
    exits["pnl_r"] = exits["pnl"] / exits["risk_usdt"]
    exits["pnl_r"] = exits["pnl_r"].replace([np.inf, -np.inf], np.nan)
else:
    exits["pnl_r"] = np.nan

exits["regime"] = exits.get("regime", "UNKNOWN").fillna("UNKNOWN")
entries["regime"] = entries.get("regime", "UNKNOWN").fillna("UNKNOWN")

exits_sorted = exits.sort_values("dt_local").copy()
exits_sorted["equity_r_total"] = exits_sorted["pnl_r"].fillna(0.0).cumsum()

# ============================================================
# Helpers
# ============================================================
def get_metrics(df: pd.DataFrame) -> dict:
    if use_decay:
        return regime_metrics_decayed(df, decay_cfg=DECAY_CFG)
    return regime_metrics(df)

def equity_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["hour_local", "trades", "sum_r", "avg_r", "winrate", "pf", "max_dd_r"])

    tmp = df.copy()
    tmp["hour_local"] = pd.to_numeric(tmp["hour_local"], errors="coerce")
    tmp = tmp.dropna(subset=["hour_local"])
    tmp["hour_local"] = tmp["hour_local"].astype(int)

    out = hourly_metrics(tmp)
    return out

# ============================================================
# Header metrics
# ============================================================
colA, colB, colC, colD = st.columns(4)

total_trades = len(exits_sorted)
total_pnl = float(exits_sorted["pnl"].sum()) if "pnl" in exits_sorted.columns else 0.0
total_r = float(exits_sorted["pnl_r"].sum(skipna=True)) if "pnl_r" in exits_sorted.columns else 0.0
winrate = float((exits_sorted["pnl"] > 0).mean()) if total_trades and "pnl" in exits_sorted.columns else np.nan

colA.metric("EXIT trades", total_trades)
colB.metric("PnL (sum)", f"{total_pnl:.4f}")
colC.metric("Sum R", f"{total_r:.2f}")
colD.metric("Winrate", f"{winrate*100:.1f}%" if total_trades else "—")

st.divider()

# ============================================================
# Equity curves
# ============================================================
st.subheader("📈 Equity (R)")

fig_total = px.line(
    exits_sorted, x="dt_local", y="equity_r_total",
    title="Equity Curve (R) — Total"
)
st.plotly_chart(fig_total, use_container_width=True)

regimes = ["TREND", "RANGE"]
cols = st.columns(2)
for i, rg in enumerate(regimes):
    ex_rg = exits_sorted[exits_sorted["regime"] == rg].copy()
    ex_rg["equity_r"] = ex_rg["pnl_r"].fillna(0.0).cumsum()
    fig = px.line(ex_rg, x="dt_local", y="equity_r", title=f"Equity Curve (R) — {rg}")
    cols[i].plotly_chart(fig, use_container_width=True)

# ============================================================
# 🧬 FASE 7 — Correlation Guard (diagnóstico multi-símbolo)
# ============================================================
st.divider()
st.subheader("🧬 FASE 7 — Correlation Guard (diagnóstico)")

# Intentamos importar sin asumir estructura de paquete perfecta
corr_available = False
compute_symbol_correlations = None
correlation_penalty_matrix = None

try:
    # Caso típico: analysis es paquete (tiene __init__.py)
    from analysis.correlation_guard import compute_symbol_correlations, correlation_penalty_matrix
    corr_available = True
except Exception:
    try:
        # Caso alternativo: import directo (por sys.path / cwd)
        from correlation_guard import compute_symbol_correlations, correlation_penalty_matrix
        corr_available = True
    except Exception:
        corr_available = False

if not corr_available:
    st.info(
        "Correlation Guard no disponible.\n"
        "Crear `analysis/correlation_guard.py` para habilitar FASE 7."
    )
else:
    try:
        trade_files = [
            f for f in os.listdir(LOGS_DIR)
            if f.endswith("_shadow_trades_v2.csv")
        ]
    except Exception as e:
        trade_files = []
        st.error(f"No pude listar LOGS_DIR ({LOGS_DIR}): {e}")

    if len(trade_files) < 2:
        st.info("Se requieren al menos 2 símbolos (logs *_shadow_trades_v2.csv) para análisis de correlación.")
    else:
        pnl_map = {}

        for fname in trade_files:
            sym = fname.replace("_shadow_trades_v2.csv", "")
            path = os.path.join(LOGS_DIR, fname)

            try:
                df = pd.read_csv(path)
            except Exception:
                continue

            # si el csv no tiene pnl_r (por ejemplo logs viejos), lo derivamos si se puede
            if "pnl_r" not in df.columns:
                if "pnl" in df.columns and "risk_usdt" in df.columns:
                    pnl = pd.to_numeric(df["pnl"], errors="coerce")
                    risk = pd.to_numeric(df["risk_usdt"], errors="coerce")
                    with np.errstate(divide="ignore", invalid="ignore"):
                        df["pnl_r"] = pnl / risk
                    df["pnl_r"] = df["pnl_r"].replace([np.inf, -np.inf], np.nan)

            if "pnl_r" not in df.columns:
                continue

            s = pd.to_numeric(df["pnl_r"], errors="coerce").fillna(0.0)
            pnl_map[sym] = s.reset_index(drop=True)

        if len(pnl_map) < 2:
            st.warning("No hay suficientes símbolos con pnl_r válido para calcular correlación.")
        else:
            corr_df = compute_symbol_correlations(pnl_map)
            penalty_df = correlation_penalty_matrix(corr_df)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### 🔗 Correlation Matrix (pnl_r)")
                st.dataframe(corr_df, use_container_width=True)

            with c2:
                st.markdown("### ⚠️ Correlation Penalty Matrix")
                st.dataframe(penalty_df, use_container_width=True)

            st.plotly_chart(
                px.imshow(
                    corr_df,
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlation Heatmap — pnl_r",
                ),
                use_container_width=True,
            )

            st.caption(
                "FASE 7 es diagnóstica.\n"
                "La penalización será aplicada por `allocator.py` "
                "en la toma de decisiones (shadow/live)."
            )

st.divider()

# ============================================================
# 🥇 FASE 1 — Equity por hora (GLOBAL)
# ============================================================
st.subheader("🕒 FASE 1 — Equity por hora del día (Global, EXIT-only)")

if len(exits_sorted) < 5:
    st.info("Muy pocos EXIT trades aún para análisis horario.")
else:
    hourly_global = equity_by_hour(exits_sorted)
    c1, c2 = st.columns(2)

    with c1:
        st.dataframe(hourly_global, use_container_width=True, hide_index=True)

    with c2:
        fig = px.bar(hourly_global, x="hour_local", y="sum_r", title="Sum R por hora (GLOBAL)")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# 🥈 FASE 2 — Equity por hora × Régimen
# ============================================================
st.subheader("🧭 FASE 2 — Equity por hora × Régimen")

for rg in regimes:
    st.markdown(f"### {rg}")
    ex_rg = exits_sorted[exits_sorted["regime"] == rg].copy()

    if len(ex_rg) < 5:
        st.info(f"{rg}: todavía no hay suficientes EXIT trades.")
        continue

    hourly_rg = equity_by_hour(ex_rg)
    c1, c2 = st.columns(2)

    with c1:
        st.dataframe(hourly_rg, use_container_width=True, hide_index=True)

    with c2:
        fig = px.bar(hourly_rg, x="hour_local", y="sum_r", title=f"Sum R por hora — {rg}")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# Kill / Promote (rolling) — ahora soporta FASE 6 decay
# ============================================================
st.subheader("🧠 Kill / Promote (rolling)")

status_rows = []
for rg in regimes:
    rg_rules = rules["kill_promote"][rg]
    ex_rg = exits_sorted[exits_sorted["regime"] == rg].tail(ROLL_N)

    m = get_metrics(ex_rg)
    status = decide_status(m, rg_rules)

    row = {
        "regime": rg,
        "status": status,
        "trades": m.get("trades", 0),
        "winrate": m.get("winrate", np.nan),
        "expectancy_r": m.get("expectancy_r", np.nan),
        "profit_factor": m.get("pf", np.nan),
        "max_dd_r": m.get("max_dd_r", np.nan),
        "sum_r": m.get("sum_r", 0.0),
        "rolling_window_trades": ROLL_N,
    }

    if use_decay:
        row.update({
            "eff_trades": m.get("eff_trades", np.nan),
            "eff_mass": m.get("eff_mass", np.nan),
            "half_life_h": m.get("half_life_hours", DECAY_HALF_LIFE),
        })

    status_rows.append(row)

st.dataframe(pd.DataFrame(status_rows), use_container_width=True)

st.divider()

# ============================================================
# 🥉 FASE 3 — Kill horario automático + Export flags + Simulador
#    + ✅ FASE 6: hourly_status_map con decay opcional
# ============================================================
st.subheader("🧨 FASE 3 — Kill horario automático (por régimen) + Export para el bot")

if len(exits_sorted) < 5:
    st.info("Muy pocos EXIT trades aún. FASE 3 necesita más muestra para sacar KILL por hora.")
else:
    hourly_flags = {
        "timezone": TZ_NAME,
        "symbol": symbol,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "min_trades_per_hour": H_MIN_TRADES,
        "decay": {
            "enabled": bool(use_decay),
            "half_life_hours": float(DECAY_HALF_LIFE),
            "min_eff_trades": float(DECAY_MIN_EFF),
        },
        "regimes": {}
    }

    # Panel por régimen
    for rg in regimes:
        rg_rules = rules["kill_promote"][rg]
        ex_rg = exits_sorted[exits_sorted["regime"] == rg].copy()

        hours_map, summary = hourly_status_map(
            exits_df=ex_rg,
            rules_regime=rg_rules,
            min_trades_hour=H_MIN_TRADES,
            use_decay=use_decay,
            decay_cfg=DECAY_CFG,
        )

        hourly_flags["regimes"][rg] = {
            "killed_hours": summary["killed_hours"],
            "promoted_hours": summary["promoted_hours"],
            "active_hours": summary["active_hours"],
            "insufficient_hours": summary["insufficient_hours"],
            "hours": {str(h): hours_map[h] for h in range(24)},
        }

        with st.expander(f"{rg} — KILL/PROMOTE por hora (min trades/h={H_MIN_TRADES})", expanded=(rg == "TREND")):
            cA, cB, cC, cD = st.columns(4)
            cA.metric("KILLED hours", len(summary["killed_hours"]))
            cB.metric("PROMOTED hours", len(summary["promoted_hours"]))
            cC.metric("ACTIVE hours", len(summary["active_hours"]))
            cD.metric("INSUFF hours", len(summary["insufficient_hours"]))

            st.write("**KILLED:**", summary["killed_hours"])
            st.write("**PROMOTED:**", summary["promoted_hours"])

            rows = []
            for hr in range(24):
                stt = hours_map[hr]["status"]
                m = hours_map[hr]["metrics"]
                row = {
                    "hour": hr,
                    "status": stt,
                    "trades": int(m.get("trades", 0)),
                    "sum_r": float(m.get("sum_r", 0.0)),
                    "expectancy_r": float(m.get("expectancy_r", 0.0)) if m.get("expectancy_r") is not None else 0.0,
                    "pf": float(m.get("pf", 0.0)) if m.get("pf") is not None else 0.0,
                    "max_dd_r": float(m.get("max_dd_r", 0.0)) if m.get("max_dd_r") is not None else 0.0,
                }
                if use_decay:
                    row["eff_trades"] = float(m.get("eff_trades", np.nan))
                    row["eff_mass"] = float(m.get("eff_mass", np.nan))
                rows.append(row)

            dfh = pd.DataFrame(rows)
            st.dataframe(dfh, use_container_width=True, hide_index=True)

            fig = px.bar(dfh, x="hour", y="sum_r", title=f"{rg} — Sum R por hora (EXIT)")
            st.plotly_chart(fig, use_container_width=True)

    # Export automático
    export_path = os.path.join(LOGS_DIR, f"{symbol}_{EXPORT_NAME}")
    try:
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(hourly_flags, f, ensure_ascii=False, indent=2)
        st.success(f"Flags exportados para el bot: {export_path}")
    except Exception as e:
        st.error(f"No pude exportar flags JSON: {e}")

    # ----------------------------
    # Simulador horario
    # ----------------------------
    st.markdown("### 🧪 Simulador horario — “¿qué pasa si apago X régimen de HH–HH?”")

    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 2])

    with sim_col1:
        sim_regime = st.selectbox("Regimen a filtrar", regimes, index=1)  # default RANGE
    with sim_col2:
        start_h = st.number_input("Desde hora", min_value=0, max_value=23, value=3, step=1)
        end_h = st.number_input("Hasta hora (inclusive)", min_value=0, max_value=23, value=6, step=1)
    with sim_col3:
        st.caption("Proxy: simula equity removiendo EXITs de ese régimen en ese rango horario. No re-simula entradas.")

    if start_h <= end_h:
        hours_block = set(range(int(start_h), int(end_h) + 1))
    else:
        hours_block = set(list(range(int(start_h), 24)) + list(range(0, int(end_h) + 1)))

    base = exits_sorted.copy()
    base["equity_base"] = base["pnl_r"].fillna(0.0).cumsum()

    filt = exits_sorted.copy()
    filt["hour_local"] = pd.to_numeric(filt["hour_local"], errors="coerce").fillna(-1).astype(int)

    mask_block = (filt["regime"] == sim_regime) & (filt["hour_local"].isin(list(hours_block)))
    filt2 = filt[~mask_block].copy()
    filt2["equity_filt"] = filt2["pnl_r"].fillna(0.0).cumsum()

    cS1, cS2, cS3 = st.columns(3)
    cS1.metric("Base Sum R", f"{float(base['pnl_r'].fillna(0.0).sum()):.2f}")
    cS2.metric("Filtrado Sum R", f"{float(filt2['pnl_r'].fillna(0.0).sum()):.2f}")
    cS3.metric("EXITs removidos", int(mask_block.sum()))

    plot_df = base[["dt_local", "equity_base"]].copy()
    plot_df = plot_df.rename(columns={"equity_base": "equity"})
    plot_df["serie"] = "BASE"

    plot_df2 = filt2[["dt_local", "equity_filt"]].copy()
    plot_df2 = plot_df2.rename(columns={"equity_filt": "equity"})
    plot_df2["serie"] = f"FILTRADO ({sim_regime} off {sorted(hours_block)})"

    comb = pd.concat([plot_df, plot_df2], ignore_index=True).sort_values("dt_local")
    fig_sim = px.line(comb, x="dt_local", y="equity", color="serie", title="Simulador (proxy) — Equity R base vs filtrado")
    st.plotly_chart(fig_sim, use_container_width=True)

st.divider()

# ============================================================
# Heatmaps
# ============================================================
st.subheader("🕒 Heatmaps horarios (local AR)")

def _heatmap(df, value_col, title):
    tmp = df.copy()
    tmp["hour_local"] = pd.to_numeric(tmp.get("hour_local"), errors="coerce")
    tmp = tmp.dropna(subset=["hour_local"])
    tmp["hour_local"] = tmp["hour_local"].astype(int)

    pivot = tmp.pivot_table(
        index="weekday_local",
        columns="hour_local",
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    )

    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = pivot.reindex([d for d in order if d in pivot.index])

    return px.imshow(pivot, aspect="auto", title=title)

hm_cols = st.columns(2)

if len(exits_sorted) and "pnl" in exits_sorted.columns:
    hm_cols[0].plotly_chart(
        _heatmap(exits_sorted, "pnl", "Heatmap — PnL (EXIT) por weekday/hora"),
        use_container_width=True
    )
else:
    hm_cols[0].info("Sin EXIT trades aún o falta columna pnl.")

if len(entries):
    tmp_entries = entries.copy()
    tmp_entries["cnt"] = 1
    hm_cols[1].plotly_chart(
        _heatmap(tmp_entries, "cnt", "Heatmap — Count (ENTRY) por weekday/hora"),
        use_container_width=True
    )
else:
    hm_cols[1].info("Sin ENTRY trades aún.")

st.divider()

# ============================================================
# Table
# ============================================================
st.subheader("📄 Últimos EXIT trades")

preferred_cols = [
    "ts_local_iso", "dt_local", "regime", "side", "qty",
    "entry_price", "exit_price", "pnl", "pnl_pct", "risk_usdt", "pnl_r", "reason"
]
show_cols = [c for c in preferred_cols if c in exits_sorted.columns]

st.dataframe(
    exits_sorted[show_cols].tail(200) if show_cols else exits_sorted.tail(200),
    use_container_width=True
)
