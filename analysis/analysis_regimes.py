# analysis/analysis_regimes.py
from __future__ import annotations

import math
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


def load_rules(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _profit_factor(pnl_r: pd.Series, weights: Optional[pd.Series] = None) -> float:
    pnl_r = pd.to_numeric(pnl_r, errors="coerce").fillna(0.0)

    if weights is None:
        gains = pnl_r[pnl_r > 0].sum()
        losses = -pnl_r[pnl_r < 0].sum()
    else:
        w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
        gains = (pnl_r.clip(lower=0.0) * w).sum()
        losses = (-pnl_r.clip(upper=0.0) * w).sum()

    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    m = x.notna()
    x = x[m]
    w = w[m]
    s = float(w.sum())
    if s <= 0:
        return float("nan")
    return float((x * w).sum() / s)


def _equity_and_maxdd(pnl_r: pd.Series, weights: Optional[pd.Series] = None) -> Tuple[pd.Series, float]:
    pnl_r = pd.to_numeric(pnl_r, errors="coerce").fillna(0.0)

    if weights is not None:
        w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
        pnl_r = pnl_r * w

    equity = pnl_r.cumsum()
    dd = equity - equity.cummax()
    max_dd = float(dd.min()) if len(dd) else 0.0
    return equity, max_dd


def _effective_sample_size(weights: pd.Series) -> float:
    """
    ESS = (sum w)^2 / sum(w^2)
    (más informativo que sum(w) para "masa efectiva" cuando hay decay fuerte)
    """
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    sw = float(w.sum())
    sw2 = float((w * w).sum())
    if sw <= 0 or sw2 <= 0:
        return 0.0
    return float((sw * sw) / (sw2 + 1e-12))


def compute_decay_weights(
    dt_local: pd.Series,
    half_life_hours: float = 72.0,
    now: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Exponential decay weights:
      w = 0.5 ** (age_hours / half_life_hours)

    - dt_local puede ser naive (local) o tz-aware.
    - now se ajusta al "tipo" (naive/tz-aware) de dt_local.
    """
    if now is None:
        now = pd.Timestamp.utcnow()

    dt = pd.to_datetime(dt_local, errors="coerce", utc=False)

    # Si dt es tz-aware, alineamos now a esa tz
    if getattr(dt.dt, "tz", None) is not None:
        if now.tzinfo is None:
            now = now.tz_localize(dt.dt.tz)
        else:
            now = now.tz_convert(dt.dt.tz)
    else:
        # dt naive => now naive
        if now.tzinfo is not None:
            now = now.tz_convert(None)

    age = (now - dt).dt.total_seconds() / 3600.0
    age = age.clip(lower=0.0)
    hl = max(1e-9, float(half_life_hours))
    w = 0.5 ** (age / hl)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return w


def regime_metrics(trades: pd.DataFrame) -> dict:
    """
    Métricas "clásicas" (sin decay). Requiere pnl_r.
    """
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "winrate": np.nan,
            "expectancy_r": np.nan,
            "pf": np.nan,
            "max_dd_r": np.nan,
            "sum_r": 0.0,
        }

    pnl_r = pd.to_numeric(trades["pnl_r"], errors="coerce").fillna(0.0)
    _, max_dd_r = _equity_and_maxdd(pnl_r)

    return {
        "trades": int(len(trades)),
        "winrate": float((pnl_r > 0).mean()) if len(pnl_r) else np.nan,
        "expectancy_r": float(pnl_r.mean()) if len(pnl_r) else np.nan,
        "pf": _profit_factor(pnl_r),
        "max_dd_r": max_dd_r,
        "sum_r": float(pnl_r.sum()),
    }


def regime_metrics_decayed(trades: pd.DataFrame, decay_cfg: Dict[str, Any]) -> dict:
    """
    Métricas "decayed" (ponderadas por recencia).
    Requiere: columns pnl_r y dt_local (o ts_local_iso convertible).
    """
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "eff_trades": 0.0,
            "eff_mass": 0.0,
            "winrate": np.nan,
            "expectancy_r": np.nan,
            "pf": np.nan,
            "max_dd_r": np.nan,
            "sum_r": 0.0,
            "min_eff_trades": float(decay_cfg.get("min_eff_trades", 10.0)),
        }

    half_life_hours = float(decay_cfg.get("half_life_hours", 72.0))
    min_eff_trades = float(decay_cfg.get("min_eff_trades", 10.0))

    dt_col = None
    if "dt_local" in trades.columns:
        dt_col = "dt_local"
    elif "ts_local_iso" in trades.columns:
        dt_col = "ts_local_iso"

    if dt_col is None:
        # fallback: sin decay real
        m = regime_metrics(trades)
        m["eff_trades"] = float(m["trades"])
        m["eff_mass"] = float(m["trades"])
        m["min_eff_trades"] = float(min_eff_trades)
        return m

    w = compute_decay_weights(trades[dt_col], half_life_hours=half_life_hours)

    pnl_r = pd.to_numeric(trades["pnl_r"], errors="coerce").fillna(0.0)

    # dos nociones útiles:
    eff_mass = float(w.sum())              # "masa" ponderada
    eff_trades = float(_effective_sample_size(w))  # ESS (recomendado)

    _, max_dd_r = _equity_and_maxdd(pnl_r, weights=w)

    winrate = _weighted_mean((pnl_r > 0).astype(float), w)
    expectancy = _weighted_mean(pnl_r, w)
    pf = _profit_factor(pnl_r, weights=w)
    sum_r = float((pnl_r * w).sum())

    return {
        "trades": int(len(trades)),
        "eff_trades": eff_trades,
        "eff_mass": eff_mass,
        "winrate": float(winrate),
        "expectancy_r": float(expectancy),
        "pf": float(pf),
        "max_dd_r": float(max_dd_r),
        "sum_r": float(sum_r),
        "min_eff_trades": float(min_eff_trades),
        "half_life_hours": float(half_life_hours),
    }


def decide_status(metrics: dict, rules_regime: dict) -> str:
    """
    Compatible con métricas clásicas o decayed.
    - Si viene eff_trades, se usa como filtro adicional.
    """
    min_trades = int(rules_regime.get("min_trades", 0))
    if int(metrics.get("trades", 0)) < min_trades:
        return "INSUFFICIENT_DATA"

    # si estamos en modo decay, exigimos "masa efectiva"
    if "eff_trades" in metrics:
        min_eff = float(metrics.get("min_eff_trades", rules_regime.get("min_eff_trades", 0.0)))
        if float(metrics.get("eff_trades", 0.0)) < float(min_eff):
            return "INSUFFICIENT_DATA"

    promote = rules_regime["promote"]
    kill = rules_regime["kill"]

    # kill primero
    if float(metrics["expectancy_r"]) <= float(kill["expectancy_r_max"]):
        return "KILLED"
    if float(metrics["max_dd_r"]) <= float(kill["max_dd_r_max"]):
        return "KILLED"

    # promote si cumple todo
    if (
        float(metrics["expectancy_r"]) >= float(promote["expectancy_r_min"])
        and float(metrics["pf"]) >= float(promote["profit_factor_min"])
        and float(metrics["max_dd_r"]) >= float(promote["max_dd_r_min"])
    ):
        return "PROMOTED"

    return "ACTIVE"


# ============================================================
# HOURLY — métricas por hora (para dashboard FASE 1/2/3/6)
# ============================================================
def hourly_metrics(exits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve tabla por hour_local con métricas clásicas:
      hour_local, trades, sum_r, expectancy_r, winrate, pf, max_dd_r

    Requiere columnas:
      - hour_local (0..23)
      - pnl_r
      - opcional pnl (solo para winrate si pnl_r existe ya funciona igual)
    """
    if exits_df is None or exits_df.empty:
        return pd.DataFrame(columns=["hour_local", "trades", "sum_r", "expectancy_r", "winrate", "pf", "max_dd_r"])

    tmp = exits_df.copy()

    if "hour_local" not in tmp.columns:
        # intentamos derivar desde dt_local si existe
        if "dt_local" in tmp.columns:
            tmp["hour_local"] = pd.to_datetime(tmp["dt_local"], errors="coerce").dt.hour
        elif "ts_local_iso" in tmp.columns:
            tmp["hour_local"] = pd.to_datetime(tmp["ts_local_iso"], errors="coerce").dt.hour
        else:
            tmp["hour_local"] = np.nan

    tmp["hour_local"] = pd.to_numeric(tmp["hour_local"], errors="coerce")
    tmp = tmp.dropna(subset=["hour_local"])
    tmp["hour_local"] = tmp["hour_local"].astype(int)
    tmp = tmp[(tmp["hour_local"] >= 0) & (tmp["hour_local"] <= 23)]

    if "pnl_r" not in tmp.columns:
        return pd.DataFrame(columns=["hour_local", "trades", "sum_r", "expectancy_r", "winrate", "pf", "max_dd_r"])

    rows = []
    for hr in range(24):
        hdf = tmp[tmp["hour_local"] == hr]
        m = regime_metrics(hdf)
        rows.append({
            "hour_local": hr,
            "trades": int(m["trades"]),
            "sum_r": float(m["sum_r"]) if m["sum_r"] is not None else 0.0,
            "expectancy_r": float(m["expectancy_r"]) if m["expectancy_r"] is not None else np.nan,
            "winrate": float(m["winrate"]) if m["winrate"] is not None else np.nan,
            "pf": float(m["pf"]) if m["pf"] is not None else np.nan,
            "max_dd_r": float(m["max_dd_r"]) if m["max_dd_r"] is not None else 0.0,
        })

    return pd.DataFrame(rows)


def hourly_status_map(
    exits_df: pd.DataFrame,
    rules_regime: Dict[str, Any],
    min_trades_hour: int = 5,
    use_decay: bool = False,
    decay_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Construye mapa por hora -> status + metrics.
    Status por hora usa:
      - min_trades_hour (en vez de rules_regime.min_trades)
      - decide_status con thresholds del régimen (promote/kill)

    Retorna:
      hours_map[hour] = {"status": str, "metrics": dict}
      summary = {"killed_hours":[], "promoted_hours":[], "active_hours":[], "insufficient_hours":[]}
    """
    if decay_cfg is None:
        decay_cfg = {}

    hours_map: Dict[int, Dict[str, Any]] = {}

    summary = {
        "killed_hours": [],
        "promoted_hours": [],
        "active_hours": [],
        "insufficient_hours": [],
    }

    if exits_df is None or exits_df.empty:
        for hr in range(24):
            hours_map[hr] = {"status": "INSUFFICIENT_DATA", "metrics": {"trades": 0}}
            summary["insufficient_hours"].append(hr)
        return hours_map, summary

    tmp = exits_df.copy()

    # asegurar hour_local
    if "hour_local" not in tmp.columns:
        if "dt_local" in tmp.columns:
            tmp["hour_local"] = pd.to_datetime(tmp["dt_local"], errors="coerce").dt.hour
        elif "ts_local_iso" in tmp.columns:
            tmp["hour_local"] = pd.to_datetime(tmp["ts_local_iso"], errors="coerce").dt.hour
        else:
            tmp["hour_local"] = np.nan

    tmp["hour_local"] = pd.to_numeric(tmp["hour_local"], errors="coerce")
    tmp = tmp.dropna(subset=["hour_local"])
    tmp["hour_local"] = tmp["hour_local"].astype(int)
    tmp = tmp[(tmp["hour_local"] >= 0) & (tmp["hour_local"] <= 23)]

    # reglas: copiamos y forzamos min_trades=min_trades_hour para evaluación por hora
    rules_hour = dict(rules_regime)
    rules_hour["min_trades"] = int(min_trades_hour)

    for hr in range(24):
        hdf = tmp[tmp["hour_local"] == hr]

        if use_decay:
            m = regime_metrics_decayed(hdf, decay_cfg=decay_cfg)
        else:
            m = regime_metrics(hdf)

        status = decide_status(m, rules_hour)

        hours_map[hr] = {"status": status, "metrics": m}

        if status == "KILLED":
            summary["killed_hours"].append(hr)
        elif status == "PROMOTED":
            summary["promoted_hours"].append(hr)
        elif status == "ACTIVE":
            summary["active_hours"].append(hr)
        else:
            summary["insufficient_hours"].append(hr)

    return hours_map, summary
