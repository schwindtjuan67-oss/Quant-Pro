# analysis/correlation_guard.py
from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List


# ============================================================
# Helpers internos (robustez institucional)
# ============================================================
_TIME_COL_CANDIDATES: List[str] = [
    "dt_local",
    "timestamp",
    "ts",
    "datetime",
    "date",
]


def _resolve_time_col(df: pd.DataFrame) -> Optional[str]:
    """
    Intenta resolver una columna temporal usable.
    Prioriza dt_local, luego fallbacks comunes.
    """
    for c in _TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _safe_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corr más robusta:
    - fuerza float
    - maneja NaNs de forma pairwise
    """
    if df is None or df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    try:
        return df.astype(float).corr()
    except Exception:
        return pd.DataFrame()


# ============================================================
# Core builders
# ============================================================
def build_r_matrix(
    exits_df: pd.DataFrame,
    symbol_col: str = "symbol",
    value_col: str = "pnl_r",
    window: int = 100,
) -> pd.DataFrame:
    """
    Construye matriz wide:
      index = tiempo (dt_local o fallback)
      columns = symbol
      values = pnl_r

    Usa últimos `window` EXIT trades por símbolo.

   ⚠️ Nota institucional:
    Para runtime (FASE 8/9) usar PortfolioState.correlation_df().
    Esto es analysis / dashboard / offline.
    """
    if exits_df is None or exits_df.empty:
        return pd.DataFrame()

    if symbol_col not in exits_df.columns or value_col not in exits_df.columns:
        return pd.DataFrame()

    time_col = _resolve_time_col(exits_df)
    if time_col is None:
        warnings.warn(
            "[correlation_guard] No se encontró columna temporal "
            "(dt_local/timestamp/ts/...). "
            "Usando índice relativo por orden de trades.",
            RuntimeWarning,
        )

    frames = []
    for sym, g in exits_df.groupby(symbol_col):
        g = g.sort_values(time_col) if time_col else g
        g = g.tail(window)
        if g.empty:
            continue

        # Index robusto:
        # - si hay time_col, usamos esa columna
        # - si no, índice relativo (0..n)
        idx = g[time_col].values if time_col else range(len(g))

        frames.append(
            pd.DataFrame(
                {sym: g[value_col].astype(float).values},
                index=idx,
            )
        )

    if not frames:
        return pd.DataFrame()

    # Concatenamos por columnas.
    # Si los índices no alinean, quedarán NaNs (esperable).
    return pd.concat(frames, axis=1)


def correlation_matrix(r_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula correlación robusta desde r_matrix.
    """
    return _safe_corr(r_matrix)


def correlation_alerts(
    corr: pd.DataFrame,
    threshold: float = 0.85,
) -> Dict[Tuple[str, str], float]:
    """
    Devuelve pares (symA, symB) con |corr| >= threshold.
    """
    alerts: Dict[Tuple[str, str], float] = {}
    if corr is None or corr.empty:
        return alerts

    syms = corr.columns.tolist()
    for i, a in enumerate(syms):
        for b in syms[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= threshold:
                alerts[(a, b)] = float(v)

    return alerts


# ============================================================
# 🔌 Dashboard / Analysis adapters (FASE 7)
# ============================================================
def compute_symbol_correlations(
    pnl_map: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Adapter simple para dashboard / análisis.

    Espera:
      pnl_map = { "SOLUSDT": Series(pnl_r), "BTCUSDT": Series(pnl_r), ... }

    ⚠️ No usar en runtime industrial.
    """
    if not pnl_map or len(pnl_map) < 2:
        return pd.DataFrame()

    try:
        df = pd.DataFrame({
            sym: s.astype(float)
            for sym, s in pnl_map.items()
            if s is not None and len(s) > 0
        })
    except Exception:
        return pd.DataFrame()

    return _safe_corr(df)


def correlation_penalty_matrix(
    corr: pd.DataFrame,
    threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Devuelve una matriz de penalización [0..1]

    0 = sin penalización  
    1 = |corr| >= threshold
    """
    if corr is None or corr.empty:
        return pd.DataFrame()

    penalty = corr.copy().astype(float)

    for c in penalty.columns:
        for r in penalty.index:
            v = penalty.loc[r, c]
            if pd.isna(v):
                penalty.loc[r, c] = 0.0
            else:
                penalty.loc[r, c] = max(
                    0.0,
                    min(1.0, abs(v) / float(threshold))
                )

    return penalty
