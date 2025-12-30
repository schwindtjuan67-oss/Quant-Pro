# analysis/portfolio_state.py
from __future__ import annotations

import pandas as pd
from collections import defaultdict, deque
from typing import Dict, Deque, List, Optional, Tuple


class PortfolioState:
    """
    Estado único del portfolio (FASE 8 industrial)

    Responsabilidades:
    - mantener pnl_r recientes por símbolo (rolling window)
    - construir matriz de correlación en memoria (sin CSV)
    - exponer helpers para correlation guard / allocator
    - trackear posiciones abiertas
    """

    def __init__(
        self,
        window: int = 100,
        *,
        min_samples_per_symbol: int = 5,
        min_symbols: int = 2,
        # Robustez: mínimo de "co-samples" para calcular corr pairwise
        min_corr_periods: int = 5,
    ):
        self.window = int(window)
        self.min_samples_per_symbol = int(min_samples_per_symbol)
        self.min_symbols = int(min_symbols)
        self.min_corr_periods = int(min_corr_periods)

        self.pnl_r: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

        # estado simple de posiciones (FASE 9 ready)
        self.open_positions: Dict[str, bool] = {}

        # debug/meta (no rompe nada)
        self._last_corr_meta: Dict[str, object] = {}

    # =========================================================
    # Updates desde runtime
    # =========================================================
    def on_trade_close(self, symbol: str, pnl_r: float):
        """
        Llamar en EXIT.
        pnl_r = PnL normalizado por riesgo (R-multiple).
        """
        sym = str(symbol).upper().strip()
        if not sym:
            return
        try:
            self.pnl_r[sym].append(float(pnl_r))
        except Exception:
            pass

    def set_position_state(self, symbol: str, open_: bool):
        """
        Track simple de posiciones abiertas.
        """
        sym = str(symbol).upper().strip()
        if not sym:
            return
        self.open_positions[sym] = bool(open_)

    # =========================================================
    # Core: Correlation
    # =========================================================
    def correlation_df(self) -> Optional[pd.DataFrame]:
        """
        Devuelve DataFrame de correlación pnl_r entre símbolos.
        Retorna None si no hay datos suficientes.

        Robustez:
        - exige min_samples_per_symbol por columna
        - usa min_corr_periods en corr (pairwise)
        - guarda meta de samples para trazabilidad/debug
        """
        # 1) Quick gate
        if len(self.pnl_r) < self.min_symbols:
            self._last_corr_meta = {
                "ok": False,
                "reason": "not_enough_symbols_total",
                "symbols_total": len(self.pnl_r),
            }
            return None

        # 2) Filtrar símbolos con suficientes samples
        data = {
            sym: list(vals)
            for sym, vals in self.pnl_r.items()
            if len(vals) >= self.min_samples_per_symbol
        }

        if len(data) < self.min_symbols:
            self._last_corr_meta = {
                "ok": False,
                "reason": "not_enough_symbols_with_samples",
                "symbols_total": len(self.pnl_r),
                "symbols_with_samples": len(data),
                "min_samples_per_symbol": self.min_samples_per_symbol,
                "samples_by_symbol": {k: len(v) for k, v in self.pnl_r.items()},
            }
            return None

        # 3) DataFrame (puede tener NaNs por longitudes distintas)
        df = pd.DataFrame(data)

        if df.shape[1] < self.min_symbols:
            self._last_corr_meta = {
                "ok": False,
                "reason": "df_not_enough_columns",
                "df_cols": int(df.shape[1]),
            }
            return None

        # 4) Corr robusta (pairwise + min_periods)
        try:
            corr = df.corr(min_periods=max(1, int(self.min_corr_periods)))
        except Exception:
            self._last_corr_meta = {
                "ok": False,
                "reason": "corr_exception",
            }
            return None

        if corr is None or corr.empty:
            self._last_corr_meta = {
                "ok": False,
                "reason": "corr_empty",
            }
            return None

        # 5) Meta para trazabilidad
        self._last_corr_meta = {
            "ok": True,
            "symbols_used": list(df.columns),
            "samples_used_by_symbol": {c: int(df[c].count()) for c in df.columns},
            "min_corr_periods": int(self.min_corr_periods),
            "min_samples_per_symbol": int(self.min_samples_per_symbol),
            "window": int(self.window),
        }

        return corr

    # =========================================================
    # Correlation Guard helpers (FASE 8)
    # =========================================================
    def correlation_alerts(
        self,
        threshold: float = 0.85,
    ) -> Dict[Tuple[str, str], float]:
        """
        Devuelve pares (symA, symB) con |corr| >= threshold
        """
        corr = self.correlation_df()
        alerts: Dict[Tuple[str, str], float] = {}

        if corr is None or corr.empty:
            return alerts

        syms = list(corr.columns)
        for i, a in enumerate(syms):
            for b in syms[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v) and abs(float(v)) >= float(threshold):
                    alerts[(a, b)] = float(v)

        return alerts

    def is_hard_block(
        self,
        a: str,
        b: str,
        hard_ge: float = 0.98,
    ) -> bool:
        """
        Hard correlation block helper.
        """
        corr = self.correlation_df()
        if corr is None:
            return False

        a = str(a).upper().strip()
        b = str(b).upper().strip()

        if a not in corr.columns or b not in corr.columns:
            return False

        v = corr.loc[a, b]
        if pd.isna(v):
            return False

        return abs(float(v)) >= float(hard_ge)

    # =========================================================
    # Debug / Dashboard helpers
    # =========================================================
    def correlation_penalty_matrix(
        self,
        threshold: float = 0.85,
    ) -> Optional[pd.DataFrame]:
        """
        Matriz [0..1] de penalización por correlación.
        0 = sin penalización
        1 = |corr| >= threshold
        """
        corr = self.correlation_df()
        if corr is None or corr.empty:
            return None

        penalty = corr.copy()
        for c in penalty.columns:
            for r in penalty.index:
                v = penalty.loc[r, c]
                if pd.isna(v):
                    penalty.loc[r, c] = 0.0
                else:
                    penalty.loc[r, c] = max(
                        0.0,
                        min(1.0, abs(float(v)) / float(threshold)),
                    )
        return penalty

    # =========================================================
    # Introspection
    # =========================================================
    def symbols(self) -> List[str]:
        return sorted(self.pnl_r.keys())

    def has_open_position(self, symbol: str) -> bool:
        return bool(self.open_positions.get(str(symbol).upper().strip(), False))

    def last_corr_meta(self) -> Dict[str, object]:
        """
        Debug/meta del último correlation_df().
        Útil para trazabilidad “institucional”.
        """
        return dict(self._last_corr_meta or {})

    def __repr__(self) -> str:
        return (
            f"PortfolioState(window={self.window}, "
            f"symbols={len(self.pnl_r)}, "
            f"open_positions={sum(1 for v in self.open_positions.values() if v)})"
        )
