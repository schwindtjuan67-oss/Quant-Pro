# analysis/correlation_provider.py
from __future__ import annotations

import pandas as pd
from typing import Optional

from analysis.correlation_guard import (
    build_r_matrix,
    correlation_matrix,
)


class CorrelationProvider:
    """
    Fuente de correlación basada en trades EXIT históricos.
    Usa pnl_r (R-multiples).
    """

    def __init__(
        self,
        trades_csv_glob: str = "logs/*_shadow_trades_v2.csv",
        window: int = 100,
    ):
        self.trades_csv_glob = trades_csv_glob
        self.window = window

    def load_exits(self) -> pd.DataFrame:
        import glob

        frames = []
        for path in glob.glob(self.trades_csv_glob):
            try:
                df = pd.read_csv(path)
                if "type" in df.columns:
                    df = df[df["type"] == "EXIT"]
                frames.append(df)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def corr_df(self) -> pd.DataFrame:
        exits = self.load_exits()
        if exits.empty:
            return pd.DataFrame()

        r_matrix = build_r_matrix(
            exits_df=exits,
            symbol_col="symbol" if "symbol" in exits.columns else "regime",  # fallback
            value_col="pnl_r",
            window=self.window,
        )

        return correlation_matrix(r_matrix)
