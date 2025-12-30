# infra/gpu_features.py
from __future__ import annotations

from typing import Optional

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

import numpy as np


class GPUFeatureEngine:
    """
    Rolling + incremental indicators on GPU (CuPy).
    Fallback transparente a NumPy si CuPy no está disponible.
    """

    def __init__(
        self,
        max_len: int = 2000,
        atr_n: int = 14,
        vwap_window: int = 96,
        vol_window: int = 30,
    ):
        self.max_len = int(max_len)
        self.atr_n = int(atr_n)
        self.vwap_window = int(vwap_window)
        self.vol_window = int(vol_window)

        self.xp = cp if _HAS_CUPY else np

        self.close = self.xp.zeros(self.max_len, dtype=self.xp.float32)
        self.high = self.xp.zeros(self.max_len, dtype=self.xp.float32)
        self.low = self.xp.zeros(self.max_len, dtype=self.xp.float32)
        self.volume = self.xp.zeros(self.max_len, dtype=self.xp.float32)

        self.size = 0

        self.tr: Optional[float] = None
        self.atr: Optional[float] = None

    # --------------------------------------------------
    # Update buffers (O(1))
    # --------------------------------------------------
    def update(self, o: float, h: float, l: float, c: float, v: float):
        xp = self.xp

        if self.size < self.max_len:
            idx = self.size
            self.size += 1
        else:
            self.close[:-1] = self.close[1:]
            self.high[:-1] = self.high[1:]
            self.low[:-1] = self.low[1:]
            self.volume[:-1] = self.volume[1:]
            idx = self.max_len - 1

        self.close[idx] = c
        self.high[idx] = h
        self.low[idx] = l
        self.volume[idx] = v

        self._update_atr(h, l, c)

    # --------------------------------------------------
    # ATR (EMA incremental)
    # --------------------------------------------------
    def _update_atr(self, high: float, low: float, close: float):
        if self.size <= 1:
            tr = high - low
        else:
            prev_close = float(self.close[self.size - 2])
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )

        alpha = 1.0 / float(self.atr_n)

        if self.atr is None:
            self.atr = tr
        else:
            self.atr = alpha * tr + (1.0 - alpha) * self.atr

    # --------------------------------------------------
    # Exposed scalars (CPU floats)
    # --------------------------------------------------
    def atr_value(self) -> Optional[float]:
        return float(self.atr) if self.atr is not None else None

    def atr_sma(self) -> Optional[float]:
        if self.size < max(10, self.atr_n):
            return None
        # EMA ATR → SMA aproximado usando últimos N TRs
        return float(self.atr)

    def vol_sma(self) -> Optional[float]:
        if self.size < self.vol_window:
            return None
        v = self.volume[self.size - self.vol_window : self.size]
        return float(self.xp.mean(v).item())

    def vwap(self) -> Optional[float]:
        if self.size == 0:
            return None
        w = min(self.vwap_window, self.size)
        c = self.close[self.size - w : self.size]
        h = self.high[self.size - w : self.size]
        l = self.low[self.size - w : self.size]
        v = self.volume[self.size - w : self.size]

        typical = (h + l + c) / 3.0
        pv = typical * v
        vol_sum = self.xp.sum(v)

        if float(vol_sum) <= 0:
            return None

        return float((self.xp.sum(pv) / vol_sum).item())
