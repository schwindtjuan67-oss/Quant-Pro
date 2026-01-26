from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class FeatureCache:
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray

    _ema_cache: Dict[Tuple[str, int, int], np.ndarray] = None
    _atr_cache: Dict[Tuple[str, int, int], np.ndarray] = None
    _vwap_cache: Dict[Tuple[str, int, int], np.ndarray] = None
    _std_cache: Dict[Tuple[str, int, int], np.ndarray] = None

    def __post_init__(self) -> None:
        if self._ema_cache is None:
            self._ema_cache = {}
        if self._atr_cache is None:
            self._atr_cache = {}
        if self._vwap_cache is None:
            self._vwap_cache = {}
        if self._std_cache is None:
            self._std_cache = {}

        self.close = self._to_float32(self.close)
        self.high = self._to_float32(self.high)
        self.low = self._to_float32(self.low)
        self.volume = self._to_float32(self.volume)

        n = len(self.close)
        if len(self.high) != n or len(self.low) != n or len(self.volume) != n:
            raise ValueError("FeatureCache: OHLCV length mismatch")

    @classmethod
    def from_candles(cls, candles) -> "FeatureCache":
        close = np.asarray([float(c.get("close", 0.0)) for c in candles], dtype=np.float32)
        high = np.asarray([float(c.get("high", 0.0)) for c in candles], dtype=np.float32)
        low = np.asarray([float(c.get("low", 0.0)) for c in candles], dtype=np.float32)
        volume = np.asarray([float(c.get("volume", 0.0)) for c in candles], dtype=np.float32)
        return cls(close=close, high=high, low=low, volume=volume)

    @staticmethod
    def _to_float32(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float32, copy=False)

    def ema(self, period: int) -> np.ndarray:
        period = max(1, int(period))
        key = ("ema", period, len(self.close))
        cached = self._ema_cache.get(key)
        if cached is not None:
            return cached

        close = self.close.astype(np.float64)
        out = np.empty_like(close, dtype=np.float64)
        alpha = 2.0 / (period + 1.0)
        out[0] = close[0] if close.size else 0.0
        for i in range(1, close.size):
            out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._ema_cache[key] = out
        return out

    def atr(self, period: int) -> np.ndarray:
        period = max(1, int(period))
        key = ("atr", period, len(self.close))
        cached = self._atr_cache.get(key)
        if cached is not None:
            return cached

        high = self.high.astype(np.float64)
        low = self.low.astype(np.float64)
        close = self.close.astype(np.float64)
        n = close.size
        out = np.zeros(n, dtype=np.float64)
        if n == 0:
            self._atr_cache[key] = out.astype(np.float32)
            return self._atr_cache[key]

        tr = np.empty(n, dtype=np.float64)
        tr[0] = max(high[0] - low[0], 0.0)
        for i in range(1, n):
            prev_close = close[i - 1]
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - prev_close),
                abs(low[i] - prev_close),
            )

        alpha = 1.0 / float(period)
        out[0] = tr[0]
        for i in range(1, n):
            out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._atr_cache[key] = out
        return out

    def vwap(self, window: int) -> np.ndarray:
        window = max(1, int(window))
        key = ("vwap", window, len(self.close))
        cached = self._vwap_cache.get(key)
        if cached is not None:
            return cached

        close = self.close.astype(np.float64)
        high = self.high.astype(np.float64)
        low = self.low.astype(np.float64)
        volume = np.maximum(self.volume.astype(np.float64), 0.0)
        n = close.size
        out = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            self._vwap_cache[key] = out.astype(np.float32)
            return self._vwap_cache[key]

        typical = (high + low + close) / 3.0
        pv = typical * volume
        cum_pv = np.cumsum(pv)
        cum_v = np.cumsum(volume)
        for i in range(window - 1, n):
            start = i - window
            pv_sum = cum_pv[i] - (cum_pv[start] if start >= 0 else 0.0)
            v_sum = cum_v[i] - (cum_v[start] if start >= 0 else 0.0)
            out[i] = pv_sum / (v_sum + 1e-9)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._vwap_cache[key] = out
        return out

    def std(self, window: int) -> np.ndarray:
        window = max(1, int(window))
        key = ("std", window, len(self.close))
        cached = self._std_cache.get(key)
        if cached is not None:
            return cached

        close = self.close.astype(np.float64)
        n = close.size
        out = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            self._std_cache[key] = out.astype(np.float32)
            return self._std_cache[key]

        cumsum = np.cumsum(close)
        cumsum2 = np.cumsum(close * close)
        for i in range(window - 1, n):
            start = i - window
            sum_x = cumsum[i] - (cumsum[start] if start >= 0 else 0.0)
            sum_x2 = cumsum2[i] - (cumsum2[start] if start >= 0 else 0.0)
            mean = sum_x / float(window)
            var = (sum_x2 / float(window)) - (mean * mean)
            if var < 0.0:
                var = 0.0
            out[i] = np.sqrt(var)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._std_cache[key] = out
        return out


_ACTIVE_CACHE: Optional[FeatureCache] = None


def set_active_cache(cache: Optional[FeatureCache]) -> None:
    global _ACTIVE_CACHE
    _ACTIVE_CACHE = cache


def get_active_cache() -> Optional[FeatureCache]:
    return _ACTIVE_CACHE
