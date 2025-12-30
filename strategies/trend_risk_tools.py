import numpy as np

# ================================================================
# Helpers NumPy
# ================================================================

def _ema_np(arr, span: int):
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    alpha = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _true_range(high, low, close):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )
    return tr


def _adx_np(high, low, close, window: int = 14):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    prev_high = np.roll(high, 1); prev_high[0] = high[0]
    prev_low = np.roll(low, 1); prev_low[0] = low[0]
    prev_close = np.roll(close, 1); prev_close[0] = close[0]

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.)

    tr = _true_range(high, low, close)

    tr_smooth = _ema_np(tr, window)
    plus_smooth = _ema_np(plus_dm, window)
    minus_smooth = _ema_np(minus_dm, window)

    plus_di = 100 * plus_smooth / (tr_smooth + 1e-9)
    minus_di = 100 * minus_smooth / (tr_smooth + 1e-9)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = _ema_np(dx, window)

    return adx


# ================================================================
# TREND FILTER PRO
# ================================================================

def trend_direction_np(close, fast=20, slow=80):
    close = np.asarray(close, dtype=float)
    ema_fast = _ema_np(close, fast)
    ema_slow = _ema_np(close, slow)
    trend = np.where(ema_fast > ema_slow, 1,
             np.where(ema_fast < ema_slow, -1, 0))
    return trend.astype(int)


# ================================================================
# Trend + Risk Module PRO
# ================================================================

class TrendRiskModule:
    """
    Trend + ADX filter PRO.

    Exponer:
      - self.trend_strength     (+1 / 0 / -1)
      - self.adx                (float array)
      - self.long_mask
      - self.short_mask

    Parámetros:
      mode: "trend" (default), "strict"
      use_adx: bool
      adx_min: float mínimo para habilitar trades
    """

    def __init__(self, data,
                 fast=20,
                 slow=80,
                 mode="trend",       # "trend" o "strict"
                 use_adx=False,
                 adx_n=14,
                 adx_min=15,
                 **kwargs):

        close = np.asarray(data.Close, dtype=float)
        high = np.asarray(data.High, dtype=float)
        low = np.asarray(data.Low, dtype=float)

        # ----------- TREND -----------
        self.trend_strength = trend_direction_np(close, fast, slow)

        n = len(close)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        if mode == "trend":
            long_mask = self.trend_strength == 1
            short_mask = self.trend_strength == -1

        elif mode == "strict":
            # cruce fuerte
            ts = self.trend_strength
            long_mask[1:] = (ts[:-1] <= 0) & (ts[1:] == 1)
            short_mask[1:] = (ts[:-1] >= 0) & (ts[1:] == -1)

        # ----------- ADX opcional -----------
        if use_adx:
            adx = _adx_np(high, low, close, window=adx_n)
            self.adx = adx
            strong = adx >= adx_min
            long_mask &= strong
            short_mask &= strong
        else:
            self.adx = np.zeros(n)

        # guardar máscaras limpias
        self.long_mask = long_mask
        self.short_mask = short_mask

    def allow_long(self, i):
        return bool(self.long_mask[i])

    def allow_short(self, i):
        return bool(self.short_mask[i])
