import numpy as np

# ================================================================
# Rolling helpers NumPy
# ================================================================

def _rolling_sum(arr, window: int):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if window <= 0 or window > n:
        return out

    cumsum = np.cumsum(arr)
    out[window - 1:] = cumsum[window - 1:] - np.concatenate(([0.0], cumsum[:-window]))
    return out


def _rolling_mean(arr, window: int):
    s = _rolling_sum(arr, window)
    out = s / float(window)
    return out


def _rolling_std(arr, window: int):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if window <= 1 or window > n:
        return np.full(n, np.nan, dtype=float)

    mean = _rolling_mean(arr, window)
    arr2 = arr * arr
    mean2 = _rolling_mean(arr2, window)

    var = mean2 - mean * mean
    var[var < 0] = 0.0
    return np.sqrt(var)


# ================================================================
# VWAP PRO (NumPy puro)
# ================================================================

def vwap_np(close, high, low, volume, window: int = 96):
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    volume = np.asarray(volume, dtype=float)

    typical = (high + low + close) / 3.0
    pv = typical * volume

    # rolling PV y rolling Vol
    pv_sum = _rolling_sum(pv, window)
    vol_sum = _rolling_sum(volume, window)

    vwap = pv_sum / (vol_sum + 1e-9)
    return vwap


def vwap_bands_np(close, high, low, volume, window: int = 96, mult: float = 2.0):
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)

    v = vwap_np(close, high, low, volume, window)
    typical = (high + low + close) / 3.0
    std = _rolling_std(typical, window)

    upper = v + mult * std
    lower = v - mult * std

    # limpiamos NaNs iniciales
    v[~np.isfinite(v)] = 0.0
    upper[~np.isfinite(upper)] = 0.0
    lower[~np.isfinite(lower)] = 0.0

    return v, upper, lower


# ================================================================
# VWAP MODULE PRO (con máscaras long/short)
# ================================================================

class VWAPModule:
    """
    VWAP + Bandas + Filtros de tendencia / reversión.
    Exponer:
      - self.vwap
      - self.upper
      - self.lower
      - self.long_mask
      - self.short_mask
    """

    def __init__(self, data,
                 window=96,
                 band_mult=2.0,
                 mode="trend",  # "trend" o "reversion"
                 strict=True,    # si True exige cruce, si False basta con posición encima/debajo
                 **kwargs):

        close = np.asarray(data.Close, dtype=float)
        high = np.asarray(data.High, dtype=float)
        low = np.asarray(data.Low, dtype=float)
        volume = np.asarray(data.Volume, dtype=float)

        self.vwap, self.upper, self.lower = vwap_bands_np(
            close, high, low, volume,
            window=window,
            mult=band_mult
        )

        n = len(close)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        if mode == "trend":
            # Tendencia:
            # long si close > VWAP y opcional cruce ascendente
            # short si close < VWAP y cruce descendente
            above = close > self.vwap
            below = close < self.vwap

            if strict:
                # cruce ascendente: close[-2] < vwap[-2] AND close[-1] > vwap[-1]
                cross_up = np.zeros(n, dtype=bool)
                cross_dn = np.zeros(n, dtype=bool)
                cross_up[1:] = (~above[:-1] & above[1:])
                cross_dn[1:] = (~below[:-1] & below[1:])

                long_mask = cross_up
                short_mask = cross_dn
            else:
                long_mask = above
                short_mask = below

        elif mode == "reversion":
            # Reversión:
            # long si toca banda inferior y rebota
            # short si toca banda superior y rechaza
            touch_lower = close < self.lower
            touch_upper = close > self.upper

            if strict:
                # rebote = vuelve dentro de la banda
                bounce_long = np.zeros(n, dtype=bool)
                bounce_short = np.zeros(n, dtype=bool)
                bounce_long[1:] = touch_lower[:-1] & (close[1:] > self.lower[1:])
                bounce_short[1:] = touch_upper[:-1] & (close[1:] < self.upper[1:])
                long_mask = bounce_long
                short_mask = bounce_short
            else:
                long_mask = touch_lower
                short_mask = touch_upper

        # guarda las máscaras limpias
        self.long_mask = long_mask
        self.short_mask = short_mask

    def allow_long(self, i):
        return bool(self.long_mask[i])

    def allow_short(self, i):
        return bool(self.short_mask[i])
