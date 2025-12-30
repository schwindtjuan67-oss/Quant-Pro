# analysis/decay_tools.py
import numpy as np
import pandas as pd
from typing import Tuple

def compute_decay_weights(
    dt_series: pd.Series,
    half_life_hours: float,
) -> Tuple[np.ndarray, float]:
    """
    Exponential time decay.
    Returns:
      - weights (np.ndarray)
      - effective sample size
    """
    if dt_series.empty:
        return np.array([]), 0.0

    now = dt_series.max()
    age_hours = (now - dt_series).dt.total_seconds() / 3600.0

    lam = np.log(2.0) / float(half_life_hours)
    weights = np.exp(-lam * age_hours.values)

    eff_n = (weights.sum() ** 2) / (np.square(weights).sum() + 1e-9)
    return weights, float(eff_n)
