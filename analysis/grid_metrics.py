from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional


# =========================
# Helpers
# =========================

def _safe_mean(x: List[float]) -> Optional[float]:
    if not x:
        return None
    return float(np.mean(x))


def _safe_std(x: List[float]) -> Optional[float]:
    if not x:
        return None
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


# =========================
# Core metrics
# =========================

def compute_max_drawdown(equity_curve: List[float]) -> float:
    """
    equity_curve: lista de equity acumulada (en R o en USD)
    retorna max drawdown como número positivo (ej: 0.18 = 18%)
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for x in equity_curve:
        peak = max(peak, x)
        dd = (peak - x)
        max_dd = max(max_dd, dd)

    return float(max_dd)


def compute_sortino(returns: List[float]) -> Optional[float]:
    """
    Sortino ratio sin risk-free rate.
    Penaliza solo volatilidad negativa.
    """
    if not returns:
        return None

    mean_r = _safe_mean(returns)
    downside = [r for r in returns if r < 0]

    if not downside:
        # sin downside risk → edge fuerte pero capado
        return 10.0

    downside_std = _safe_std(downside)
    if downside_std == 0:
        return 10.0

    return float(mean_r / downside_std)


def compute_profit_factor(returns: List[float]) -> Optional[float]:
    gains = [r for r in returns if r > 0]
    losses = [-r for r in returns if r < 0]

    if not losses:
        return None

    return sum(gains) / sum(losses) if gains else 0.0


# =========================
# Public API
# =========================

def compute_metrics_from_trades(trades: List[Dict]) -> Dict[str, float]:
    """
    trades: lista de dicts con al menos:
        - pnl_r
        - holding_time_sec (opcional)

    Devuelve métricas agregadas listas para ranking.
    """

    if not trades:
        return {
            "trades": 0,
            "equity_r": 0.0,
            "max_drawdown_r": 0.0,
            "sortino": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "winrate": 0.0,
            "avg_trade_duration": 0.0,
        }

    returns = [float(t.get("pnl_r", 0.0)) for t in trades]

    # equity curve en R
    equity_curve = np.cumsum(returns).tolist()

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    metrics = {
        "trades": len(trades),
        "equity_r": float(sum(returns)),
        "max_drawdown_r": compute_max_drawdown(equity_curve),
        "sortino": compute_sortino(returns) or 0.0,
        "expectancy": _safe_mean(returns) or 0.0,
        "profit_factor": compute_profit_factor(returns) or 0.0,
        "winrate": len(wins) / len(returns) if returns else 0.0,
    }

    # duración promedio
    durations = [t.get("holding_time_sec") for t in trades if t.get("holding_time_sec") is not None]
    metrics["avg_trade_duration"] = float(np.mean(durations)) if durations else 0.0

    return metrics


# =========================
# Composite score (ranking)
# =========================

def compute_score(metrics: Dict[str, float]) -> float:
    """
    Score antifragil para optimización (anti-overfitting).
    """

    trades = metrics.get("trades", 0)
    if trades < 20:
        return -1e9  # descarta configs sin data

    equity_r = metrics.get("equity_r", 0.0)
    max_dd = abs(metrics.get("max_drawdown_r", 0.0))
    sortino = metrics.get("sortino", 0.0)

    # caps defensivos
    sortino = min(max(sortino, 0.0), 5.0)

    score = (
        equity_r
        - 0.5 * max_dd
    ) * math.log1p(trades) * sortino

    return float(score)
