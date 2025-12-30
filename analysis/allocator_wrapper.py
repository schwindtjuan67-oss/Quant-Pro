# analysis/allocator_wrapper.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, Optional

import pandas as pd

from analysis.allocator import allocate_risk, AllocatorConfig


class AllocatorWrapper:
    """
    Wrapper de producción para allocate_risk().

    Interfaz pensada para engines/adapter:
    - decide_for_symbol(): dado un symbol y contexto, devuelve allow + vol_target
    - decide_portfolio(): dado symbol_scores, devuelve allocations completas

    IMPORTANT:
    - NO toca estrategia.
    - Devuelve vol_target_override para el Hybrid.
    """

    def __init__(self, cfg: Optional[AllocatorConfig] = None):
        self.cfg = cfg or AllocatorConfig()

    def decide_portfolio(
        self,
        *,
        symbol_scores: Dict[str, float],
        corr_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return allocate_risk(
            symbol_scores=symbol_scores,
            corr_df=corr_df,
            cfg=self.cfg,
            context=context or {},
        )

    def decide_for_symbol(
        self,
        *,
        symbol: str,
        balance_usdt: float,
        max_vol_target: float,
        symbol_scores: Dict[str, float],
        corr_df: Optional[pd.DataFrame] = None,
        risk_mult: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Devuelve:
          {
            "allow": bool,
            "reason": str,
            "vol_target": float,
            "risk_usdt": float,
            "meta": {...}
          }

        Nota:
        - vol_target = (risk_usdt / balance_usdt) * risk_mult
        - clamp a [0, max_vol_target]
        """

        symbol = str(symbol).upper()
        balance_usdt = float(balance_usdt or 0.0)
        max_vol_target = float(max_vol_target or 0.0)
        risk_mult = float(risk_mult or 1.0)

        if balance_usdt <= 0 or max_vol_target <= 0:
            return {
                "allow": False,
                "reason": "NO_BALANCE_OR_MAXVOL",
                "vol_target": 0.0,
                "risk_usdt": 0.0,
                "meta": {"cfg": asdict(self.cfg), "context": context or {}},
            }

        decision = self.decide_portfolio(
            symbol_scores=symbol_scores,
            corr_df=corr_df,
            context=context or {},
        )

        allocs = decision.get("allocations", {}) or {}
        if symbol not in allocs:
            reason = decision.get("meta", {}).get("reason", "NOT_SELECTED")
            return {
                "allow": False,
                "reason": str(reason),
                "vol_target": 0.0,
                "risk_usdt": 0.0,
                "meta": decision.get("meta", {}),
            }

        risk_usdt = float(allocs[symbol].get("risk_usdt", 0.0))
        if risk_usdt <= 0:
            return {
                "allow": False,
                "reason": "RISK_USDT_ZERO",
                "vol_target": 0.0,
                "risk_usdt": 0.0,
                "meta": decision.get("meta", {}),
            }

        # convertir risk budget -> fraction of equity
        vol_target = (risk_usdt / balance_usdt) * max(0.0, risk_mult)

        # clamp por seguridad
        if vol_target < 0:
            vol_target = 0.0
        if vol_target > max_vol_target:
            vol_target = max_vol_target

        return {
            "allow": vol_target > 0.0,
            "reason": "OK",
            "vol_target": float(vol_target),
            "risk_usdt": float(risk_usdt),
            "meta": decision.get("meta", {}),
        }
