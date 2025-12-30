# analysis/allocator_bridge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from analysis.allocator import allocate_risk, AllocatorConfig

# PortfolioState es opcional (FASE 8 industrial). Si no existe/no lo tenés, no rompe.
try:
    from analysis.portfolio_state import PortfolioState  # type: ignore
except Exception:
    PortfolioState = None  # fallback


# ============================================================
# Decision dataclass (simple + práctico)
# ============================================================
@dataclass
class AllocationDecision:
    allow: bool
    reason: str
    symbol: str

    # allocator outputs
    weight: float = 0.0
    risk_usdt: float = 0.0
    score: float = 0.0

    # sizing output for Hybrid
    vol_target_override: float = 0.0

    # debug/meta
    meta: Optional[Dict[str, Any]] = None


# ============================================================
# Utils
# ============================================================
def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _upper_sym(x: str) -> str:
    return str(x or "").upper().strip()


# ============================================================
# Bridge
# ============================================================
class AllocatorBridge:
    """
    Bridge entre:
      - scores (por símbolo)
      - correlaciones (corr_df opcional o desde PortfolioState)
      - allocate_risk() (analysis/allocator.py)

    y devuelve:
      - vol_target_override para Hybrid (fracción del equity/balance)

    Importante:
      allocator.risk_usdt acá se interpreta como "presupuesto USDT a asignar"
      (similar a cash_to_risk), NO como stop-risk (porque allocator no conoce SL/ATR).

    FASE 8 (industrial):
      - opcionalmente usa PortfolioState para corr_df runtime (sin CSV).
      - helpers multi-symbol (allocate_multi).

    PATCH (tú pedido):
      - trazabilidad explícita de fuente de correlación:
        corr_source = "explicit" | "last" | "portfolio" | "none"
        (se guarda en payload["meta"])
    """

    def __init__(
        self,
        cfg: Optional[AllocatorConfig] = None,
        *,
        # FASE 8: estado de portfolio (correlación in-memory)
        portfolio: Optional["PortfolioState"] = None,
        # sizing clamps (anti "errores humanos")
        min_vol_target: float = 0.0,
        max_vol_target: Optional[float] = None,  # si None -> lo decide el caller (ej. Hybrid.VOL_TARGET)
        # si True, el bridge reduce total_risk_usdt por risk_mult global (de RiskManager)
        apply_risk_mult_to_budget: bool = True,
        # FASE 8: si True y hay portfolio, usa corr_df del portfolio por defecto
        use_portfolio_corr_by_default: bool = True,
    ):
        self.cfg = cfg or AllocatorConfig()
        self.portfolio = portfolio

        self.min_vol_target = float(min_vol_target)
        self.max_vol_target = max_vol_target
        self.apply_risk_mult_to_budget = bool(apply_risk_mult_to_budget)
        self.use_portfolio_corr_by_default = bool(use_portfolio_corr_by_default)

        # state opcional (modo incremental / cache)
        self._last_scores: Dict[str, float] = {}
        self._last_context: Dict[str, Any] = {}
        self._last_corr_df: Optional[pd.DataFrame] = None
        self._last_alloc_payload: Optional[Dict[str, Any]] = None

    # -------------------------
    # State helpers (opcionales)
    # -------------------------
    def set_scores(self, symbol_scores: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> None:
        self._last_scores = dict(symbol_scores or {})
        self._last_context = dict(context or {})

    def set_corr_df(self, corr_df: Optional[pd.DataFrame]) -> None:
        self._last_corr_df = corr_df

    def last_payload(self) -> Optional[Dict[str, Any]]:
        return self._last_alloc_payload

    # -------------------------
    # FASE 8 + TRACE: corr_df resolver
    # -------------------------
    def _resolve_corr_df_with_source(self, corr_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Precedencia:
          1) corr_df explícito (arg)      -> "explicit"
          2) self._last_corr_df (cache)   -> "last"
          3) self.portfolio.correlation_df() -> "portfolio"
          4) none                         -> "none"

        Nota: si la fuente existe pero viene vacía/None, igualmente
        se reporta la fuente, y corr_used=False.
        """
        # 1) corr_df explícito gana
        if corr_df is not None:
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                return corr_df, "explicit"
            return None, "explicit"

        # 2) si el usuario seteó manualmente vía set_corr_df
        if self._last_corr_df is not None:
            if isinstance(self._last_corr_df, pd.DataFrame) and not self._last_corr_df.empty:
                return self._last_corr_df, "last"
            return None, "last"

        # 3) si hay portfolio y está habilitado usarlo por defecto
        if self.use_portfolio_corr_by_default and self.portfolio is not None:
            try:
                c = self.portfolio.correlation_df()
                if c is not None and isinstance(c, pd.DataFrame) and not c.empty:
                    return c, "portfolio"
                return None, "portfolio"
            except Exception:
                return None, "portfolio"

        return None, "none"

    # -------------------------
    # Core call
    # -------------------------
    def allocate(
        self,
        symbol_scores: Optional[Dict[str, float]] = None,
        *,
        corr_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
        # risk softening (FASE 7.2): 1.0 normal, 0.5 conservador, 0.0 hard stop
        risk_mult: float = 1.0,
        # si querés forzar presupuesto total por tick (override cfg)
        total_risk_usdt_override: Optional[float] = None,
        base_risk_usdt_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Devuelve el payload completo del allocator (allocations + meta).
        Compatible con FASE 7.3 y extendido con FASE 8 (corr desde portfolio).
        """
        scores = symbol_scores if symbol_scores is not None else self._last_scores
        ctx = context if context is not None else self._last_context

        cdf, corr_source = self._resolve_corr_df_with_source(corr_df)

        rm = _safe_float(risk_mult, 1.0)
        rm = _clamp(rm, 0.0, 1.0)

        # clonar cfg para no mutar el objeto original
        cfg = AllocatorConfig(**self.cfg.__dict__)

        if total_risk_usdt_override is not None:
            cfg.total_risk_usdt = float(total_risk_usdt_override)
        if base_risk_usdt_override is not None:
            cfg.base_risk_usdt = float(base_risk_usdt_override)

        # aplicar risk_mult al presupuesto (si está habilitado)
        if self.apply_risk_mult_to_budget:
            cfg.total_risk_usdt = float(cfg.total_risk_usdt) * float(rm)
            cfg.base_risk_usdt = float(cfg.base_risk_usdt) * float(rm)

        payload = allocate_risk(
            symbol_scores=scores or {},
            corr_df=cdf,
            cfg=cfg,
            context=ctx,
        )

        # --- TRACE PATCH: agregar trazabilidad sin romper contrato ---
        meta = payload.get("meta", {}) or {}
        corr_used = bool(cdf is not None and isinstance(cdf, pd.DataFrame) and not cdf.empty)
        corr_shape = tuple(cdf.shape) if corr_used else None

        meta["corr_source"] = str(corr_source)
        meta["corr_used"] = bool(corr_used)
        meta["corr_shape"] = corr_shape

        payload["meta"] = meta

        self._last_alloc_payload = payload
        return payload

    # Alias “industrial” (FASE 8): scores -> payload (sin pensar)
    def decide(
        self,
        symbol_scores: Dict[str, float],
        *,
        context: Optional[Dict[str, Any]] = None,
        corr_df: Optional[pd.DataFrame] = None,
        risk_mult: float = 1.0,
        total_risk_usdt_override: Optional[float] = None,
        base_risk_usdt_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.set_scores(symbol_scores or {}, context=context)
        return self.allocate(
            symbol_scores=symbol_scores,
            corr_df=corr_df,
            context=context,
            risk_mult=risk_mult,
            total_risk_usdt_override=total_risk_usdt_override,
            base_risk_usdt_override=base_risk_usdt_override,
        )

    # -------------------------
    # Convenience: decisión por símbolo + vol_target_override
    # -------------------------
    def decide_for_symbol(
        self,
        symbol: str,
        *,
        balance_usdt: float,
        # cap típico: Hybrid.VOL_TARGET (para no exceder lo que optimizaste)
        max_vol_target: Optional[float] = None,
        # inputs del allocator
        symbol_scores: Optional[Dict[str, float]] = None,
        corr_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
        risk_mult: float = 1.0,
        # si querés forzar presupuesto total por tick (override cfg)
        total_risk_usdt_override: Optional[float] = None,
        base_risk_usdt_override: Optional[float] = None,
    ) -> AllocationDecision:
        """
        Devuelve una decisión “lista para Hybrid”:
          - allow (si el símbolo fue seleccionado y tiene risk_usdt > 0)
          - vol_target_override (fracción de balance)
          - reason + meta para debugging
        """
        sym = _upper_sym(symbol)
        bal = _safe_float(balance_usdt, 0.0)

        if not sym:
            return AllocationDecision(
                allow=False,
                reason="invalid_symbol",
                symbol=sym,
                meta={"why": "symbol vacío"},
            )

        if bal <= 0:
            return AllocationDecision(
                allow=False,
                reason="zero_balance",
                symbol=sym,
                meta={"balance_usdt": bal},
            )

        payload = self.allocate(
            symbol_scores=symbol_scores,
            corr_df=corr_df,
            context=context,
            risk_mult=risk_mult,
            total_risk_usdt_override=total_risk_usdt_override,
            base_risk_usdt_override=base_risk_usdt_override,
        )

        allocs = payload.get("allocations", {}) or {}
        if sym not in allocs:
            return AllocationDecision(
                allow=False,
                reason="not_selected_by_allocator",
                symbol=sym,
                meta={
                    "used_symbols": payload.get("used_symbols", []),
                    "blocked_pairs": payload.get("blocked_pairs", []),
                    "allocator_meta": payload.get("meta", {}),
                },
            )

        a = allocs.get(sym, {}) or {}
        w = _safe_float(a.get("weight"), 0.0)
        r_usdt = _safe_float(a.get("risk_usdt"), 0.0)
        sc = _safe_float(a.get("score"), 0.0)

        if r_usdt <= 0:
            return AllocationDecision(
                allow=False,
                reason="allocated_risk_zero",
                symbol=sym,
                weight=w,
                risk_usdt=r_usdt,
                score=sc,
                meta={"allocator_meta": payload.get("meta", {})},
            )

        # convertir risk_usdt (presupuesto) -> vol_target_override
        vt_raw = (r_usdt / bal) if bal > 0 else 0.0

        # clamps anti “errores humanos”
        vt_cap = max_vol_target if max_vol_target is not None else self.max_vol_target
        if vt_cap is None:
            vt_cap = 1.0  # última red: no explotar

        vt = _clamp(_safe_float(vt_raw, 0.0), float(self.min_vol_target), float(vt_cap))
        allow = vt > 0.0

        return AllocationDecision(
            allow=allow,
            reason="OK" if allow else "vol_target_zero_after_clamp",
            symbol=sym,
            weight=w,
            risk_usdt=r_usdt,
            score=sc,
            vol_target_override=vt,
            meta={
                "balance_usdt": bal,
                "vt_raw": vt_raw,
                "vt_clamped": vt,
                "vt_cap": vt_cap,
                "allocator_used_symbols": payload.get("used_symbols", []),
                "allocator_blocked_pairs": payload.get("blocked_pairs", []),
                "allocator_meta": payload.get("meta", {}),  # incluye corr_source/corr_used/corr_shape
            },
        )

    # -------------------------
    # FASE 8: multi-symbol helper
    # -------------------------
    def allocate_multi(
        self,
        *,
        symbol_scores: Dict[str, float],
        balances_usdt: Dict[str, float],
        max_vol_targets: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
        corr_df: Optional[pd.DataFrame] = None,
        risk_mult: float = 1.0,
        total_risk_usdt_override: Optional[float] = None,
        base_risk_usdt_override: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, AllocationDecision]]:
        """
        FASE 8 — “industrial”:
        - corre allocate() una vez para todos los símbolos
        - devuelve:
            payload allocator
            decisions por símbolo con vol_target_override (cada uno con su balance/cap)
        """
        scores_clean = {_upper_sym(k): _safe_float(v, 0.0) for k, v in (symbol_scores or {}).items()}
        balances_clean = {_upper_sym(k): _safe_float(v, 0.0) for k, v in (balances_usdt or {}).items()}

        payload = self.allocate(
            symbol_scores=scores_clean,
            corr_df=corr_df,
            context=context,
            risk_mult=risk_mult,
            total_risk_usdt_override=total_risk_usdt_override,
            base_risk_usdt_override=base_risk_usdt_override,
        )

        decisions: Dict[str, AllocationDecision] = {}
        allocs = payload.get("allocations", {}) or {}

        for sym, sc_in in scores_clean.items():
            bal = balances_clean.get(sym, 0.0)
            cap = None
            if max_vol_targets is not None:
                cap = max_vol_targets.get(sym)

            if bal <= 0:
                decisions[sym] = AllocationDecision(
                    allow=False,
                    reason="zero_balance",
                    symbol=sym,
                    score=float(sc_in),
                    meta={"balance_usdt": bal, "allocator_meta": payload.get("meta", {})},
                )
                continue

            if sym not in allocs:
                decisions[sym] = AllocationDecision(
                    allow=False,
                    reason="not_selected_by_allocator",
                    symbol=sym,
                    score=float(sc_in),
                    meta={
                        "used_symbols": payload.get("used_symbols", []),
                        "blocked_pairs": payload.get("blocked_pairs", []),
                        "allocator_meta": payload.get("meta", {}),
                    },
                )
                continue

            a = allocs.get(sym, {}) or {}
            w = _safe_float(a.get("weight"), 0.0)
            r_usdt = _safe_float(a.get("risk_usdt"), 0.0)
            sc = _safe_float(a.get("score"), float(sc_in))

            if r_usdt <= 0:
                decisions[sym] = AllocationDecision(
                    allow=False,
                    reason="allocated_risk_zero",
                    symbol=sym,
                    weight=w,
                    risk_usdt=r_usdt,
                    score=sc,
                    meta={"allocator_meta": payload.get("meta", {})},
                )
                continue

            vt_raw = (r_usdt / bal) if bal > 0 else 0.0

            vt_cap = cap if cap is not None else self.max_vol_target
            if vt_cap is None:
                vt_cap = 1.0

            vt = _clamp(_safe_float(vt_raw, 0.0), float(self.min_vol_target), float(vt_cap))
            allow = vt > 0.0

            decisions[sym] = AllocationDecision(
                allow=allow,
                reason="OK" if allow else "vol_target_zero_after_clamp",
                symbol=sym,
                weight=w,
                risk_usdt=r_usdt,
                score=sc,
                vol_target_override=vt,
                meta={
                    "balance_usdt": bal,
                    "vt_raw": vt_raw,
                    "vt_clamped": vt,
                    "vt_cap": vt_cap,
                    "allocator_used_symbols": payload.get("used_symbols", []),
                    "allocator_blocked_pairs": payload.get("blocked_pairs", []),
                    "allocator_meta": payload.get("meta", {}),  # incluye corr_source/corr_used/corr_shape
                },
            )

        return payload, decisions
