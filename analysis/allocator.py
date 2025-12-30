# analysis/allocator.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd


# ============================================================
# Config dataclass
# ============================================================
@dataclass
class AllocatorConfig:
    # --- risk budget ---
    total_risk_usdt: float = 10.0            # Riesgo total que querés repartir entre símbolos "en el tick"
    base_risk_usdt: float = 5.0              # Riesgo base por trade si solo hay 1 símbolo
    min_risk_usdt: float = 0.5               # Corte mínimo por símbolo
    max_risk_usdt: float = 20.0              # Corte máximo por símbolo

    # --- portfolio constraints ---
    max_positions: int = 3                   # Máximo de símbolos simultáneos "activos" en decisión
    max_weight_per_symbol: float = 0.70      # Máximo % del presupuesto total en un símbolo
    min_weight_per_symbol: float = 0.05      # Mínimo % si el símbolo fue seleccionado

    # --- correlation control ---
    corr_threshold: float = 0.85             # Umbral "alto" para alertas
    corr_penalty_power: float = 1.5          # Exponente para volver más agresiva la penalización
    hard_block_if_corr_ge: float = 0.98      # Si |corr| >= esto, se bloquea co-existencia (hard guard)

    # --- scoring ---
    score_floor: float = 0.0                # Si score <= floor, se ignora
    normalize_scores: bool = True           # Normalizar scores positivos a sum=1

    # --- logging ---
    enable_logging: bool = True
    logs_dir: str = "logs"
    decisions_csv: str = "allocator_decisions_v1.csv"
    state_json: str = "allocator_state_v1.json"


# ============================================================
# Helpers
# ============================================================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _top_k_symbols(scores: Dict[str, float], k: int) -> List[str]:
    return [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


# ============================================================
# Correlation-based penalty
# ============================================================
def build_pair_penalties(
    corr_df: Optional[pd.DataFrame],
    threshold: float,
    power: float,
) -> Dict[Tuple[str, str], float]:
    """
    Devuelve un dict de penalizaciones por par:
      penalty in [0..1], 0 no penaliza, 1 penaliza fuerte.
    Penaliza por abs(corr)/threshold, saturado, y elevado a power.
    """
    out: Dict[Tuple[str, str], float] = {}
    if corr_df is None or corr_df.empty:
        return out

    syms = list(corr_df.columns)
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            v = corr_df.loc[a, b]
            if pd.isna(v):
                continue
            av = abs(float(v))
            raw = av / float(threshold) if threshold > 0 else av
            p = max(0.0, min(1.0, raw)) ** float(power)
            out[_pair_key(a, b)] = float(p)
    return out


def is_hard_block_pair(
    corr_df: Optional[pd.DataFrame],
    a: str,
    b: str,
    hard_ge: float,
) -> bool:
    if corr_df is None or corr_df.empty:
        return False
    if a not in corr_df.columns or b not in corr_df.columns:
        return False
    v = corr_df.loc[a, b]
    if pd.isna(v):
        return False
    return abs(float(v)) >= float(hard_ge)


# ============================================================
# Main allocator (multi-symbol core)
# ============================================================
def allocate_risk(
    symbol_scores: Dict[str, float],
    corr_df: Optional[pd.DataFrame] = None,
    cfg: Optional[AllocatorConfig] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Entrada:
      symbol_scores: { "SOLUSDT": score, "BTCUSDT": score, ... }
        score = fuerza/edge de la señal (más alto => más asignación)
      corr_df: DataFrame corr(symbol x symbol) de pnl_r (FASE 7)
      cfg: AllocatorConfig
      context: info extra opcional (regime, tf, etc) para logging

    Salida:
      dict con:
        - allocations: {sym: {"weight": w, "risk_usdt": r, "score": s}}
        - blocked_pairs: [(a,b),...]
        - used_symbols: [...]
        - meta: ...
    """
    cfg = cfg or AllocatorConfig()
    context = context or {}

    # 1) filtrar scores
    cleaned: Dict[str, float] = {
        s: _safe_float(v, default=-1e9) for s, v in (symbol_scores or {}).items()
    }
    cleaned = {s: v for s, v in cleaned.items() if v > float(cfg.score_floor)}

    if not cleaned:
        return {
            "allocations": {},
            "blocked_pairs": [],
            "used_symbols": [],
            "meta": {
                "generated_at_utc": _utc_now_iso(),
                "reason": "no_scores_above_floor",
                "cfg": asdict(cfg),
                "context": context,
            },
        }

    # 2) elegir top K por score
    used = _top_k_symbols(cleaned, int(cfg.max_positions))

    # 3) hard blocks por correlación (si aparece una pareja demasiado correlacionada)
    blocked_pairs = []
    final_used = []
    for sym in used:
        ok = True
        for chosen in final_used:
            if is_hard_block_pair(corr_df, sym, chosen, cfg.hard_block_if_corr_ge):
                blocked_pairs.append((chosen, sym))
                ok = False
                break
        if ok:
            final_used.append(sym)

    used = final_used

    if not used:
        return {
            "allocations": {},
            "blocked_pairs": blocked_pairs,
            "used_symbols": [],
            "meta": {
                "generated_at_utc": _utc_now_iso(),
                "reason": "all_blocked_by_hard_corr",
                "cfg": asdict(cfg),
                "context": context,
            },
        }

    # 4) construir penalizaciones por par
    pair_pen = build_pair_penalties(
        corr_df=corr_df,
        threshold=float(cfg.corr_threshold),
        power=float(cfg.corr_penalty_power),
    )

    # 5) convertir scores a weights base
    scores_arr = np.array([max(0.0, float(cleaned[s])) for s in used], dtype=float)
    denom = float(scores_arr.sum())
    base_w = scores_arr / denom if denom > 0 else np.ones_like(scores_arr) / len(scores_arr)

    # 6) aplicar penalización por correlación “intra basket”
    adj_w = base_w.copy()
    for i, a in enumerate(used):
        pens = []
        for j, b in enumerate(used):
            if i == j:
                continue
            p = pair_pen.get(_pair_key(a, b), 0.0)
            pens.append(float(p))
        avg_p = float(np.mean(pens)) if pens else 0.0
        adj_w[i] = adj_w[i] * (1.0 - avg_p)

    # renormalizar
    s_adj = float(adj_w.sum())
    if s_adj > 0:
        adj_w = adj_w / s_adj
    else:
        adj_w = np.ones_like(adj_w) / len(adj_w)

    # 7) aplicar min/max weight por símbolo
    adj_w = np.clip(adj_w, float(cfg.min_weight_per_symbol), float(cfg.max_weight_per_symbol))
    adj_w = adj_w / float(adj_w.sum())

    # 8) mapear a riesgo USDT
    total_risk = float(cfg.total_risk_usdt)
    if len(used) == 1:
        risk_single = min(float(cfg.base_risk_usdt), total_risk)
        risk_usdts = np.array([risk_single], dtype=float)
        weights = np.array([1.0], dtype=float)
    else:
        weights = adj_w
        risk_usdts = weights * total_risk

    # clamp min/max por símbolo
    risk_usdts = np.clip(risk_usdts, float(cfg.min_risk_usdt), float(cfg.max_risk_usdt))

    # si al clipear se pasó del total, renormalizamos hacia abajo proporcionalmente
    sum_r = float(risk_usdts.sum())
    if sum_r > 0 and sum_r > total_risk:
        risk_usdts = risk_usdts * (total_risk / sum_r)

    allocations = {}
    for i, sym in enumerate(used):
        allocations[sym] = {
            "weight": float(weights[i]) if len(weights) == len(used) else float(1.0 / len(used)),
            "risk_usdt": float(risk_usdts[i]),
            "score": float(cleaned[sym]),
        }

    out = {
        "allocations": allocations,
        "blocked_pairs": blocked_pairs,
        "used_symbols": used,
        "meta": {
            "generated_at_utc": _utc_now_iso(),
            "cfg": asdict(cfg),
            "context": context,
        },
    }

    if cfg.enable_logging:
        try:
            _log_allocator_decision(out, cfg=cfg)
        except Exception:
            pass

    return out


# ============================================================
# FASE 7.4 — Stable Allocator Wrapper (single-symbol interface)
# ============================================================
def allocate(
    symbol: str,
    score: float,
    regime: str,
    balance: float,
    risk_mult: float,
    corr_df: Optional[pd.DataFrame] = None,
    cfg: Optional[AllocatorConfig] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Wrapper estable (para Hybrid/ShadowRunner).

    Devuelve SIEMPRE:
      {
        "allow": bool,
        "vol_target": float,   # fracción [0..VOL_TARGET_base*?] que el Hybrid puede usar como override
        "reason": str,
        "penalties": {...},
        "allocations": {...},  # passthrough debug (opcional pero útil)
        "meta": {...},
      }

    Nota:
      - Este wrapper NO cambia la lógica del allocator multi-símbolo.
      - Solo traduce a un contrato estable y aplica risk_mult.
    """
    cfg = cfg or AllocatorConfig()
    context = context or {}
    sym = str(symbol or "").upper().strip()

    sc = _safe_float(score, default=-1e9)
    bal = _safe_float(balance, default=0.0)
    rm = _safe_float(risk_mult, default=1.0)
    rm = _clamp(rm, 0.0, 1.0)

    # Score gate (si no pasa floor -> no trade)
    if sym == "" or sc <= float(cfg.score_floor):
        return {
            "allow": False,
            "vol_target": 0.0,
            "reason": "SCORE_BELOW_FLOOR",
            "penalties": {
                "risk_mult": rm,
                "correlation": 0.0,
            },
            "allocations": {},
            "meta": {
                "generated_at_utc": _utc_now_iso(),
                "cfg": asdict(cfg),
                "context": {**context, "symbol": sym, "regime": str(regime), "score": sc},
            },
        }

    # risk_mult hard block
    if rm <= 0.0:
        return {
            "allow": False,
            "vol_target": 0.0,
            "reason": "RISK_MULT_BLOCK",
            "penalties": {
                "risk_mult": rm,
                "correlation": 0.0,
            },
            "allocations": {},
            "meta": {
                "generated_at_utc": _utc_now_iso(),
                "cfg": asdict(cfg),
                "context": {**context, "symbol": sym, "regime": str(regime), "score": sc},
            },
        }

    # Balance sanity (si no hay equity, no se puede convertir riesgo->vol_target)
    if bal <= 0:
        return {
            "allow": False,
            "vol_target": 0.0,
            "reason": "NO_BALANCE",
            "penalties": {
                "risk_mult": rm,
                "correlation": 0.0,
            },
            "allocations": {},
            "meta": {
                "generated_at_utc": _utc_now_iso(),
                "cfg": asdict(cfg),
                "context": {**context, "symbol": sym, "regime": str(regime), "score": sc},
            },
        }

    # Llamada al core multi-symbol (single-symbol basket)
    payload = allocate_risk(
        symbol_scores={sym: sc},
        corr_df=corr_df,
        cfg=cfg,
        context={**context, "symbol": sym, "regime": str(regime)},
    )

    allocs = payload.get("allocations", {}) or {}
    used = payload.get("used_symbols", []) or []
    meta = payload.get("meta", {}) or {}
    reason_core = str(meta.get("reason") or "OK")

    # Si el símbolo no quedó en used/allocs -> bloqueado por hard corr o similar
    if sym not in allocs:
        # razón más explícita
        if reason_core == "all_blocked_by_hard_corr":
            rsn = "HARD_BLOCK_CORRELATION"
        elif reason_core == "no_scores_above_floor":
            rsn = "SCORE_BELOW_FLOOR"
        else:
            rsn = f"ALLOCATOR_BLOCK_{reason_core}"
        return {
            "allow": False,
            "vol_target": 0.0,
            "reason": rsn,
            "penalties": {
                "risk_mult": rm,
                "correlation": 1.0 if rsn == "HARD_BLOCK_CORRELATION" else 0.0,
            },
            "allocations": allocs,
            "meta": {
                "generated_at_utc": meta.get("generated_at_utc", _utc_now_iso()),
                "cfg": meta.get("cfg", asdict(cfg)),
                "context": meta.get("context", {**context, "symbol": sym, "regime": str(regime)}),
                "used_symbols": used,
                "blocked_pairs": payload.get("blocked_pairs", []) or [],
            },
        }

    # Convertir risk_usdt -> vol_target (fracción de equity)
    risk_usdt = _safe_float(allocs[sym].get("risk_usdt"), default=0.0)
    base_vol_target = risk_usdt / bal  # fracción de equity aproximada
    base_vol_target = max(0.0, base_vol_target)

    # Aplicar risk_mult (softening institucional)
    vol_target = base_vol_target * rm

    # Clamp final defensivo (evita errores humanos si alguien toca knobs)
    vol_target = _clamp(vol_target, 0.0, 1.0)

    # Penalización “correlation” estimada (single-symbol = 0, pero dejamos hook)
    corr_pen = 0.0
    # si en el futuro pasás más símbolos al wrapper, acá se puede poblar con avg_penalty.

    return {
        "allow": vol_target > 0.0,
        "vol_target": float(vol_target),
        "reason": "OK" if vol_target > 0.0 else "VOL_TARGET_ZERO",
        "penalties": {
            "risk_mult": float(rm),
            "correlation": float(corr_pen),
        },
        "allocations": allocs,  # debug passthrough
        "meta": {
            "generated_at_utc": meta.get("generated_at_utc", _utc_now_iso()),
            "cfg": meta.get("cfg", asdict(cfg)),
            "context": meta.get("context", {**context, "symbol": sym, "regime": str(regime)}),
            "used_symbols": used,
            "blocked_pairs": payload.get("blocked_pairs", []) or [],
            "risk_usdt_raw": float(risk_usdt),
            "balance": float(bal),
            "score": float(sc),
        },
    }


# ============================================================
# Logging
# ============================================================
def _log_allocator_decision(payload: Dict[str, Any], cfg: AllocatorConfig) -> None:
    logs_dir = cfg.logs_dir
    _ensure_dir(logs_dir)

    # 1) append CSV rows (one per symbol allocation)
    csv_path = os.path.join(logs_dir, cfg.decisions_csv)
    rows = []
    meta = payload.get("meta", {}) or {}
    ctx = meta.get("context", {}) or {}
    ts = meta.get("generated_at_utc", _utc_now_iso())

    allocs = payload.get("allocations", {}) or {}
    blocked = payload.get("blocked_pairs", []) or []

    for sym, d in allocs.items():
        rows.append({
            "ts_utc": ts,
            "symbol": sym,
            "score": _safe_float(d.get("score"), 0.0),
            "weight": _safe_float(d.get("weight"), 0.0),
            "risk_usdt": _safe_float(d.get("risk_usdt"), 0.0),
            "blocked_pairs": json.dumps(blocked, ensure_ascii=False),
            "context": json.dumps(ctx, ensure_ascii=False),
        })

    if rows:
        df = pd.DataFrame(rows)
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", index=False, header=write_header)

    # 2) write state JSON (last decision)
    state_path = os.path.join(logs_dir, cfg.state_json)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
