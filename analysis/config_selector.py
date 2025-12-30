# analysis/config_selector.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


def _now_ts() -> int:
    return int(time.time())


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _upper(x: str) -> str:
    return str(x or "").upper().strip()


def _load_json(path: str) -> Optional[Any]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _as_items(payload: Any) -> List[Dict[str, Any]]:
    """
    Soporta:
      - {"items":[...]} ✅ (top_k_library / indexador)
      - {"top_k":[...]} / {"rows":[...]} ...
      - list[dict]
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in ("items", "top_k", "rows", "configs", "results"):
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _get_meta(it: Dict[str, Any]) -> Dict[str, Any]:
    m = it.get("meta")
    return m if isinstance(m, dict) else {}


# ==============================
# ✅ PATCH: soporte de "metrics"
# ==============================
def _get_metrics(it: Dict[str, Any]) -> Dict[str, Any]:
    """
    Soporta:
      - it["metrics"] (pipeline output)
      - it["meta"]["metrics"] (compat extra)
    """
    m = it.get("metrics")
    if isinstance(m, dict):
        return m
    meta = _get_meta(it)
    mm = meta.get("metrics")
    if isinstance(mm, dict):
        return mm
    return {}


def _get_metric(it: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Busca key en:
      1) top-level
      2) metrics (pipeline)
      3) meta
      4) meta.metrics (compat)
    """
    # 1) top-level
    for k in keys:
        if k in it:
            return it.get(k)

    # 2) metrics (nuevo pipeline)
    metrics = _get_metrics(it)
    for k in keys:
        if k in metrics:
            return metrics.get(k)

    # 3) meta
    meta = _get_meta(it)
    for k in keys:
        if k in meta:
            return meta.get(k)

    # 4) meta.metrics (por si vino anidado)
    mm = meta.get("metrics")
    if isinstance(mm, dict):
        for k in keys:
            if k in mm:
                return mm.get(k)

    return default


def _extract_regime(it: Dict[str, Any]) -> str:
    reg = str(_get_metric(it, "regime", "REGIME", default="") or "").upper().strip()
    return reg


def _extract_params(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 1) library/indexador: params top-level
    if isinstance(it.get("params"), dict):
        return dict(it["params"])

    # 2) robust/best_config style
    bc = it.get("best_config")
    if isinstance(bc, dict):
        if isinstance(bc.get("strategy_params"), dict):
            return dict(bc["strategy_params"])
        strat = bc.get("strategy")
        if isinstance(strat, dict) and isinstance(strat.get("params"), dict):
            return dict(strat["params"])

    # 3) meta payloads
    meta = _get_meta(it)
    if isinstance(meta.get("params"), dict):
        return dict(meta["params"])

    # 4) compat vieja
    for k in ("config", "hyperparams", "best_params"):
        if isinstance(it.get(k), dict):
            return dict(it[k])

    return None


def _pick_rank_value(it: Dict[str, Any]) -> float:
    """
    Orden:
      1) score
      2) sortino
      3) sharpe
      4) tstat_pnl_r / avg_pnl_r
    (soporta top-level o meta o metrics)
    """
    for k in ("score", "sortino", "sharpe", "tstat_pnl_r", "avg_pnl_r", "sharpe_pnl_r"):
        v = _get_metric(it, k, default=None)
        if v is not None:
            return _safe_float(v, 0.0)
    return 0.0


class ConfigSelector:
    """
    Selector:
      - Lee top_k.json o top_k_library.json
      - Filtra por symbol y (opcional) regime
      - Filtros hard: min_trades, max_dd_limit
      - Cache por mtime
    """

    def __init__(
        self,
        *,
        top_k_path: str = "logs/top_k.json",
        min_trades: int = 12,
        max_dd_limit: float = 0.30,
        cooldown_sec: int = 1800,
    ):
        self.top_k_path = top_k_path
        self.min_trades = int(min_trades)
        self.max_dd_limit = float(max_dd_limit)
        self.cooldown_sec = int(cooldown_sec)

        self._cache_mtime: float = -1.0
        self._cache_items: List[Dict[str, Any]] = []

    def _refresh(self) -> None:
        try:
            mtime = os.path.getmtime(self.top_k_path)
        except Exception:
            mtime = -1.0

        if mtime == self._cache_mtime:
            return

        payload = _load_json(self.top_k_path)
        items = _as_items(payload)

        out: List[Dict[str, Any]] = []
        for it in items:
            sym = _upper(_get_metric(it, "symbol", "SYM", "ticker", default=""))
            if sym:
                it["_symbol_u"] = sym

            reg = _extract_regime(it)
            if reg:
                it["_regime_u"] = reg

            out.append(it)

        self._cache_items = out
        self._cache_mtime = mtime

    def select(
        self,
        *,
        symbol: str,
        regime: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        self._refresh()

        sym = _upper(symbol)
        reg = str(regime or "").upper().strip()

        candidates = [it for it in self._cache_items if it.get("_symbol_u") == sym]

        if reg:
            reg_match = [it for it in candidates if it.get("_regime_u") == reg]
            if reg_match:
                candidates = reg_match
            else:
                candidates = [it for it in candidates if not it.get("_regime_u")]

        filtered: List[Dict[str, Any]] = []
        for it in candidates:
            n_trades = _safe_int(_get_metric(it, "n_trades", "trades", "Trades", default=0), 0)
            max_dd = _safe_float(_get_metric(it, "max_dd_r", "max_dd", "MaxDD", "dd", default=0.0), 0.0)

            if n_trades < self.min_trades:
                continue
            if max_dd > self.max_dd_limit:
                continue

            filtered.append(it)

        filtered.sort(key=_pick_rank_value, reverse=True)

        best = filtered[0] if filtered else None
        params = _extract_params(best) if isinstance(best, dict) else None

        selector_meta: Dict[str, Any] = {
            "selector": "ConfigSelector",
            "top_k_path": self.top_k_path,
            "top_k_mtime": self._cache_mtime,
            "symbol": sym,
            "regime": reg or None,
            "min_trades": self.min_trades,
            "max_dd_limit": self.max_dd_limit,
            "cooldown_sec": self.cooldown_sec,
            "candidates": len(candidates),
            "filtered": len(filtered),
            "picked": bool(params is not None),
            "picked_rank": _pick_rank_value(best) if best else None,
            "picked_item": None,
            "ts": _now_ts(),
        }

        if isinstance(best, dict):
            meta = _get_meta(best)
            selector_meta["picked_item"] = {
                "score": _get_metric(best, "score", default=meta.get("score")),
                "n_trades": _get_metric(best, "n_trades", "trades", default=meta.get("n_trades")),
                "max_dd_r": _get_metric(best, "max_dd_r", "max_dd", default=meta.get("max_dd_r")),
                "window": _get_metric(best, "window", default=meta.get("window")),
                "seed": _get_metric(best, "seed", default=meta.get("seed")),
                "source_files": _get_metric(best, "source_files", default=meta.get("source_files")),
                "regime": best.get("_regime_u") or meta.get("regime"),
            }

        return params, selector_meta

