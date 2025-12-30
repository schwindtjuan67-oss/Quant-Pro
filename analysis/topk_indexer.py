# analysis/topk_indexer.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional


def _now_ts() -> int:
    return int(time.time())


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _hash_params(params: Dict[str, Any]) -> str:
    s = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _extract_params(best_cfg_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Soporta best_config.json típicos:
      - {"best_config": {"strategy_params": {...}}, "score": ...}
      - {"best_config": {"strategy": {"params": {...}}}, "score": ...}
      - {"strategy_params": {...}}
    """
    if isinstance(best_cfg_payload.get("strategy_params"), dict):
        return dict(best_cfg_payload["strategy_params"])

    bc = best_cfg_payload.get("best_config")
    if isinstance(bc, dict):
        if isinstance(bc.get("strategy_params"), dict):
            return dict(bc["strategy_params"])
        strat = bc.get("strategy")
        if isinstance(strat, dict) and isinstance(strat.get("params"), dict):
            return dict(strat["params"])

    return None


def _extract_score(best_cfg_payload: Dict[str, Any]) -> float:
    for k in ("score", "robust_score", "picked_rank"):
        v = best_cfg_payload.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    # a veces viene en best_config/meta
    bc = best_cfg_payload.get("best_config")
    if isinstance(bc, dict):
        v = bc.get("score")
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return 0.0


def _infer_from_path(path: str) -> Dict[str, Optional[str]]:
    """
    Si el path viene como:
      results/grid_refine/SOLUSDT/TREND/seed_1337/2020-01_2021-12/best_config.json
    intenta inferir symbol/regime/seed/window.
    """
    p = os.path.normpath(path)
    parts = p.split(os.sep)

    out = {"symbol": None, "regime": None, "seed": None, "window": None}
    for i, part in enumerate(parts):
        if part.lower().startswith("seed_"):
            out["seed"] = part.split("_", 1)[1]
            if i >= 2:
                out["regime"] = parts[i - 1]
                out["symbol"] = parts[i - 2]
            if i + 1 < len(parts):
                out["window"] = parts[i + 1]
            break
    return out


def main():
    ap = argparse.ArgumentParser("topk_indexer")
    ap.add_argument("--input", required=True, help="path to best_config.json")
    ap.add_argument("--library", required=True, help="output library json (top_k_library.json)")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--regime", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--window", default=None)
    ap.add_argument("--max_items", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    payload = _load_json(args.input)
    if not isinstance(payload, dict):
        raise RuntimeError("best_config.json inválido (no dict)")

    params = _extract_params(payload)
    if not isinstance(params, dict):
        raise RuntimeError("No pude extraer params desde best_config.json")

    score = _extract_score(payload)

    inferred = _infer_from_path(args.input)
    symbol = (args.symbol or inferred["symbol"] or "UNKNOWN").upper()
    regime = (args.regime or inferred["regime"] or "UNKNOWN").upper()
    seed = str(args.seed or inferred["seed"] or "")
    window = str(args.window or inferred["window"] or "")

    item = {
        "ts": _now_ts(),
        "symbol": symbol,
        "regime": regime,
        "window": window or None,
        "seed": seed or None,
        "score": score,
        "params": params,
        "params_hash": _hash_params(params),
        "source_file": os.path.normpath(args.input),
        "meta": {
            "producer": "topk_indexer",
        },
    }

    lib = {"version": 1, "updated_ts": _now_ts(), "items": []}
    if os.path.exists(args.library):
        try:
            existing = _load_json(args.library)
            if isinstance(existing, dict) and isinstance(existing.get("items"), list):
                lib = existing
        except Exception:
            pass

    items = lib.get("items") if isinstance(lib.get("items"), list) else []
    # dedupe por (symbol,regime,window,seed,params_hash)
    key = (symbol, regime, item["window"], item["seed"], item["params_hash"])
    seen = set()
    new_items = []
    for it in items:
        if not isinstance(it, dict):
            continue
        k = (
            str(it.get("symbol", "")).upper(),
            str(it.get("regime", "")).upper(),
            it.get("window"),
            it.get("seed"),
            it.get("params_hash"),
        )
        if k in seen:
            continue
        seen.add(k)
        new_items.append(it)

    if key not in seen:
        new_items.append(item)

    # ordenar por score desc
    new_items.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    if args.max_items and args.max_items > 0:
        new_items = new_items[: args.max_items]

    lib["items"] = new_items
    lib["updated_ts"] = _now_ts()
    _save_json(args.library, lib)

    print(f"[INDEX] added={key not in seen} items={len(new_items)} -> {args.library}")


if __name__ == "__main__":
    main()
