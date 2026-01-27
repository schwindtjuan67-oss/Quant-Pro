#!/usr/bin/env python3
# analysis/stage_b_from_survivors.py
# Fase A -> B handoff contract
"""Stage B handoff consumer: revalidación + risk sanity (sin optimizar).

Lee survivors.json de Fase A, revalida cada survivor con BacktestRunner
y aplica gates_B simples para filtrar candidatos.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from analysis.grid_metrics import compute_metrics_from_trades
from analysis.robust_optimizer import load_candles_from_path, window_to_dates, _map_params_to_hybrid_kwargs
from backtest.backtest_runner import BacktestRunner


def _atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _parse_env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _parse_env_float(name: str) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _normalize_max_dd_r_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if out > 1.0:
        out = out / 100.0
    return out


def _resolve_gates_b(
    cli_min_trades: Optional[int],
    cli_min_pf: Optional[float],
    cli_min_winrate: Optional[float],
    cli_max_dd_r: Optional[float],
) -> Dict[str, float]:
    gates = {
        "min_trades": 150,
        "min_pf": 1.10,
        "min_winrate": 0.30,
        "max_dd_r": 0.35,
    }

    env_min_trades = _parse_env_int("STAGE_B_MIN_TRADES")
    if env_min_trades is not None:
        gates["min_trades"] = env_min_trades
    env_min_pf = _parse_env_float("STAGE_B_MIN_PF")
    if env_min_pf is not None:
        gates["min_pf"] = env_min_pf
    env_min_winrate = _parse_env_float("STAGE_B_MIN_WINRATE")
    if env_min_winrate is not None:
        gates["min_winrate"] = env_min_winrate
    env_max_dd_r = _parse_env_float("STAGE_B_MAX_DD_R")
    if env_max_dd_r is not None:
        gates["max_dd_r"] = env_max_dd_r

    if cli_min_trades is not None:
        gates["min_trades"] = int(cli_min_trades)
    if cli_min_pf is not None:
        gates["min_pf"] = float(cli_min_pf)
    if cli_min_winrate is not None:
        gates["min_winrate"] = float(cli_min_winrate)
    if cli_max_dd_r is not None:
        gates["max_dd_r"] = float(cli_max_dd_r)

    normalized = _normalize_max_dd_r_value(gates.get("max_dd_r"))
    if normalized is not None:
        gates["max_dd_r"] = normalized

    return gates


def _load_survivors_payload(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("survivors"), list):
        return payload.get("meta") or {}, payload.get("survivors") or []
    if isinstance(payload, list):
        return {}, payload
    raise RuntimeError(f"[B] Unsupported survivors payload at {path}")


def _normalize_survivor_list(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    survivors: List[Dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        params = item.get("params")
        if isinstance(params, dict):
            survivors.append(item)
        else:
            survivors.append({"params": item})
    return survivors


def _build_backtest_config(base_cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    cfg_payload = json.loads(json.dumps(base_cfg)) if isinstance(base_cfg, dict) else {}
    cfg_payload.setdefault("params", {})
    cfg_payload["params"].update(params or {})

    strategy_kwargs = dict(params or {})
    mapped_kwargs = _map_params_to_hybrid_kwargs(params or {})
    for key, value in mapped_kwargs.items():
        strategy_kwargs.setdefault(key, value)

    cfg_payload["strategy"] = {
        "name": "hybrid_scalper_pro",
        "kwargs": strategy_kwargs,
    }
    cfg_payload["strategy_kwargs"] = strategy_kwargs
    cfg_payload["params"] = dict(cfg_payload.get("params") or {})
    return cfg_payload


def _passes_gates(metrics: Dict[str, float], gates: Dict[str, float]) -> Tuple[bool, str]:
    trades = metrics.get("trades", 0)
    if trades < gates.get("min_trades", 0):
        return False, f"min_trades ({trades})"

    pf = metrics.get("profit_factor", 0.0)
    if pf < gates.get("min_pf", 0.0):
        return False, f"min_pf ({pf:.2f})"

    winrate = metrics.get("winrate", 0.0)
    if winrate < gates.get("min_winrate", 0.0):
        return False, f"min_winrate ({winrate:.2f})"

    max_dd = abs(metrics.get("max_drawdown_r", 0.0))
    if max_dd > gates.get("max_dd_r", 1.0):
        return False, f"max_dd_r ({max_dd:.2f})"

    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser("stage_b_from_survivors")
    ap.add_argument("--in", dest="input_path", required=True, help="survivors.json de Fase A")
    ap.add_argument("--base-config", required=True, help="config base para BacktestRunner")
    ap.add_argument("--out", default=os.path.join("results", "promotions", "faseB_candidates.json"))
    ap.add_argument("--symbol", default=None, help="override symbol (default: meta/base config)")
    ap.add_argument("--interval", default=None, help="override interval (default: meta/base config)")
    ap.add_argument("--window", default=None, help="override window YYYY-MM_YYYY-MM")
    ap.add_argument("--warmup", type=int, default=500, help="candles warmup")
    ap.add_argument("--min-trades", type=int, default=None, help="gate B min_trades")
    ap.add_argument("--min-pf", type=float, default=None, help="gate B min_pf")
    ap.add_argument("--min-winrate", type=float, default=None, help="gate B min_winrate")
    ap.add_argument("--max-dd-r", type=float, default=None, help="gate B max_dd_r")
    args = ap.parse_args()

    meta_in, raw_survivors = _load_survivors_payload(args.input_path)
    survivors = _normalize_survivor_list(raw_survivors)
    if not survivors:
        raise SystemExit("[B] No survivors found in input.")

    data_spec = meta_in.get("data_spec") if isinstance(meta_in, dict) else None
    data_path = None
    if isinstance(data_spec, dict):
        data_path = data_spec.get("input_path") or data_spec.get("path")
    if not data_path:
        raise SystemExit("[B] Missing data_spec.input_path in survivors meta. Re-run Fase A with --save-survivors.")

    window = args.window or meta_in.get("window")
    if not window:
        raise SystemExit("[B] Missing window (use --window or ensure survivors meta contains window).")
    date_from, date_to = window_to_dates(window)

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    symbol = (args.symbol or meta_in.get("symbol") or base_cfg.get("symbol") or (base_cfg.get("symbols") or ["SOLUSDT"])[0]).upper()
    interval = args.interval or meta_in.get("interval") or (data_spec.get("interval") if isinstance(data_spec, dict) else None) or base_cfg.get("interval") or "1m"

    candles = load_candles_from_path(
        data_path,
        date_from=date_from,
        date_to=date_to,
        debug=False,
        interval=interval,
        data_spec=data_spec if isinstance(data_spec, dict) else None,
    )
    if not candles:
        raise SystemExit("[B] Loaded 0 candles. Check data_spec and window coverage.")

    gates_b = _resolve_gates_b(args.min_trades, args.min_pf, args.min_winrate, args.max_dd_r)

    candidates: List[Dict[str, Any]] = []
    for idx, item in enumerate(survivors, start=1):
        params = item.get("params") if isinstance(item, dict) else {}
        robust_score = item.get("robust_score") if isinstance(item, dict) else None

        cfg_payload = _build_backtest_config(base_cfg, params or {})
        runner = BacktestRunner(
            config=cfg_payload,
            candles=candles,
            symbol=symbol,
            interval=interval,
            warmup_candles=int(args.warmup),
        )
        result = runner.run() or {}
        trades_list = result.get("trades_list") if isinstance(result, dict) else None
        metrics = compute_metrics_from_trades(trades_list or [])
        pass_b, reason = _passes_gates(metrics, gates_b)

        candidates.append({
            "rank": idx,
            "robust_score_from_A": robust_score,
            "params": params,
            "metrics_B": metrics,
            "pass_B": bool(pass_b),
            "fail_reason_B": reason or None,
        })

    out_payload = {
        "meta": {
            "input_file": args.input_path,
            "created_at_iso": datetime.now(timezone.utc).isoformat(),
            "base_config_path": args.base_config,
            "symbol": symbol,
            "interval": interval,
            "window": window,
            "gates_B": gates_b,
        },
        "candidates": candidates,
    }

    _atomic_write_json(args.out, out_payload)
    print(f"[B] saved -> {args.out}")


if __name__ == "__main__":
    main()
