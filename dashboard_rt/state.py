# dashboard_rt/state.py
import os
import json
import time
from datetime import datetime

# ============================================================
# Config
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DEFAULT_PATH = os.path.join(ROOT, "logs", "runtime_state.json")

# Cache (último estado válido)
_LAST_GOOD = None
_LAST_GOOD_TS = 0.0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _fmt_updated(ms: int) -> str:
    try:
        dt = datetime.fromtimestamp(ms / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def _normalize_to_frontend(raw: dict, path: str) -> dict:
    """
    Target (lo que app.js espera):
    {
      mode, status, age_sec, updated_at, _path,
      symbols: {
        "SOLUSDT": {
          status, side, qty, regime,
          last_price, unrealized_pnl,
          equity_r, dd_day_r
        }
      }
    }
    """
    now_ms = _now_ms()

    # --- si YA viene normalizado ---
    if isinstance(raw.get("symbols"), dict) and "age_sec" in raw and "status" in raw:
        out = dict(raw)
        out["_path"] = path
        return out

    gen_ms = _safe_int(
        raw.get("generated_at_ms")
        or raw.get("ts_ms")
        or raw.get("generated_at")
        or 0,
        0,
    )
    if gen_ms <= 0:
        gen_ms = now_ms

    age_sec = max(0, int((now_ms - gen_ms) / 1000))
    updated_at = _fmt_updated(gen_ms)
    mode = str(raw.get("mode") or "shadow")

    # --- detectar símbolo (unisímbolo tolerante) ---
    sym = None
    if isinstance(raw.get("symbols"), list) and raw["symbols"]:
        sym = str(raw["symbols"][0]).upper()
    elif isinstance(raw.get("by_symbol"), dict) and raw["by_symbol"]:
        sym = str(next(iter(raw["by_symbol"].keys()))).upper()

    snap = None
    if sym and isinstance(raw.get("by_symbol"), dict):
        snap = raw["by_symbol"].get(sym)

    # defaults
    status = "NO_DATA"
    side = "—"
    qty = 0.0
    regime = "UNKNOWN"
    last_price = 0.0
    unreal = 0.0
    equity_r = 0.0
    dd_day_r = 0.0

    if isinstance(snap, dict):
        pos = snap.get("position", {})
        met = snap.get("metrics", {})

        side = str(pos.get("side") or "FLAT").upper()
        qty = _safe_float(pos.get("qty"), 0.0)
        regime = str(pos.get("regime") or "UNKNOWN").upper()
        last_price = _safe_float(snap.get("last_price"), 0.0)
        unreal = _safe_float(pos.get("unrealized_pnl"), 0.0)

        equity_r = _safe_float(
            met.get("equity_r") or raw.get("metrics", {}).get("equity_r"), 0.0
        )
        dd_day_r = _safe_float(
            met.get("dd_day") or raw.get("metrics", {}).get("dd_day"), 0.0
        )

        status = "OPEN" if side in ("LONG", "SHORT") and qty > 0 else "FLAT"

    symbols_obj = {}
    if sym:
        symbols_obj[sym] = {
            "symbol": sym,
            "status": status,
            "side": side,
            "qty": qty,
            "regime": regime,
            "last_price": last_price,
            "unrealized_pnl": unreal,
            "equity_r": equity_r,
            "dd_day_r": dd_day_r,
        }

    global_status = (
        "RUNNING"
        if symbols_obj and age_sec <= 3
        else ("STALE" if symbols_obj else "NO_DATA")
    )

    return {
        "mode": mode,
        "status": global_status,
        "age_sec": age_sec,
        "updated_at": updated_at,
        "_path": path,
        "symbols": symbols_obj,
    }


def read_runtime_state(path: str = DEFAULT_PATH) -> dict:
    """
    Lectura segura + cache:
    - si falla lectura/parsing -> último bueno
    - si no existe -> placeholder estable
    """
    global _LAST_GOOD, _LAST_GOOD_TS

    base = {
        "mode": "shadow",
        "status": "NO_DATA",
        "age_sec": None,
        "updated_at": "—",
        "_path": path,
        "symbols": {},
    }

    try:
        if not os.path.exists(path):
            return _LAST_GOOD or base

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        out = _normalize_to_frontend(raw, path)
        _LAST_GOOD = out
        _LAST_GOOD_TS = time.time()
        return out

    except Exception:
        return _LAST_GOOD or base
