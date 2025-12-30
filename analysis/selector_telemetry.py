from __future__ import annotations

import json
import hashlib
import os
import time
from typing import Any, Dict, Optional


def _now_ts() -> int:
    return int(time.time())


def _hash_params(params: Dict[str, Any]) -> str:
    s = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def emit_selector_telemetry(
    *,
    path: str,
    symbol: str,
    params: Dict[str, Any],
    selector_meta: Dict[str, Any],
) -> None:
    payload = {
        "ts": _now_ts(),
        "symbol": symbol,
        "source": selector_meta.get("selector"),
        "score": selector_meta.get("score"),
        "robust_path": selector_meta.get("robust_path"),
        "regime": selector_meta.get("regime"),
        "params_hash": _hash_params(params),
        "selector": selector_meta.get("selector"),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
