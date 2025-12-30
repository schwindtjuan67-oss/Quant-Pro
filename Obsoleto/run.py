# run.py
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional


def load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


async def run_shadow(config: Dict[str, Any], symbol: Optional[str], interval: Optional[str]) -> None:
    from Live.shadow_runner import main as shadow_main

    sym = symbol or config.get("symbol") or "SOLUSDT"
    itv = interval or config.get("interval") or "1m"
    await shadow_main(symbol=sym, interval=itv, config=config)


async def run_live(config: Dict[str, Any], symbol: Optional[str], interval: Optional[str]) -> None:
    """
    Placeholder industrial:
    - Cuando tengas LiveRunner real, lo conectamos acá.
    - Por ahora, no simulamos LIVE en silencio.
    """
    raise NotImplementedError("LIVE runner todavía no está cableado. Usá mode=shadow por ahora.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["shadow", "live"], default="shadow")
    p.add_argument("--config", default="configs/shadow_prod.json")
    p.add_argument("--symbol", default=None)
    p.add_argument("--interval", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)

    if args.mode == "shadow":
        asyncio.run(run_shadow(cfg, args.symbol, args.interval))
        return

    asyncio.run(run_live(cfg, args.symbol, args.interval))


if __name__ == "__main__":
    main()
