from __future__ import annotations

import json
import math
import os 
from typing import Dict, List, Any


def _expand_local(center: Any, radius: int = 1, step: int = 1):
    if isinstance(center, bool):
        return [center]
    if isinstance(center, int):
        return list(range(center - radius, center + radius + 1, step))
    if isinstance(center, float):
        return [round(center * (1 + x), 4) for x in (-0.1, 0, 0.1)]
    return [center]


def make_local_grid(
    params: Dict[str, Any],
    radius_map: Dict[str, int],
) -> Dict[str, List[Any]]:
    grid = {}
    for k, v in params.items():
        r = radius_map.get(k, 0)
        grid[k] = _expand_local(v, radius=r)
    return grid


def robust_results_to_grids(
    robust_json: str,
    out_dir: str,
    top_n: int = 5,
):
    with open(robust_json, "r", encoding="utf-8") as f:
        robust = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    for i, r in enumerate(robust[:top_n]):
        params = r["params"]

        # radio pequeño → anti overfit
        grid = make_local_grid(
            params,
            radius_map={
                "VWAP_WINDOW": 1,
                "VWAP_BAND_MULT": 1,
                "min_score_long": 1,
                "min_score_short": 1,
            },
        )

        path = f"{out_dir}/grid_refine_{i:02d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(grid, f, indent=2, ensure_ascii=False)

        print(f"[ROBUST→GRID] {path}")

        
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("robust_json")
    ap.add_argument("out_dir")
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    robust_results_to_grids(
        robust_json=args.robust_json,
        out_dir=args.out_dir,
        top_n=args.top_n,
    )
