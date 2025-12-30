from __future__ import annotations

import json
import glob
import os
from typing import Dict, Any, List


LIB_ROOT = "results/library"
OUT_PATH = "logs/top_k_library.json"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_library() -> Dict[str, Any]:
    """
    Recorre results/library/**/best__seed_*.json
    y construye un top_k compatible con ConfigSelector.
    """
    items: List[Dict[str, Any]] = []

    pattern = os.path.join(
        LIB_ROOT,
        "**",
        "best__seed_*.json",
    )

    for best_path in glob.glob(pattern, recursive=True):
        base_dir = os.path.dirname(best_path)
        fname = os.path.basename(best_path)

        # seed desde el nombre
        # best__seed_0001.json
        seed = fname.replace("best__seed_", "").replace(".json", "")

        # meta correspondiente
        meta_path = os.path.join(base_dir, f"meta__seed_{seed}.json")
        if not os.path.exists(meta_path):
            print(f"[LIB][WARN] meta missing for {best_path}")
            continue

        try:
            params = load_json(best_path)
            meta = load_json(meta_path)
        except Exception as e:
            print(f"[LIB][ERROR] {best_path}: {e}")
            continue

        item = {
            "params": params,
            "meta": {
                "symbol": meta.get("symbol"),
                "regime": meta.get("regime"),
                "window": meta.get("window"),
                "from_date": meta.get("from_date"),
                "to_date": meta.get("to_date"),
                "seed": meta.get("seed"),
                "source": best_path.replace("\\", "/"),
            },
        }

        items.append(item)

    return {
        "generated_from": LIB_ROOT,
        "count": len(items),
        "items": items,
    }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    payload = build_library()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[LIB] written {OUT_PATH}")
    print(f"[LIB] total configs = {payload['count']}")


if __name__ == "__main__":
    main()
