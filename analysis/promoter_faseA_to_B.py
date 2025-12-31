#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Dict, Any, List, Tuple


SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)
WINDOW_RE = re.compile(r"robust_(\d{4}-\d{2}_\d{4}-\d{2})", re.IGNORECASE)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_seed_window(source_name: str) -> Tuple[str, str]:
    seed = "unknown"
    window = "unknown"
    m = SEED_RE.search(source_name)
    if m:
        seed = m.group(1)
    m = WINDOW_RE.search(source_name)
    if m:
        window = m.group(1)
    return seed, window


def main():
    ap = argparse.ArgumentParser("promoter_faseA_to_B")
    ap.add_argument("--in", dest="inp", default="results/post_analysis_summary.json")
    ap.add_argument("--rules", default="configs/promotion_rules_faseA_to_B.json")
    ap.add_argument("--out", default="results/promotions/faseA_promoted.json")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    data = load_json(args.inp)
    rules = load_json(args.rules)

    # compat: si el post viejo era una lista, lo adaptamos
    if isinstance(data, list):
        summary = data
    else:
        summary = data.get("summary", [])

    # gates
    min_appear = int(rules.get("min_appearances", 2))
    min_unique_seeds = int(rules.get("min_unique_seeds", 2))
    min_unique_windows = int(rules.get("min_unique_windows", 2))

    min_score_mean = float(rules.get("min_score_mean", -1e9))
    max_score_std = float(rules.get("max_score_std", 1e9))
    min_score_min = float(rules.get("min_score_min", -1e9))

    # opcional: scalper heavy trading
    min_trades_mean = rules.get("min_trades_mean", None)
    if min_trades_mean is not None:
        min_trades_mean = float(min_trades_mean)

    selected: List[Dict[str, Any]] = []

    for s in summary:
        if not isinstance(s, dict):
            continue

        appearances = int(s.get("appearances", 0))
        if appearances < min_appear:
            continue

        score_mean = float(s.get("score_mean", -1e9))
        score_std = float(s.get("score_std", 1e9))
        score_min = float(s.get("score_min", -1e9))

        if score_mean < min_score_mean:
            continue
        if score_std > max_score_std:
            continue
        if score_min < min_score_min:
            continue

        trades_mean = s.get("trades_mean", None)
        if min_trades_mean is not None:
            if trades_mean is None or float(trades_mean) < min_trades_mean:
                continue

        sources = s.get("sources", []) or []
        seeds = set()
        windows = set()
        for src in sources:
            seed, window = parse_seed_window(str(src))
            seeds.add(seed)
            windows.add(window)

        if len(seeds) < min_unique_seeds:
            continue
        if len(windows) < min_unique_windows:
            continue

        selected.append({
            "params": s.get("params", {}) or {},
            "appearances": appearances,
            "score_mean": score_mean,
            "score_std": score_std,
            "score_min": score_min,
            "score_max": float(s.get("score_max", score_mean)),
            "trades_mean": trades_mean,
            "unique_seeds": sorted(seeds),
            "unique_windows": sorted(windows),
            "sources": sorted(set(map(str, sources))),
        })

    selected.sort(
        key=lambda x: (x["appearances"], x["score_mean"], -x["score_std"]),
        reverse=True,
    )

    payload = {
        "phase_from": "A",
        "phase_to": "B",
        "rules_used": rules,
        "selected_count": min(len(selected), int(args.top)),
        "selected": selected[: int(args.top)],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[PROMOTER] input={args.inp}")
    print(f"[PROMOTER] rules={args.rules}")
    print(f"[PROMOTER] selected={payload['selected_count']}")
    print(f"[PROMOTER] saved -> {args.out}")


if __name__ == "__main__":
    main()
