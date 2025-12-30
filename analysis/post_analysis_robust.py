#!/usr/bin/env python3
import json
import glob
import os
import hashlib
import numpy as np
from collections import defaultdict

RESULTS_DIR = "results/robust"
OUT_FILE = "results/post_analysis_summary.json"

def params_key(params: dict) -> str:
    """Hash estable del dict de params"""
    s = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def load_all_results():
    rows = []
    for fp in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            rows.append({
                "params": r["params"],
                "key": params_key(r["params"]),
                "robust_score": r.get("agg", {}).get("robust_score", -1e9),
                "passed": bool(r.get("passed", False)),
                "fail_reason": r.get("fail_reason", ""),
                "source": os.path.basename(fp),
            })
    return rows

def aggregate(rows):
    bucket = defaultdict(list)
    for r in rows:
        if r["passed"]:
            bucket[r["key"]].append(r)

    summary = []
    for key, items in bucket.items():
        scores = [x["robust_score"] for x in items]
        summary.append({
            "key": key,
            "params": items[0]["params"],
            "appearances": len(items),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "sources": sorted({x["source"] for x in items}),
        })

    summary.sort(
        key=lambda x: (
            x["appearances"],
            x["score_mean"],
            -x["score_std"],
        ),
        reverse=True,
    )
    return summary

def main():
    rows = load_all_results()
    summary = aggregate(rows)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[POST] Total raw runs: {len(rows)}")
    print(f"[POST] Survivors (unique param sets): {len(summary)}")
    print(f"[POST] Saved -> {OUT_FILE}")

    print("\nTop 5 candidates:")
    for s in summary[:5]:
        print(
            f"- appear={s['appearances']} "
            f"mean={s['score_mean']:.2f} "
            f"std={s['score_std']:.2f}"
        )

if __name__ == "__main__":
    main()
