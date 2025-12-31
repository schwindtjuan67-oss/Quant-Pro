#!/usr/bin/env python3
import argparse
import json
import glob
import os
import hashlib
from collections import defaultdict

import numpy as np


def params_key(params: dict) -> str:
    """Hash estable del dict de params"""
    s = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def safe_load_json(fp: str):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_all_results(results_dir: str):
    rows = []
    for fp in glob.glob(os.path.join(results_dir, "*.json")):
        data = safe_load_json(fp)
        if not isinstance(data, list):
            # archivo roto/corrupto => lo salteamos sin tumbar pipeline
            continue

        for r in data:
            if not isinstance(r, dict):
                continue
            params = r.get("params")
            if not isinstance(params, dict):
                continue

            agg = r.get("agg") or {}
            rows.append({
                "params": params,
                "key": params_key(params),
                "robust_score": float(agg.get("robust_score", -1e9)),
                "passed": bool(r.get("passed", False)),
                "fail_reason": str(r.get("fail_reason", "") or ""),
                # EXTRA: si existe, lo agregamos (sirve para gate de trades>=300)
                "trades": float(agg.get("trades", -1)),
                "source": os.path.basename(fp),
            })
    return rows


def aggregate(rows):
    bucket = defaultdict(list)
    fails = defaultdict(int)

    for r in rows:
        if r["passed"]:
            bucket[r["key"]].append(r)
        else:
            fails[r.get("fail_reason", "unknown")] += 1

    summary = []
    for key, items in bucket.items():
        scores = [x["robust_score"] for x in items]
        trades = [x["trades"] for x in items if x.get("trades", -1) >= 0]

        summary.append({
            "key": key,
            "params": items[0]["params"],
            "appearances": len(items),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "trades_mean": float(np.mean(trades)) if trades else None,
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

    return {
        "summary": summary,
        "fails": dict(sorted(fails.items(), key=lambda kv: kv[1], reverse=True)),
        "raw_rows": len(rows),
        "survivors": len(summary),
    }


def main():
    ap = argparse.ArgumentParser("analysis_post_robust")
    ap.add_argument("--results-dir", default="results/robust")
    ap.add_argument("--out", default="results/post_analysis_summary.json")
    args = ap.parse_args()

    rows = load_all_results(args.results_dir)
    payload = aggregate(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[POST] Total raw runs: {payload['raw_rows']}")
    print(f"[POST] Survivors (unique param sets): {payload['survivors']}")
    print(f"[POST] Saved -> {args.out}")

    top = payload["summary"][:5]
    if top:
        print("\nTop 5 candidates:")
        for s in top:
            print(
                f"- appear={s['appearances']} "
                f"mean={s['score_mean']:.2f} "
                f"std={s['score_std']:.2f} "
                f"trades_mean={s['trades_mean']}"
            )


if __name__ == "__main__":
    main()

