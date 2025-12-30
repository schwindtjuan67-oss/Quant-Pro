#!/usr/bin/env python3
# analysis/eval_tensors.py
from __future__ import annotations

import os
import json
import argparse
import numpy as np
from typing import Dict, Any, List

TENSORS_DIR = os.getenv("PIPELINE_TENSORS_DIR", "results/pipeline_tensors")
OUT_JSON = os.getenv("PIPELINE_CANDIDATES_JSON", "results/pipeline_candidates_for_shadow.json")
OUT_CSV  = os.getenv("PIPELINE_CANDIDATES_CSV", "results/pipeline_candidates_for_shadow.csv")

# Thresholds (alineados a analyze_pipeline.py)
MIN_TRADES = int(os.getenv("PIPELINE_MIN_TRADES", "300"))
TH_EXPECTANCY = float(os.getenv("PIPELINE_TH_EXPECTANCY", "0.05"))
TH_SORTINO    = float(os.getenv("PIPELINE_TH_SORTINO", "1.50"))
TH_PF         = float(os.getenv("PIPELINE_TH_PF", "1.30"))
TH_DD         = float(os.getenv("PIPELINE_TH_DD", "-0.20"))
TH_WINRATE    = float(os.getenv("PIPELINE_TH_WINRATE", "0.40"))
TH_WORST5     = float(os.getenv("PIPELINE_TH_WORST5", "-1.50"))
TOP_N_JSON    = int(os.getenv("PIPELINE_TOP_N_JSON", "50"))
EPS = 1e-12

def sortino_from_R(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    # R,M: (G,T)
    out = np.zeros(R.shape[0], dtype=np.float32)
    for i in range(R.shape[0]):
        r = R[i][M[i]]
        if r.size == 0:
            out[i] = np.nan
            continue
        down = r[r < 0]
        if down.size == 0:
            out[i] = np.inf
            continue
        dstd = np.std(down, ddof=1) if down.size > 1 else 0.0
        if dstd <= 0:
            out[i] = np.inf if r.mean() > 0 else -np.inf
        else:
            out[i] = r.mean() / dstd
    return out

def max_dd_from_R(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    out = np.zeros(R.shape[0], dtype=np.float32)
    for i in range(R.shape[0]):
        r = R[i][M[i]]
        if r.size == 0:
            out[i] = np.nan
            continue
        eq = np.cumsum(r)
        roll = np.maximum.accumulate(eq)
        dd = (eq - roll) / np.where(roll == 0, np.nan, roll)
        dd = dd[np.isfinite(dd)]
        out[i] = dd.min() if dd.size else np.nan
    return out

def profit_factor_from_R(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    out = np.zeros(R.shape[0], dtype=np.float32)
    for i in range(R.shape[0]):
        r = R[i][M[i]]
        pos = r[r > 0].sum()
        neg = r[r < 0].sum()
        if abs(neg) < EPS:
            out[i] = np.inf if pos > 0 else np.nan
        else:
            out[i] = pos / abs(neg)
    return out

def worst5_from_R(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    out = np.zeros(R.shape[0], dtype=np.float32)
    for i in range(R.shape[0]):
        r = R[i][M[i]]
        out[i] = np.quantile(r, 0.05) if r.size else np.nan
    return out

def main():
    ap = argparse.ArgumentParser("Evaluate tensors")
    ap.add_argument("--dir", default=TENSORS_DIR)
    args = ap.parse_args()

    R = np.load(os.path.join(args.dir, "R.npy"))
    M = np.load(os.path.join(args.dir, "M.npy"))
    with open(os.path.join(args.dir, "groups.json"), "r", encoding="utf-8") as f:
        groups = json.load(f)

    # Métricas
    N = M.sum(axis=1).astype(np.int32)
    expectancy = np.where(N > 0, (R * M).sum(axis=1) / np.maximum(N, 1), np.nan)
    winrate = np.where(N > 0, ((R > 0) & M).sum(axis=1) / np.maximum(N, 1), np.nan)
    srt = sortino_from_R(R, M)
    pf = profit_factor_from_R(R, M)
    dd = max_dd_from_R(R, M)
    w5 = worst5_from_R(R, M)

    rows = []
    payload = []

    for i, g in enumerate(groups):
        if N[i] < MIN_TRADES:
            continue

        passed = (
            expectancy[i] > TH_EXPECTANCY and
            srt[i] >= TH_SORTINO and
            pf[i] >= TH_PF and
            np.isfinite(dd[i]) and dd[i] > TH_DD and
            winrate[i] >= TH_WINRATE and
            w5[i] > TH_WORST5
        )

        rows.append({
            "symbol": g["symbol"],
            "regime": g["regime"],
            "seed": g["seed"],
            "window": g["window"],
            "params_key": g["params_key"],
            "trades": int(N[i]),
            "expectancy_r": float(expectancy[i]),
            "sortino": float(srt[i]),
            "profit_factor": float(pf[i]),
            "max_dd": float(dd[i]),
            "winrate": float(winrate[i]),
            "worst_5pct_r": float(w5[i]),
            "PASS_SHADOW": bool(passed),
        })

        if passed:
            payload.append({
                "symbol": g["symbol"],                 # Opción A ✅
                "params": g.get("params", {}),
                "metrics": {
                    "trades": int(N[i]),
                    "expectancy_r": float(expectancy[i]),
                    "sortino": float(srt[i]),
                    "profit_factor": float(pf[i]),
                    "max_dd": float(dd[i]),
                    "winrate": float(winrate[i]),
                    "worst_5pct_r": float(w5[i]),
                }
            })

    # Guardar outputs
    import pandas as pd
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    pd.DataFrame(rows).sort_values(
        by=["PASS_SHADOW", "expectancy_r", "sortino", "profit_factor"],
        ascending=[False, False, False, False],
    ).to_csv(OUT_CSV, index=False)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload[:TOP_N_JSON], f, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} rows -> {OUT_CSV}")
    print(f"Saved {len(payload[:TOP_N_JSON])} PASS_SHADOW -> {OUT_JSON}")

if __name__ == "__main__":
    main()

