#!/usr/bin/env python3
# analysis/validate_pipeline_consistency.py
from __future__ import annotations

import json
import argparse
import numpy as np
import pandas as pd
from typing import Any, Dict


EPS = 1e-9


def safe_json(x: Any) -> Dict[str, Any]:
    if not isinstance(x, str):
        return {}
    try:
        return json.loads(x)
    except Exception:
        return {}


def sortino(r: np.ndarray) -> float:
    if r.size == 0:
        return np.nan
    downside = r[r < 0]
    if downside.size == 0:
        return np.inf
    dstd = np.std(downside, ddof=1) if downside.size > 1 else 0.0
    if dstd <= 0:
        return np.inf if r.mean() > 0 else -np.inf
    return r.mean() / dstd


def max_dd(eq: np.ndarray) -> float:
    if eq.size == 0:
        return np.nan
    roll = np.maximum.accumulate(eq)
    dd = (eq - roll) / np.where(roll == 0, np.nan, roll)
    return np.nanmin(dd)


def main():
    ap = argparse.ArgumentParser("Validate pipeline CSV vs tensors")
    ap.add_argument("--csv", default="results/pipeline_trades.csv")
    ap.add_argument("--tensors", default="results/pipeline_tensors")
    ap.add_argument("--group-index", type=int, required=True)
    args = ap.parse_args()

    # =========================
    # Load tensors
    # =========================
    R = np.load(f"{args.tensors}/R.npy")
    M = np.load(f"{args.tensors}/M.npy")
    with open(f"{args.tensors}/groups.json", "r", encoding="utf-8") as f:
        groups = json.load(f)

    g = groups[args.group_index]
    mask = M[args.group_index]
    r_tensor = R[args.group_index][mask]

    # =========================
    # Load CSV
    # =========================
    df = pd.read_csv(args.csv)
    df = df[df["type"].astype(str).str.upper() == "EXIT"].copy()

    df["_meta"] = df.get("meta_json", "").apply(safe_json)
    df["_ctx"] = df["_meta"].apply(lambda m: m.get("context", {}) if isinstance(m, dict) else {})
    df["_params"] = df["_meta"].apply(lambda m: m.get("params", {}) if isinstance(m, dict) else {})

    df_match = df[
        (df["_ctx"].apply(lambda c: c.get("symbol")) == g["symbol"]) &
        (df["_ctx"].apply(lambda c: c.get("regime")) == g["regime"]) &
        (df["_ctx"].apply(lambda c: str(c.get("seed"))) == str(g["seed"])) &
        (df["_ctx"].apply(lambda c: c.get("window")) == g["window"])
    ].copy()

    # Filtrar por params_key
    def params_key(p):
        return json.dumps(p or {}, sort_keys=True, separators=(",", ":"))

    df_match["_pk"] = df_match["_params"].apply(params_key)
    df_match = df_match[df_match["_pk"] == g["params_key"]]

    df_match["pnl_r"] = pd.to_numeric(df_match["pnl_r"], errors="coerce")
    df_match["equity_after"] = pd.to_numeric(df_match["equity_after"], errors="coerce")

    r_csv = df_match["pnl_r"].to_numpy(dtype=float)
    eq_csv = df_match["equity_after"].to_numpy(dtype=float)

    # =========================
    # Metrics
    # =========================
    out = {
        "group_index": args.group_index,
        "symbol": g["symbol"],
        "regime": g["regime"],
        "seed": g["seed"],
        "window": g["window"],
        "trades_csv": len(r_csv),
        "trades_tensor": len(r_tensor),
        "mean_csv": float(np.mean(r_csv)),
        "mean_tensor": float(np.mean(r_tensor)),
        "sortino_csv": float(sortino(r_csv)),
        "sortino_tensor": float(sortino(r_tensor)),
        "max_dd_csv": float(max_dd(eq_csv)),
    }

    print(json.dumps(out, indent=2))

    # =========================
    # Assertions
    # =========================
    assert len(r_csv) == len(r_tensor), "❌ trade count mismatch"
    assert abs(out["mean_csv"] - out["mean_tensor"]) < EPS, "❌ mean mismatch"
    assert abs(out["sortino_csv"] - out["sortino_tensor"]) < 1e-6, "❌ sortino mismatch"

    print("\n✅ VALIDATION OK: CSV == TENSORS")


if __name__ == "__main__":
    main()
