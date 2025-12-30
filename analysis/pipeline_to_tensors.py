#!/usr/bin/env python3
# analysis/pipeline_to_tensors.py
from __future__ import annotations

import os
import json
import argparse
import glob
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

CSV_PATH = os.getenv("PIPELINE_TRADES_CSV", "results/pipeline_trades.csv")
PARQUET_DIR = os.getenv("PIPELINE_PARQUET_DIR", "results/pipeline_trades_parquet")
OUT_DIR = os.getenv("PIPELINE_TENSORS_DIR", "results/pipeline_tensors")

# Optional behavior toggles
ASSUME_TIME_SORTED = os.getenv("PIPELINE_TENSORS_ASSUME_SORTED", "1").lower() in ("1", "true", "yes")

# ======================
# helpers
# ======================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _safe_json_loads(s: Any) -> Dict[str, Any]:
    if not isinstance(s, str):
        return {}
    s = s.strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _stable_params_key(params: dict) -> str:
    try:
        return json.dumps(
            params or {},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:
        return str(params)

def _get_ctx(meta: Dict[str, Any], k: str) -> Optional[Any]:
    ctx = meta.get("context")
    if isinstance(ctx, dict):
        return ctx.get(k)
    return None

def _get_params(meta: Dict[str, Any]) -> Dict[str, Any]:
    p = meta.get("params")
    return p if isinstance(p, dict) else {}

def _iter_parquet_files(parquet_dir: str) -> List[str]:
    # Works with partitioned layout: parquet_dir/date_local=YYYY-MM-DD/part-*.parquet
    pat = os.path.join(parquet_dir, "**", "*.parquet")
    files = glob.glob(pat, recursive=True)

    # Sort by partition (date_local=...) so we process chronologically as much as possible
    # This is "good enough" if data was appended in time order (your case).
    def key_fn(path: str) -> str:
        # brings "date_local=YYYY-MM-DD" to the front if present
        parts = path.replace("\\", "/").split("/")
        dl = ""
        for p in parts:
            if p.startswith("date_local="):
                dl = p.split("=", 1)[1]
                break
        return f"{dl}|{os.path.basename(path)}"

    return sorted(files, key=key_fn)

# ======================
# streaming accumulator
# ======================
class GroupAccum:
    __slots__ = ("r_list", "eq_list", "pnl_list", "n", "params")

    def __init__(self):
        self.r_list: List[float] = []
        self.eq_list: Optional[List[float]] = None
        self.pnl_list: Optional[List[float]] = None
        self.n: int = 0
        self.params: Dict[str, Any] = {}

    def ensure_eq(self):
        if self.eq_list is None:
            self.eq_list = []

    def ensure_pnl(self):
        if self.pnl_list is None:
            self.pnl_list = []

# ======================
# main
# ======================
def main() -> None:
    ap = argparse.ArgumentParser("pipeline trades -> tensors (Parquet-first, streaming)")
    ap.add_argument("--out", default=OUT_DIR, help="output dir for tensors")
    ap.add_argument("--parquet-dir", default=PARQUET_DIR, help="parquet dataset dir (partitioned by date_local=...)")
    ap.add_argument("--csv", default=CSV_PATH, help="fallback CSV path (if parquet not found)")
    args = ap.parse_args()

    out_dir = args.out
    _ensure_dir(out_dir)

    parquet_files = _iter_parquet_files(args.parquet_dir) if os.path.isdir(args.parquet_dir) else []
    use_parquet = len(parquet_files) > 0

    if not use_parquet:
        if not os.path.exists(args.csv):
            raise SystemExit(
                f"[pipeline_to_tensors] No parquet found in {args.parquet_dir} and missing CSV: {args.csv}"
            )
        print(f"[pipeline_to_tensors] Parquet not found. Falling back to CSV: {args.csv}")

    # Accumulators keyed by (symbol, regime, seed, window, params_key)
    acc: Dict[Tuple[Any, Any, Any, Any, str], GroupAccum] = {}

    # Detect whether we can emit EQ/PNL tensors (only if we see these columns at least once)
    saw_eq = False
    saw_pnl = False

    def consume_df(df: pd.DataFrame) -> None:
        nonlocal saw_eq, saw_pnl

        if df is None or df.empty:
            return

        # keep only EXIT rows
        if "type" not in df.columns:
            return
        t = df["type"].astype(str).str.upper()
        df = df.loc[t == "EXIT"].copy()
        if df.empty:
            return

        # numeric coercions
        for c in ("pnl_r", "equity_after", "pnl_net_est", "pnl", "timestamp_ms"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "meta_json" not in df.columns:
            # If no meta_json, we cannot group properly
            return

        # parse meta_json -> context + params
        metas = df["meta_json"].apply(_safe_json_loads)

        df["_symbol"] = metas.apply(lambda m: _get_ctx(m, "symbol"))
        df["_regime"] = metas.apply(lambda m: _get_ctx(m, "regime"))
        df["_seed"] = metas.apply(lambda m: _get_ctx(m, "seed"))
        df["_window"] = metas.apply(lambda m: _get_ctx(m, "window"))
        df["_params"] = metas.apply(_get_params)
        df["_params_key"] = df["_params"].apply(_stable_params_key)

        # optional ordering inside this chunk
        if not ASSUME_TIME_SORTED and "timestamp_ms" in df.columns:
            df = df.sort_values("timestamp_ms")

        have_eq = "equity_after" in df.columns
        have_pnl = ("pnl_net_est" in df.columns) or ("pnl" in df.columns)

        if have_eq:
            saw_eq = True
        if have_pnl:
            saw_pnl = True

        group_cols = ["_symbol", "_regime", "_seed", "_window", "_params_key"]

        # stream groups for this file
        for key, g in df.groupby(group_cols, dropna=False, sort=False):
            # Extract r (R-multiple)
            if "pnl_r" not in g.columns:
                continue
            r = g["pnl_r"].to_numpy(dtype=np.float32)
            r = r[np.isfinite(r)]
            if r.size == 0:
                continue

            sym, reg, seed, window, pkey = key
            k = (sym, reg, seed, window, str(pkey))

            a = acc.get(k)
            if a is None:
                a = GroupAccum()
                acc[k] = a

            # params (store first non-empty)
            if (not a.params) and len(g["_params"].values) > 0:
                p0 = g["_params"].iloc[0]
                if isinstance(p0, dict) and p0:
                    a.params = p0

            # Append r
            a.r_list.extend(r.tolist())
            a.n += int(r.size)

            # Optional EQ
            if have_eq:
                eq = g["equity_after"].to_numpy(dtype=np.float32)
                # keep aligned with r rows: use the raw (before dropping non-finite pnl_r)?
                # To keep tensors consistent, we align by original rows: filter eq by finite pnl_r too.
                mask = np.isfinite(g["pnl_r"].to_numpy(dtype=np.float32))
                eq = eq[mask]
                a.ensure_eq()
                a.eq_list.extend(eq.tolist())

            # Optional PNL
            if have_pnl:
                if "pnl_net_est" in g.columns:
                    pnl = g["pnl_net_est"].to_numpy(dtype=np.float32)
                else:
                    pnl = g["pnl"].to_numpy(dtype=np.float32)
                mask = np.isfinite(g["pnl_r"].to_numpy(dtype=np.float32))
                pnl = pnl[mask]
                a.ensure_pnl()
                a.pnl_list.extend(pnl.tolist())

    if use_parquet:
        # Read each parquet file individually (streaming, no huge concat)
        # Limit columns to what we need to reduce IO
        needed_cols = ["type", "timestamp_ms", "pnl_r", "equity_after", "pnl_net_est", "pnl", "meta_json"]
        for fp in parquet_files:
            try:
                df = pd.read_parquet(fp, columns=needed_cols)
            except Exception:
                # Some engines error if columns missing; read all then
                df = pd.read_parquet(fp)
            consume_df(df)
    else:
        # CSV fallback (legacy)
        df = pd.read_csv(
            args.csv,
            engine="python",
            quotechar='"',
            doublequote=True,
            skipinitialspace=False,
        )
        consume_df(df)

    if not acc:
        print("[pipeline_to_tensors] No valid EXIT groups created. Nothing to tensorize.")
        return

    # Build final group list
    keys = list(acc.keys())
    # optional: enforce deterministic ordering (useful for reproducibility)
    keys.sort(key=lambda k: (str(k[0]), str(k[1]), str(k[2]), str(k[3]), str(k[4])))

    G = len(keys)
    T_max = max(len(acc[k].r_list) for k in keys)

    R = np.zeros((G, T_max), dtype=np.float32)
    M = np.zeros((G, T_max), dtype=np.bool_)

    EQ = None
    PNL = None
    if saw_eq:
        EQ = np.zeros((G, T_max), dtype=np.float32)
    if saw_pnl:
        PNL = np.zeros((G, T_max), dtype=np.float32)

    groups_meta: List[Dict[str, Any]] = []

    for i, k in enumerate(keys):
        a = acc[k]
        n = len(a.r_list)
        if n <= 0:
            continue

        R[i, :n] = np.asarray(a.r_list, dtype=np.float32)
        M[i, :n] = True

        if EQ is not None and a.eq_list is not None:
            eqn = min(len(a.eq_list), n)
            EQ[i, :eqn] = np.asarray(a.eq_list[:eqn], dtype=np.float32)

        if PNL is not None and a.pnl_list is not None:
            pnln = min(len(a.pnl_list), n)
            PNL[i, :pnln] = np.asarray(a.pnl_list[:pnln], dtype=np.float32)

        symbol, regime, seed, window, params_key = k
        groups_meta.append(
            {
                "group_index": i,
                "symbol": symbol,
                "regime": regime,
                "seed": seed,
                "window": window,
                "params_key": params_key,
                "params": a.params or {},
                "n_trades": int(n),
            }
        )

    # Save outputs (UNCHANGED names)
    np.save(os.path.join(out_dir, "R.npy"), R)
    np.save(os.path.join(out_dir, "M.npy"), M)
    if EQ is not None:
        np.save(os.path.join(out_dir, "EQ.npy"), EQ)
    if PNL is not None:
        np.save(os.path.join(out_dir, "PNL.npy"), PNL)

    with open(os.path.join(out_dir, "groups.json"), "w", encoding="utf-8") as f:
        json.dump(groups_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved tensors to {out_dir}")
    print(f"R shape = {R.shape}")
    print(f"M shape = {M.shape}")
    if EQ is not None:
        print(f"EQ shape = {EQ.shape}")
    if PNL is not None:
        print(f"PNL shape = {PNL.shape}")
    print(f"Groups  = {len(groups_meta)}")
    print(f"Source  = {'PARQUET' if use_parquet else 'CSV'}")
    if use_parquet:
        print(f"Parquet files read = {len(parquet_files)}")
    print(f"ASSUME_TIME_SORTED={ASSUME_TIME_SORTED}")

if __name__ == "__main__":
    main()








