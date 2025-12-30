# analysis/autotuning_fase8_1.py
from __future__ import annotations

import os
import json
import glob
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Helpers
# =========================================================
def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_json_loads(s: Any) -> Optional[dict]:
    if s is None:
        return None
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        try:
            return dict(s)  # last resort
        except Exception:
            return None
    if s.strip() == "":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _group_meta(meta: Optional[dict]) -> Optional[dict]:
    """
    IMPORTANT (industrial):
    Para evitar que el grouping explote por claves dinámicas,
    el tuner agrupa SOLO por:
      - meta["params"] (dict)
      - meta["regime"] (opcional)

    Si meta_json trae extras, los ignoramos para el group_key.
    """
    if not meta or not isinstance(meta, dict):
        return None

    params = meta.get("params")
    regime = meta.get("regime")

    out: Dict[str, Any] = {}
    if isinstance(params, dict) and params:
        out["params"] = params
    if regime is not None and regime != "":
        out["regime"] = regime

    return out or None


def _canonical_meta_key(meta: Optional[dict]) -> str:
    """
    Canonicaliza meta para agrupar de forma estable:
    - ordena keys
    - elimina separadores innecesarios
    """
    if not meta:
        return "NO_META"
    try:
        return json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return "NO_META"


def _max_drawdown(series: np.ndarray) -> float:
    """
    Max drawdown sobre curva acumulada (ej: cumulative pnl_r).
    Devuelve valor positivo (ej 2.3 = 2.3R de drawdown).
    """
    if series.size == 0:
        return 0.0
    peak = np.maximum.accumulate(series)
    dd = peak - series
    return float(np.max(dd)) if dd.size else 0.0


def _sharpe(x: np.ndarray) -> float:
    """
    Sharpe simple sobre retornos (aquí pnl_r por trade).
    No anualiza; sirve para ranking.
    """
    if x.size < 2:
        return np.nan
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return np.nan
    return mu / sd


def _t_stat(x: np.ndarray) -> float:
    """
    t-stat de la media: mean / (std/sqrt(n))
    """
    if x.size < 2:
        return np.nan
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return np.nan
    return mu / (sd / np.sqrt(x.size))


# =========================================================
# IO: load trades
# =========================================================
def _find_trade_files(logs_dir: str, symbol: Optional[str] = None) -> List[str]:
    sym_pat = f"{symbol}_" if symbol else ""
    pat_v4 = os.path.join(logs_dir, f"{sym_pat}*shadow_trades_v4.csv")
    pat_v3 = os.path.join(logs_dir, f"{sym_pat}*shadow_trades_v3.csv")

    v4 = sorted(glob.glob(pat_v4))
    v3 = sorted(glob.glob(pat_v3))

    # prioridad natural: v4 primero, luego v3
    return v4 + v3


def _load_trades_any(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df is None or df.empty:
            return pd.DataFrame()
        df["__source_file__"] = os.path.basename(path)
        return df
    except Exception:
        return pd.DataFrame()


def load_all_trades(logs_dir: str, symbol: Optional[str] = None) -> pd.DataFrame:
    files = _find_trade_files(logs_dir, symbol=symbol)
    if not files:
        return pd.DataFrame()

    frames = []
    for fp in files:
        df = _load_trades_any(fp)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    if "type" not in out.columns:
        out["type"] = ""

    if "timestamp_ms" not in out.columns:
        out["timestamp_ms"] = np.nan

    if "pnl_r" not in out.columns:
        out["pnl_r"] = np.nan

    if "meta_json" not in out.columns:
        out["meta_json"] = ""

    return out


# =========================================================
# Core: compute metrics per meta group
# =========================================================
@dataclass
class GroupResult:
    group_key: str
    symbol: str
    n_trades: int
    winrate: float
    avg_pnl_r: float
    med_pnl_r: float
    sharpe_pnl_r: float
    tstat_pnl_r: float
    max_dd_r: float
    cum_pnl_r: float
    score: float
    example_meta: dict     # <- meta "operable" (params/regime)
    source_files: List[str]


def compute_groups(df: pd.DataFrame, *, min_trades: int = 10) -> List[GroupResult]:
    if df is None or df.empty:
        return []

    d = df[df["type"].astype(str).str.upper() == "EXIT"].copy()
    if d.empty:
        return []

    # symbol
    if "symbol" not in d.columns:
        d["symbol"] = ""
        try:
            d["symbol"] = d["__source_file__"].astype(str).str.split("_").str[0]
        except Exception:
            pass

    d["symbol"] = d["symbol"].astype(str).str.upper().str.strip()

    # pnl_r numérico
    d["pnl_r"] = pd.to_numeric(d["pnl_r"], errors="coerce")

    # meta parse + group_key stable (params/regime only)
    meta_objs: List[Optional[dict]] = []
    group_metas: List[Optional[dict]] = []
    group_keys: List[str] = []

    for s in d["meta_json"].tolist():
        meta = _safe_json_loads(s)
        meta_objs.append(meta)

        gm = _group_meta(meta)
        group_metas.append(gm)
        group_keys.append(_canonical_meta_key(gm))

    d["__meta_obj__"] = meta_objs
    d["__group_meta__"] = group_metas
    d["__group_key__"] = group_keys

    # ordenar por timestamp si existe
    if "timestamp_ms" in d.columns:
        d["timestamp_ms"] = pd.to_numeric(d["timestamp_ms"], errors="coerce")
        d = d.sort_values("timestamp_ms", kind="mergesort")

    out: List[GroupResult] = []

    for (sym, gkey), g in d.groupby(["symbol", "__group_key__"], dropna=False):
        g = g.dropna(subset=["pnl_r"]).copy()
        n = int(len(g))
        if n < int(min_trades):
            continue

        x = g["pnl_r"].to_numpy(dtype=float)
        wins = float(np.mean(x > 0.0)) if x.size else 0.0

        cum = float(np.sum(x))
        curve = np.cumsum(x)
        mdd = _max_drawdown(curve)

        sh = _sharpe(x)
        ts = _t_stat(x)

        avg = float(np.mean(x))
        med = float(np.median(x))

        # score industrial default (tuneable)
        score = float(avg * 0.6 + (ts if not np.isnan(ts) else 0.0) * 0.4 - mdd * 0.15)

        meta_example = {}
        try:
            # usamos group_meta (params/regime), no el meta completo
            last_gm = g["__group_meta__"].iloc[-1]
            if isinstance(last_gm, dict):
                meta_example = last_gm
        except Exception:
            meta_example = {}

        files = sorted(list(set(g.get("__source_file__", pd.Series([], dtype=str)).astype(str).tolist())))

        out.append(
            GroupResult(
                group_key=str(gkey),
                symbol=str(sym),
                n_trades=n,
                winrate=float(wins),
                avg_pnl_r=avg,
                med_pnl_r=med,
                sharpe_pnl_r=float(sh) if not np.isnan(sh) else np.nan,
                tstat_pnl_r=float(ts) if not np.isnan(ts) else np.nan,
                max_dd_r=float(mdd),
                cum_pnl_r=float(cum),
                score=score,
                example_meta=meta_example,
                source_files=files,
            )
        )

    out.sort(key=lambda r: (r.score, r.tstat_pnl_r if not np.isnan(r.tstat_pnl_r) else -1e9), reverse=True)
    return out


# =========================================================
# Export
# =========================================================
def export_report(
    results: List[GroupResult],
    *,
    logs_dir: str,
    top_k: int = 20,
    report_csv: str = "autotune_report.csv",
    top_json: str = "top_k.json",
) -> Tuple[str, str]:
    os.makedirs(logs_dir, exist_ok=True)

    # CSV
    rows = []
    for r in results:
        rows.append(
            {
                "symbol": r.symbol,
                "n_trades": r.n_trades,
                "winrate": r.winrate,
                "avg_pnl_r": r.avg_pnl_r,
                "med_pnl_r": r.med_pnl_r,
                "sharpe_pnl_r": r.sharpe_pnl_r,
                "tstat_pnl_r": r.tstat_pnl_r,
                "max_dd_r": r.max_dd_r,
                "cum_pnl_r": r.cum_pnl_r,
                "score": r.score,
                "source_files": json.dumps(r.source_files, ensure_ascii=False),
                "group_key": r.group_key,
                "meta_canon": _canonical_meta_key(r.example_meta),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(logs_dir, report_csv)
    df.to_csv(csv_path, index=False)

    # TOP-K JSON “operable” para FASE 10 selector
    items = []
    for r in results[: int(top_k)]:
        meta = r.example_meta if isinstance(r.example_meta, dict) else {}
        params = meta.get("params", {}) if isinstance(meta.get("params", None), dict) else {}
        regime = meta.get("regime", None)

        items.append(
            {
                "symbol": r.symbol,
                "score": r.score,
                "n_trades": r.n_trades,
                "winrate": r.winrate,
                "avg_pnl_r": r.avg_pnl_r,
                "med_pnl_r": r.med_pnl_r,
                "sharpe_pnl_r": r.sharpe_pnl_r,
                "tstat_pnl_r": r.tstat_pnl_r,
                "max_dd_r": r.max_dd_r,
                "cum_pnl_r": r.cum_pnl_r,
                "group_key": r.group_key,   # canonical del group_meta (params/regime)
                "params": params,           # <- directo para aplicar al Hybrid (FASE 10.1)
                "regime": regime,           # <- opcional
                "meta": meta,               # <- meta operable (igual que group_meta)
                "source_files": r.source_files,
            }
        )

    json_path = os.path.join(logs_dir, top_json)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "top_k": int(top_k),
                "items": items,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return csv_path, json_path


# =========================================================
# CLI
# =========================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", type=str, default="logs")
    p.add_argument("--symbol", type=str, default=None, help="Ej: SOLUSDT (opcional)")
    p.add_argument("--min_trades", type=int, default=12)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    df = load_all_trades(args.logs_dir, symbol=args.symbol)
    if df is None or df.empty:
        print("[AUTOTUNE] No hay trades en logs (v4/v3).")
        # igual generamos outputs vacíos para no romper workers/selectores
        export_report([], logs_dir=args.logs_dir, top_k=args.top_k)
        return

    res = compute_groups(df, min_trades=args.min_trades)
    if not res:
        print("[AUTOTUNE] No hay grupos con suficientes trades para rankear.")
        # outputs vacíos (sin fricción)
        export_report([], logs_dir=args.logs_dir, top_k=args.top_k)
        return

    csv_path, json_path = export_report(res, logs_dir=args.logs_dir, top_k=args.top_k)
    print(f"[AUTOTUNE] OK -> report: {csv_path}")
    print(f"[AUTOTUNE] OK -> top_k:  {json_path}")
    print(f"[AUTOTUNE] groups ranked: {len(res)}")


if __name__ == "__main__":
    main()
