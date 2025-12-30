#!/usr/bin/env python3
# analysis/gpu_evaluator_parquet.py
from __future__ import annotations

import os
import json
import glob
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# ============================
# Backend: CuPy (GPU)
# ============================
try:
    import cupy as cp
except Exception as e:
    raise SystemExit("[gpu_evaluator_parquet] CuPy not available. Install cupy-cudaXX.") from e

# ============================
# Paths
# ============================
PARQUET_DIR = os.getenv("PIPELINE_PARQUET_DIR", "results/pipeline_trades_parquet")
OUT_DIR = os.getenv("PIPELINE_EVAL_DIR", "results/pipeline_eval")

# ============================
# Thresholds
# ============================
MIN_TRADES = int(os.getenv("PIPELINE_MIN_TRADES", "300"))
MIN_R_OBS = int(os.getenv("PIPELINE_MIN_R_OBS", "200"))

TH_EXPECTANCY = float(os.getenv("PIPELINE_TH_EXPECTANCY", "0.05"))
TH_SORTINO = float(os.getenv("PIPELINE_TH_SORTINO", "1.50"))
TH_PF = float(os.getenv("PIPELINE_TH_PF", "1.30"))
TH_DD = float(os.getenv("PIPELINE_TH_DD", "-0.20"))  # (no DD si no hay EQ aquí)
TH_WINRATE = float(os.getenv("PIPELINE_TH_WINRATE", "0.40"))
TH_WORST5 = float(os.getenv("PIPELINE_TH_WORST5", "-1.50"))

TOP_N = int(os.getenv("PIPELINE_TOP_N_JSON", "50"))
EPS = 1e-12

# ============================
# Score v2 (ranking único)
# ============================
S_CAP_SORTINO = float(os.getenv("PIPELINE_SCORE_CAP_SORTINO", "5.0"))
S_WIN_CLIP_LO = float(os.getenv("PIPELINE_SCORE_WIN_CLIP_LO", "0.20"))
S_WIN_CLIP_HI = float(os.getenv("PIPELINE_SCORE_WIN_CLIP_HI", "0.80"))

S2_STD_PENALTY = float(os.getenv("PIPELINE_S2_STD_PENALTY", "1.0"))
S2_TAIL_PENALTY = float(os.getenv("PIPELINE_S2_TAIL_PENALTY", "1.0"))
S2_TRADES_SOFT = int(os.getenv("PIPELINE_S2_TRADES_SOFT", "800"))
S2_DD_WEIGHT = float(os.getenv("PIPELINE_S2_DD_WEIGHT", "1.0"))  # compat

# ============================
# Approx quantile (pass 1)
# ============================
Q_METHOD = os.getenv("PIPELINE_Q_METHOD", "sample").lower().strip()  # sample|hist|exact
Q_SAMPLE_K = int(os.getenv("PIPELINE_Q_SAMPLE_K", "512"))
Q_HIST_BINS = int(os.getenv("PIPELINE_Q_HIST_BINS", "1024"))
Q_HIST_CLIP = float(os.getenv("PIPELINE_Q_HIST_CLIP", "10.0"))

# ============================
# Top-K exact refinement
# ============================
TOPK_REFINE = int(os.getenv("PIPELINE_TOPK_REFINE", "512"))

# ============================
# Streams + pinned (fase 2)
# ============================
STREAMS = int(os.getenv("PIPELINE_GPU_STREAMS", "2"))  # 2 = double-buffer estándar
PINNED_POOL_BYTES = int(os.getenv("PIPELINE_PINNED_POOL_MB", "256")) * 1024 * 1024

# ============================
# IO helpers
# ============================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _safe_json(s: Any) -> Dict[str, Any]:
    if not isinstance(s, str):
        return {}
    s = s.strip()
    if not s:
        return {}
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}

def _ctx(meta: Dict[str, Any], k: str):
    c = meta.get("context")
    return c.get(k) if isinstance(c, dict) else None

def _params(meta: Dict[str, Any]):
    p = meta.get("params")
    return p if isinstance(p, dict) else {}

def _stable_params_key(p: Dict[str, Any]) -> str:
    try:
        return json.dumps(p or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(p)

# ============================
# GPU metric helpers
# ============================
def _worst5_approx_gpu(R: cp.ndarray, method: str, q: float = 0.05) -> cp.ndarray:
    n = int(R.size)
    if n <= 0:
        return cp.asarray(cp.nan, dtype=cp.float32)

    method = (method or "sample").lower().strip()
    if method == "exact":
        return cp.quantile(R, q).astype(cp.float32)

    if method == "hist":
        clip = float(Q_HIST_CLIP)
        bins = int(Q_HIST_BINS)
        x = cp.clip(R.astype(cp.float32), -clip, clip)
        scaled = (x + clip) * (bins / (2.0 * clip))
        b = cp.floor(scaled).astype(cp.int32)
        b = cp.clip(b, 0, bins - 1)

        hist = cp.zeros((bins,), dtype=cp.int32)
        cp.scatter_add(hist, (b,), 1)
        cdf = cp.cumsum(hist).astype(cp.float32)
        tot = cdf[-1]
        if float(tot.item()) <= 0:
            return cp.asarray(cp.nan, dtype=cp.float32)
        cdfn = cdf / tot
        idx = int(cp.argmax(cdfn >= q).item())
        val = (-clip + (float(idx) + 0.5) * (2.0 * clip / bins))
        return cp.asarray(val, dtype=cp.float32)

    # sample default
    k = int(Q_SAMPLE_K)
    if n <= k:
        return cp.quantile(R, q).astype(cp.float32)
    idx = cp.random.randint(0, n, size=(k,), dtype=cp.int32)
    samp = R[idx]
    return cp.quantile(samp, q).astype(cp.float32)

def _sortino_exact_gpu(R: cp.ndarray) -> cp.ndarray:
    n = int(R.size)
    if n <= 0:
        return cp.asarray(cp.nan, dtype=cp.float32)

    x = R.astype(cp.float32)
    mu = cp.mean(x)
    dn = x[x < 0]
    if int(dn.size) == 0:
        return cp.asarray(cp.inf, dtype=cp.float32)

    if int(dn.size) > 1:
        dstd = cp.std(dn, ddof=1)
    else:
        dstd = cp.asarray(0.0, dtype=cp.float32)

    dstd_v = float(dstd.item())
    mu_v = float(mu.item())
    if dstd_v <= 0.0:
        return cp.asarray(cp.inf if mu_v > 0 else -cp.inf, dtype=cp.float32)

    return (mu / dstd).astype(cp.float32)

def _profit_factor_gpu(R: cp.ndarray) -> cp.ndarray:
    x = R.astype(cp.float32)
    pos = cp.sum(cp.maximum(x, 0.0))
    neg = cp.sum(cp.maximum(-x, 0.0))
    pf = cp.where(neg > EPS, pos / neg, cp.inf)
    pf = cp.where(pos <= 0, cp.nan, pf)
    return pf.astype(cp.float32)

def _score_v2_gpu(expectancy, sortino, winrate, std_r, worst5, n_trades, max_dd: Optional[cp.ndarray] = None):
    w = cp.clip(winrate, S_WIN_CLIP_LO, S_WIN_CLIP_HI).astype(cp.float32)
    stability = (1.0 / (1.0 + cp.maximum(std_r, 0.0) * float(S2_STD_PENALTY))).astype(cp.float32)
    tail = (1.0 / (1.0 + cp.maximum(0.0, -worst5) * float(S2_TAIL_PENALTY))).astype(cp.float32)
    trades_fac = (n_trades / (n_trades + float(max(int(S2_TRADES_SOFT), 1)))).astype(cp.float32)

    if max_dd is None:
        dd_extra = cp.asarray(1.0, dtype=cp.float32)
    else:
        dd_fac = cp.maximum(1.0 + max_dd.astype(cp.float32), 0.0)
        dd_extra = (dd_fac ** float(max(S2_DD_WEIGHT, 0.0))).astype(cp.float32)

    return (expectancy * sortino * w * stability * tail * trades_fac * dd_extra).astype(cp.float32)

# ============================
# Streams + pinned helpers
# ============================
def _setup_pinned_pool():
    # Pool de pinned host memory (para H2D async real)
    pool = cp.cuda.PinnedMemoryPool()
    # si querés limitar, podés pre-alloc (no es hard cap, pero ayuda)
    try:
        _ = pool.malloc(min(PINNED_POOL_BYTES, 64 * 1024 * 1024))
    except Exception:
        pass
    cp.cuda.set_pinned_memory_allocator(pool.malloc)
    return pool

def _to_gpu_async(x_np: np.ndarray, stream: cp.cuda.Stream) -> cp.ndarray:
    """
    Copia host->device en el stream dado.
    Si x_np fue alocado con pinned allocator (via cupy pinned pool),
    la copia puede ser verdaderamente async.
    """
    # aseguramos contiguous float32
    x_np = np.ascontiguousarray(x_np, dtype=np.float32)
    nbytes = x_np.nbytes

    # device buffer
    d = cp.empty(x_np.shape, dtype=cp.float32)

    # memcpy async
    with stream:
        cp.cuda.runtime.memcpyAsync(
            d.data.ptr,
            x_np.ctypes.data,
            nbytes,
            cp.cuda.runtime.memcpyHostToDevice,
            stream.ptr,
        )
    return d

def _alloc_pinned_like(x_np: np.ndarray) -> np.ndarray:
    """
    Devuelve un numpy array backed by pinned memory (via cupy allocator),
    copiando el contenido desde x_np.
    """
    # cupy usa el allocator seteado para pinned cuando pedís empty_pinned
    pin = cp.empty_pinned(x_np.shape, dtype=np.float32)
    # copia CPU->pinned (memcpy normal, pero después H2D puede ser async)
    np.copyto(pin, np.asarray(x_np, dtype=np.float32), casting="unsafe")
    return pin

# ============================
# Main
# ============================
def main():
    ap = argparse.ArgumentParser("GPU evaluator from Parquet (groupby chunk + streams+pinned + TopK exact refine)")
    ap.add_argument("--parquet-dir", default=PARQUET_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--topk-refine", type=int, default=TOPK_REFINE)
    ap.add_argument("--q-method", choices=["sample", "hist", "exact"], default=Q_METHOD)
    ap.add_argument("--streams", type=int, default=STREAMS)
    args = ap.parse_args()

    _ensure_dir(args.out)

    files = glob.glob(os.path.join(args.parquet_dir, "**", "*.parquet"), recursive=True)
    if not files:
        raise SystemExit(f"No parquet files found under: {args.parquet_dir}")

    # ----------------------------
    # FASE 1: streaming + groupby por chunk (sin iterrows)
    # ----------------------------
    groups: Dict[Tuple, List[float]] = {}
    meta_map: Dict[Tuple, Dict[str, Any]] = {}

    # column pruning
    need_cols = ["type", "pnl_r", "meta_json", "regime"]
    for fp in sorted(files):
        df = pd.read_parquet(fp, columns=need_cols)
        if df.empty:
            continue

        # solo EXIT
        df = df[df["type"] == "EXIT"]
        if df.empty:
            continue

        # pnl_r numérico
        df["pnl_r"] = pd.to_numeric(df["pnl_r"], errors="coerce")
        arr = df["pnl_r"].to_numpy(dtype=np.float32, copy=False)
        m = np.isfinite(arr)
        if not m.any():
            continue
        df = df[m].copy()
        if df.empty:
            continue

        metas = df["meta_json"].apply(_safe_json)

        df["_symbol"] = metas.apply(lambda mm: _ctx(mm, "symbol"))
        df["_seed"] = metas.apply(lambda mm: _ctx(mm, "seed"))
        df["_window"] = metas.apply(lambda mm: _ctx(mm, "window"))
        df["_params"] = metas.apply(_params)
        df["_pkey"] = df["_params"].apply(_stable_params_key)

        group_cols = ["_symbol", "regime", "_seed", "_window", "_pkey"]

        # ✅ clave: groupby por chunk (vectorizado)
        for key, subdf in df.groupby(group_cols, sort=False, dropna=False):
            # key = (symbol, regime, seed, window, pkey)
            pnl_vals = subdf["pnl_r"].to_numpy(dtype=np.float32, copy=False)
            if pnl_vals.size == 0:
                continue

            # append list (CPU) -> luego se convierte a np.float32
            groups.setdefault(key, []).extend(pnl_vals.tolist())

            if key not in meta_map:
                sym, reg, seed, win, pkey = key
                # params representativo
                p0 = subdf["_params"].iloc[0] if "_params" in subdf.columns and len(subdf) else {}
                meta_map[key] = {
                    "symbol": sym,
                    "regime": reg,
                    "seed": seed,
                    "window": win,
                    "params_key": pkey,
                    "params": p0 if isinstance(p0, dict) else {},
                }

    if not groups:
        raise SystemExit("No groups found from Parquet (no EXIT with valid pnl_r).")

    # ----------------------------
    # FASE 2: streams + pinned memory (doble buffer)
    #   pipeline: CPU prep (pinned) -> async H2D -> compute (stream)
    # ----------------------------
    _setup_pinned_pool()
    n_streams = max(1, int(args.streams))
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]

    # pre-store arrays (CPU) para refine
    r_store: Dict[Tuple, np.ndarray] = {}
    eligible_keys: List[Tuple] = []

    for k, r_list in groups.items():
        r_np = np.asarray(r_list, dtype=np.float32)
        r_store[k] = r_np
        n = int(r_np.size)
        if n >= MIN_TRADES and n >= MIN_R_OBS:
            eligible_keys.append(k)

    if not eligible_keys:
        _write_json(os.path.join(args.out, "metrics.json"), [])
        _write_json(os.path.join(args.out, "candidates.json"), [])
        _write_json(os.path.join(args.out, "summary.json"), {
            "groups_total": 0,
            "pass_count": 0,
            "ranking": "score_v2",
            "source": "parquet",
            "gpu_only": True,
            "topk_refine": int(args.topk_refine),
            "refined_count": 0,
            "q_method_first_pass": args.q_method,
            "streams": int(n_streams),
        })
        print("[gpu_evaluator_parquet] No eligible groups after MIN_TRADES/MIN_R_OBS.")
        return

    results: List[Dict[str, Any]] = []

    # --- doble buffer simple (round-robin streams)
    # pre-stage pinned for first few
    pinned_bufs: Dict[int, Optional[np.ndarray]] = {i: None for i in range(n_streams)}
    keys_bufs: Dict[int, Optional[Tuple]] = {i: None for i in range(n_streams)}
    dev_bufs: Dict[int, Optional[cp.ndarray]] = {i: None for i in range(n_streams)}

    def stage(slot: int, k: Tuple):
        r_np = r_store[k]
        pin = _alloc_pinned_like(r_np)
        pinned_bufs[slot] = pin
        keys_bufs[slot] = k
        dev_bufs[slot] = _to_gpu_async(pin, streams[slot])

    # stage initial
    it = iter(eligible_keys)
    staged = 0
    for s in range(n_streams):
        try:
            k = next(it)
        except StopIteration:
            break
        stage(s, k)
        staged += 1

    processed = 0

    def compute_from_dev(R: cp.ndarray, n: int, q_method: str):
        # métricas GPU (pass 1 barato)
        expectancy = cp.mean(R)
        winrate = cp.mean(R > 0)
        std_r = cp.std(R)

        dn = R[R < 0]
        if int(dn.size) == 0:
            sortino = cp.asarray(cp.inf, dtype=cp.float32)
        else:
            dstd0 = cp.std(dn.astype(cp.float32))  # ddof=0 (barato)
            dstd0v = float(dstd0.item())
            if dstd0v <= 0.0:
                sortino = cp.asarray(cp.inf if float(expectancy.item()) > 0 else -cp.inf, dtype=cp.float32)
            else:
                sortino = (expectancy / dstd0).astype(cp.float32)

        sortino = cp.clip(sortino, 0.0, float(S_CAP_SORTINO)).astype(cp.float32)
        pf = _profit_factor_gpu(R)
        worst5 = _worst5_approx_gpu(R, method=q_method, q=0.05)

        ntr = cp.asarray(float(n), dtype=cp.float32)
        score_v2 = _score_v2_gpu(expectancy, sortino, winrate, std_r, worst5, ntr)

        passed = (
            float(expectancy.item()) > TH_EXPECTANCY and
            float(sortino.item()) >= TH_SORTINO and
            float(pf.item()) >= TH_PF and
            float(winrate.item()) >= TH_WINRATE and
            float(worst5.item()) > TH_WORST5
        )

        return expectancy, sortino, pf, winrate, worst5, std_r, score_v2, passed

    while staged > 0:
        # consumir slots en orden (round-robin)
        for s in range(n_streams):
            k = keys_bufs.get(s)
            Rdev = dev_bufs.get(s)
            if k is None or Rdev is None:
                continue

            # asegurar que la copia H2D terminó antes de computar en este slot
            streams[s].synchronize()

            r_np = r_store[k]
            n = int(r_np.size)

            # compute (en default stream está ok porque ya sincronizamos; si querés full overlap,
            # podés envolver compute en `with streams[s]:` también)
            with streams[s]:
                expectancy, sortino, pf, winrate, worst5, std_r, score_v2, passed = compute_from_dev(
                    Rdev, n=n, q_method=args.q_method
                )

            meta = meta_map[k]
            results.append({
                "group_key": k,  # interno
                "symbol": meta.get("symbol"),
                "regime": meta.get("regime"),
                "seed": meta.get("seed"),
                "window": meta.get("window"),
                "params_key": meta.get("params_key"),
                "params": meta.get("params", {}),
                "n_trades": int(n),
                "expectancy_r": float(expectancy.item()),
                "sortino": float(sortino.item()) if np.isfinite(float(sortino.item())) else None,
                "profit_factor": float(pf.item()) if np.isfinite(float(pf.item())) else None,
                "winrate": float(winrate.item()),
                "worst_5pct_r": float(worst5.item()) if np.isfinite(float(worst5.item())) else None,
                "std_r": float(std_r.item()),
                "PASS_SHADOW": bool(passed),
                "score_v2": float(score_v2.item()) if np.isfinite(float(score_v2.item())) else None,
                "_refined": False,
            })

            processed += 1

            # stage next key into same slot (mantiene pipeline)
            try:
                nk = next(it)
                stage(s, nk)
            except StopIteration:
                # liberar slot
                pinned_bufs[s] = None
                keys_bufs[s] = None
                dev_bufs[s] = None
                staged -= 1

    if not results:
        _write_json(os.path.join(args.out, "metrics.json"), [])
        _write_json(os.path.join(args.out, "candidates.json"), [])
        _write_json(os.path.join(args.out, "summary.json"), {
            "groups_total": 0,
            "pass_count": 0,
            "ranking": "score_v2",
            "source": "parquet",
            "gpu_only": True,
            "topk_refine": int(args.topk_refine),
            "refined_count": 0,
            "q_method_first_pass": args.q_method,
            "streams": int(n_streams),
        })
        print("[gpu_evaluator_parquet] No results produced.")
        return

    # ordenar por score_v2 (pass 1)
    results.sort(key=lambda x: (x["score_v2"] is not None, x["score_v2"]), reverse=True)

    # ============================
    # PASS 2: Top-K exact refinement (GPU)
    # ============================
    topk = min(max(int(args.topk_refine), 0), len(results))
    refined_count = 0

    for i in range(topk):
        row = results[i]
        k = row["group_key"]
        r_np = r_store.get(k)
        if r_np is None or r_np.size <= 0:
            continue

        # reuse stream 0 para refine (simple y estable)
        st = streams[0]
        pin = _alloc_pinned_like(r_np)
        R = _to_gpu_async(pin, st)
        st.synchronize()

        with st:
            worst5_ex = cp.quantile(R, 0.05).astype(cp.float32)
            sortino_ex = _sortino_exact_gpu(R)
            sortino_ex = cp.clip(sortino_ex, 0.0, float(S_CAP_SORTINO)).astype(cp.float32)
            pf_ex = _profit_factor_gpu(R)

            expectancy = cp.mean(R)
            winrate = cp.mean(R > 0)
            std_r = cp.std(R)

            ntr = cp.asarray(float(int(R.size)), dtype=cp.float32)
            score_v2_ex = _score_v2_gpu(expectancy, sortino_ex, winrate, std_r, worst5_ex, ntr)

            passed_ex = (
                float(expectancy.item()) > TH_EXPECTANCY and
                float(sortino_ex.item()) >= TH_SORTINO and
                float(pf_ex.item()) >= TH_PF and
                float(winrate.item()) >= TH_WINRATE and
                float(worst5_ex.item()) > TH_WORST5
            )

        row["expectancy_r"] = float(expectancy.item())
        row["sortino"] = float(sortino_ex.item()) if np.isfinite(float(sortino_ex.item())) else None
        row["profit_factor"] = float(pf_ex.item()) if np.isfinite(float(pf_ex.item())) else None
        row["winrate"] = float(winrate.item())
        row["worst_5pct_r"] = float(worst5_ex.item()) if np.isfinite(float(worst5_ex.item())) else None
        row["std_r"] = float(std_r.item())
        row["PASS_SHADOW"] = bool(passed_ex)
        row["score_v2"] = float(score_v2_ex.item()) if np.isfinite(float(score_v2_ex.item())) else None
        row["_refined"] = True
        refined_count += 1

    # ranking final
    results.sort(key=lambda x: (x["score_v2"] is not None, x["score_v2"]), reverse=True)

    # ============================
    # EXPORT (mismas salidas)
    # ============================
    metrics_out: List[Dict[str, Any]] = []
    for idx, r in enumerate(results):
        rr = dict(r)
        rr.pop("group_key", None)
        rr["group_index"] = idx
        metrics_out.append(rr)

    _write_json(os.path.join(args.out, "metrics.json"), metrics_out)

    candidates = [r for r in metrics_out if r.get("PASS_SHADOW")][: int(args.top)]
    _write_json(os.path.join(args.out, "candidates.json"), candidates)

    summary = {
        "groups_total": int(len(metrics_out)),
        "pass_count": int(len(candidates)),
        "ranking": "score_v2",
        "source": "parquet",
        "gpu_only": True,
        "topk_refine": int(args.topk_refine),
        "refined_count": int(refined_count),
        "q_method_first_pass": args.q_method,
        "streams": int(n_streams),
        "outputs": ["metrics.json", "candidates.json", "summary.json"],
    }
    _write_json(os.path.join(args.out, "summary.json"), summary)

    print(
        f"[gpu_evaluator_parquet] Parquet->GPU | groupby-chunk + pinned/streams={n_streams} | "
        f"TopK refine={int(args.topk_refine)} (refined={refined_count}) | "
        f"PASS={len(candidates)} / G={len(metrics_out)}"
    )

if __name__ == "__main__":
    main()
