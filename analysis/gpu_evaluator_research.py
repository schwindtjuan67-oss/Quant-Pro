#!/usr/bin/env python3
# analysis/gpu_evaluator.py
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================
# Backend: CuPy or CPU fallback
# ============================
try:
    import cupy as cp
    GPU_ENABLED = True
except Exception:
    cp = None
    GPU_ENABLED = False

# ----------------------------
# Defaults (paths)
# ----------------------------
TENSORS_DIR = os.getenv("PIPELINE_TENSORS_DIR", "results/pipeline_tensors")
OUT_DIR = os.getenv("PIPELINE_EVAL_DIR", "results/pipeline_eval")

# ----------------------------
# Thresholds (env-compatible)
# ----------------------------
MIN_TRADES = int(os.getenv("PIPELINE_MIN_TRADES", "300"))
MIN_R_OBS = int(os.getenv("PIPELINE_MIN_R_OBS", "200"))

TH_EXPECTANCY = float(os.getenv("PIPELINE_TH_EXPECTANCY", "0.05"))
TH_SORTINO = float(os.getenv("PIPELINE_TH_SORTINO", "1.50"))
TH_PF = float(os.getenv("PIPELINE_TH_PF", "1.30"))
TH_DD = float(os.getenv("PIPELINE_TH_DD", "-0.20"))
TH_WINRATE = float(os.getenv("PIPELINE_TH_WINRATE", "0.40"))
TH_WORST5 = float(os.getenv("PIPELINE_TH_WORST5", "-1.50"))

TOP_N_JSON = int(os.getenv("PIPELINE_TOP_N_JSON", "50"))
EPS = 1e-12

# ----------------------------
# Score v1 (simple, estable)
# ----------------------------
S_CAP_SORTINO = float(os.getenv("PIPELINE_SCORE_CAP_SORTINO", "5.0"))
S_WIN_CLIP_LO = float(os.getenv("PIPELINE_SCORE_WIN_CLIP_LO", "0.20"))
S_WIN_CLIP_HI = float(os.getenv("PIPELINE_SCORE_WIN_CLIP_HI", "0.80"))

# ----------------------------
# Score v2 (fase 2: robustez)
# ----------------------------
S2_STD_PENALTY = float(os.getenv("PIPELINE_S2_STD_PENALTY", "1.0"))
S2_TAIL_PENALTY = float(os.getenv("PIPELINE_S2_TAIL_PENALTY", "1.0"))
S2_TRADES_SOFT = int(os.getenv("PIPELINE_S2_TRADES_SOFT", "800"))
S2_DD_WEIGHT = float(os.getenv("PIPELINE_S2_DD_WEIGHT", "1.0"))

# ----------------------------
# Quantile approx config
# ----------------------------
# "sample" o "hist"
Q_METHOD = os.getenv("PIPELINE_Q_METHOD", "sample").lower().strip()
Q_SAMPLE_K = int(os.getenv("PIPELINE_Q_SAMPLE_K", "512"))
Q_HIST_BINS = int(os.getenv("PIPELINE_Q_HIST_BINS", "1024"))
Q_HIST_CLIP = float(os.getenv("PIPELINE_Q_HIST_CLIP", "10.0"))

# ----------------------------
# Batch / multi-GPU
# ----------------------------
DEFAULT_BATCH_G = int(os.getenv("PIPELINE_BATCH_G", "512"))
DEFAULT_DEVICES = os.getenv("PIPELINE_DEVICES", "")  # e.g. "0,1"

# ----------------------------
# Top-K refinement (GPU exact)
# ----------------------------
TOPK_REFINE = int(os.getenv("PIPELINE_TOPK_REFINE", "512"))  # 3060 Ti friendly default

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

# ============================
# CPU helpers (solo fallback / json)
# ============================
def _score_v1(expectancy: float, sortino: float, winrate: float, max_dd: float) -> float:
    s = sortino
    if not np.isfinite(s):
        s = S_CAP_SORTINO if s > 0 else 0.0
    s = float(np.clip(s, 0.0, S_CAP_SORTINO))
    w = float(np.clip(winrate, S_WIN_CLIP_LO, S_WIN_CLIP_HI))
    dd_fac = (1.0 + float(max_dd)) if np.isfinite(max_dd) else 0.0
    dd_fac = float(max(dd_fac, 0.0))
    return float(expectancy) * s * w * dd_fac

def _score_v2(expectancy: float, sortino: float, winrate: float, max_dd: float,
             std_r: float, worst5: float, n_trades: int) -> float:
    base = _score_v1(expectancy, sortino, winrate, max_dd)
    if not np.isfinite(base) or base <= 0:
        return float(base)

    std_r = std_r if np.isfinite(std_r) else 1e9
    worst5 = worst5 if np.isfinite(worst5) else -1e9
    max_dd = max_dd if np.isfinite(max_dd) else -1.0

    stability = 1.0 / (1.0 + (max(std_r, 0.0) * max(S2_STD_PENALTY, 0.0)))
    tail = 1.0 / (1.0 + (max(0.0, -worst5) * max(S2_TAIL_PENALTY, 0.0)))

    soft = max(int(S2_TRADES_SOFT), 1)
    trades_fac = float(n_trades) / float(n_trades + soft)

    dd_fac = max(1.0 + max_dd, 0.0)
    dd_extra = dd_fac ** max(S2_DD_WEIGHT, 0.0)

    return float(base) * float(stability) * float(tail) * float(trades_fac) * float(dd_extra)

# ============================
# GPU quantile approx
# ============================
def _worst5pct_sample_gpu(Rb, Mb, q: float = 0.05, k: int = 512):
    cp_ = cp
    G, T = Rb.shape
    rnd = cp_.random.random((G, T), dtype=cp_.float32)
    rnd = cp_.where(Mb, rnd, -cp_.inf)
    kk = min(k, T)
    idx = cp_.argpartition(rnd, T - kk, axis=1)[:, T - kk:]  # (G, kk)
    samp = cp_.take_along_axis(Rb, idx, axis=1)
    m_samp = cp_.take_along_axis(Mb, idx, axis=1)
    samp = cp_.where(m_samp, samp, cp_.nan)
    return cp_.nanquantile(samp.astype(cp_.float32), q, axis=1)

def _worst5pct_hist_gpu(Rb, Mb, q: float = 0.05, bins: int = 1024, clip: float = 10.0):
    cp_ = cp
    G, T = Rb.shape
    x = cp_.clip(Rb.astype(cp_.float32), -clip, clip)
    scaled = (x + clip) * (bins / (2.0 * clip))
    b = cp_.floor(scaled).astype(cp_.int32)
    b = cp_.clip(b, 0, bins - 1)
    b = cp_.where(Mb, b, -1)

    hist = cp_.zeros((G, bins), dtype=cp_.int32)
    rows = cp_.repeat(cp_.arange(G, dtype=cp_.int32), T)
    bb = b.reshape(-1)
    mask = bb >= 0
    rows = rows[mask]
    bb = bb[mask]
    cp_.scatter_add(hist, (rows, bb), 1)

    cdf = cp_.cumsum(hist, axis=1).astype(cp_.float32)
    totals = cdf[:, -1:]
    totals = cp_.where(totals <= 0, cp_.nan, totals)
    cdfn = cdf / totals
    ge = cdfn >= q
    idx = cp_.argmax(ge, axis=1).astype(cp_.int32)
    val = (-clip + (idx.astype(cp_.float32) + 0.5) * (2.0 * clip / bins))
    val = cp_.where(cp_.isfinite(totals[:, 0]), val, cp_.nan)
    return val

# ============================
# GPU metrics: PF, DD, Sortino (vectorized)
# ============================
def _profit_factor_gpu(PNLb, Mb):
    cp_ = cp
    M_f = Mb.astype(cp_.float32)
    x = PNLb.astype(cp_.float32)

    pos = cp_.sum(cp_.maximum(x, 0.0) * M_f, axis=1)
    neg = cp_.sum(cp_.minimum(x, 0.0) * M_f, axis=1)  # negative or 0

    absneg = cp_.abs(neg)
    pf = cp_.where(absneg <= EPS, cp_.inf, pos / absneg)
    pf = cp_.where(pos <= 0.0, cp_.nan, pf)
    return pf.astype(cp_.float32)

def _max_drawdown_gpu(EQb, Mb):
    """
    Requiere: padding SOLO al final (confirmado).
    Para evitar branches por fila:
    - fill tail invalid con last valid equity
    - peak = cummax
    - dd = (eq-peak)/peak
    - nanmin sobre Mb
    """
    cp_ = cp
    G, T = EQb.shape
    eq = EQb.astype(cp_.float32)
    m = Mb.astype(cp_.bool_)

    # filas sin trades
    nt = cp_.sum(m, axis=1).astype(cp_.int32)

    # last_pos: posición del último True (si nt==0 => -1)
    rev = cp_.flip(m, axis=1)
    dist = cp_.argmax(rev, axis=1).astype(cp_.int32)  # 0 si último es True
    last_pos = (T - 1 - dist).astype(cp_.int32)
    last_pos = cp_.where(nt > 0, last_pos, -1)

    # last_val por fila (si no hay trades => nan)
    safe_pos = cp_.clip(last_pos, 0, T - 1)
    last_val = cp_.take_along_axis(eq, safe_pos[:, None], axis=1)[:, 0]
    last_val = cp_.where(nt > 0, last_val, cp_.nan)

    # fill: invalid tail -> last_val (ok porque padding SOLO al final)
    filled = cp_.where(m, eq, last_val[:, None])

    peak = cp_.maximum.accumulate(filled, axis=1)
    peak = cp_.where(peak == 0, cp_.nan, peak)
    dd = (filled - peak) / peak
    dd = cp_.where(m, dd, cp_.nan)
    mdd = cp_.nanmin(dd, axis=1)
    return mdd.astype(cp_.float32)

def _sortino_downside_gpu(Rb, Mb):
    """
    Sortino = mean(all valid r) / std(downside r, ddof=1)
    downside = r < 0 AND Mb
    Vectorizado.
    """
    cp_ = cp
    m = Mb.astype(cp_.bool_)
    M_f = m.astype(cp_.float32)

    nt = cp_.sum(m, axis=1).astype(cp_.int32)
    denom = cp_.maximum(nt.astype(cp_.float32), 1.0)

    r = Rb.astype(cp_.float32)
    mu = cp_.sum(r * M_f, axis=1) / denom

    md = m & (r < 0)
    nd = cp_.sum(md, axis=1).astype(cp_.int32)
    md_f = md.astype(cp_.float32)

    sum_d = cp_.sum(r * md_f, axis=1).astype(cp_.float32)
    sum2_d = cp_.sum((r * r) * md_f, axis=1).astype(cp_.float32)

    # var ddof=1: (sum2 - sum^2/n) / (n-1)
    nd_f = nd.astype(cp_.float32)
    mean_d = cp_.where(nd_f > 0, sum_d / cp_.maximum(nd_f, 1.0), 0.0)
    var_num = sum2_d - (sum_d * mean_d)  # sum2 - sum^2/n
    var = cp_.where(nd_f > 1.0, var_num / cp_.maximum(nd_f - 1.0, 1.0), 0.0)
    var = cp_.maximum(var, 0.0)
    dstd = cp_.sqrt(var)

    # si no hay downside => inf
    sortino = cp_.where(nd == 0, cp_.inf, cp_.where(dstd <= 0, cp_.where(mu > 0, cp_.inf, -cp_.inf), mu / dstd))
    # si no trades => nan
    sortino = cp_.where(nt > 0, sortino, cp_.nan)
    return sortino.astype(cp_.float32)

# ============================
# GPU scores (vectorized)
# ============================
def _score_v1_gpu(expectancy, sortino, winrate, max_dd):
    cp_ = cp
    s = sortino.astype(cp_.float32)

    # sanitize sortino
    s = cp_.where(cp_.isfinite(s), s, cp_.where(s > 0, S_CAP_SORTINO, 0.0))
    s = cp_.clip(s, 0.0, S_CAP_SORTINO).astype(cp_.float32)

    w = cp_.clip(winrate.astype(cp_.float32), S_WIN_CLIP_LO, S_WIN_CLIP_HI)
    dd_fac = cp_.where(cp_.isfinite(max_dd), (1.0 + max_dd.astype(cp_.float32)), 0.0)
    dd_fac = cp_.maximum(dd_fac, 0.0)

    out = expectancy.astype(cp_.float32) * s * w * dd_fac
    return out.astype(cp_.float32)

def _score_v2_gpu(expectancy, sortino, winrate, max_dd, std_r, worst5, n_trades):
    cp_ = cp
    base = _score_v1_gpu(expectancy, sortino, winrate, max_dd)

    std_r = cp_.where(cp_.isfinite(std_r), std_r.astype(cp_.float32), 1e9).astype(cp_.float32)
    worst5 = cp_.where(cp_.isfinite(worst5), worst5.astype(cp_.float32), -1e9).astype(cp_.float32)
    max_dd = cp_.where(cp_.isfinite(max_dd), max_dd.astype(cp_.float32), -1.0).astype(cp_.float32)

    stability = 1.0 / (1.0 + (cp_.maximum(std_r, 0.0) * max(S2_STD_PENALTY, 0.0)))
    tail = 1.0 / (1.0 + (cp_.maximum(0.0, -worst5) * max(S2_TAIL_PENALTY, 0.0)))

    soft = float(max(int(S2_TRADES_SOFT), 1))
    nt = n_trades.astype(cp_.float32)
    trades_fac = nt / (nt + soft)

    dd_fac = cp_.maximum(1.0 + max_dd, 0.0)
    dd_extra = dd_fac ** float(max(S2_DD_WEIGHT, 0.0))

    out = base * stability.astype(cp_.float32) * tail.astype(cp_.float32) * trades_fac.astype(cp_.float32) * dd_extra.astype(cp_.float32)

    # si base <= 0 => mantener (compat con CPU)
    out = cp_.where((cp_.isfinite(base) & (base > 0)), out, base)
    return out.astype(cp_.float32)

# ============================
# Top-K exact refinement (GPU)
# ============================
def _refine_topk_exact_gpu(R_all, M_all, EQ_all, PNL_all, worst5, sortino, pf, max_dd, std_r, expectancy, winrate, n_trades, scores_v1, scores_v2, topk: int):
    """
    Recalcula EXACT en GPU, SOLO en top-k (por score_v2 actual):
    - worst5 exact: quantile exact sobre valid r
    - sortino exact: std(downside) ddof=1 sobre valid downside
    Re-scorea v1/v2 solo en esos índices.
    """
    cp_ = cp
    G, T = R_all.shape
    if topk <= 0:
        return worst5, sortino, scores_v1, scores_v2

    k = min(int(topk), int(G))
    # top-k indices por score_v2
    idx = cp_.argpartition(scores_v2, G - k)[G - k:]  # no ordenado
    # orden descendente dentro del topk
    idx = idx[cp_.argsort(-scores_v2[idx])]

    for j in idx.tolist():
        Mb = M_all[j].astype(cp_.bool_)
        if int(cp_.sum(Mb).item()) <= 0:
            continue

        r = R_all[j].astype(cp_.float32)
        rv = r[Mb]

        # worst5 exact
        try:
            w5 = cp_.quantile(rv, 0.05).astype(cp_.float32)
        except Exception:
            w5 = cp_.nan
        worst5[j] = w5

        # sortino exact (downside-only std ddof=1)
        dn = rv[rv < 0]
        if dn.size == 0:
            srt = cp_.inf
        else:
            if dn.size > 1:
                dstd = cp_.std(dn.astype(cp_.float32), ddof=1)
            else:
                dstd = cp_.float32(0.0)
            mu = cp_.mean(rv.astype(cp_.float32))
            if float(dstd.item()) <= 0.0:
                srt = cp_.inf if float(mu.item()) > 0 else -cp_.inf
            else:
                srt = mu / dstd
        sortino[j] = srt.astype(cp_.float32)

        # re-score
        sv1 = _score_v1_gpu(expectancy[j:j+1], sortino[j:j+1], winrate[j:j+1], max_dd[j:j+1])[0]
        sv2 = _score_v2_gpu(expectancy[j:j+1], sortino[j:j+1], winrate[j:j+1], max_dd[j:j+1],
                            std_r[j:j+1], worst5[j:j+1], n_trades[j:j+1])[0]
        scores_v1[j] = sv1
        scores_v2[j] = sv2

    return worst5, sortino, scores_v1, scores_v2

# ============================
# Core batch evaluator on one device
# ============================
def _eval_batches_on_device(
    device_id: int,
    R_path: str,
    M_path: str,
    EQ_path: Optional[str],
    PNL_path: Optional[str],
    batch_g: int,
    q_method: str,
    q: float,
) -> Dict[str, np.ndarray]:
    if not GPU_ENABLED:
        raise RuntimeError("CuPy not available but GPU path was requested.")

    with cp.cuda.Device(device_id):
        R_all = cp.load(R_path)  # (G,T)
        M_all = cp.load(M_path)  # (G,T)
        G, T = R_all.shape

        EQ_all = cp.load(EQ_path) if (EQ_path and os.path.exists(EQ_path)) else None
        PNL_all = cp.load(PNL_path) if (PNL_path and os.path.exists(PNL_path)) else None

        # outputs on GPU (float32), then to CPU at end
        expectancy = cp.full((G,), cp.nan, dtype=cp.float32)
        winrate = cp.full((G,), cp.nan, dtype=cp.float32)
        n_trades = cp.zeros((G,), dtype=cp.int32)
        worst5 = cp.full((G,), cp.nan, dtype=cp.float32)
        std_r = cp.full((G,), cp.nan, dtype=cp.float32)

        sortino = cp.full((G,), cp.nan, dtype=cp.float32)
        pf = cp.full((G,), cp.nan, dtype=cp.float32)
        max_dd = cp.full((G,), cp.nan, dtype=cp.float32)

        scores_v1 = cp.full((G,), -1e9, dtype=cp.float32)
        scores_v2 = cp.full((G,), -1e9, dtype=cp.float32)

        for s in range(0, G, batch_g):
            e = min(G, s + batch_g)
            Rb = R_all[s:e].astype(cp.float32)
            Mb = M_all[s:e].astype(cp.bool_)
            Gb = e - s

            M_f = Mb.astype(cp.float32)
            nt = cp.sum(Mb, axis=1).astype(cp.int32)
            denom = cp.maximum(nt.astype(cp.float32), 1.0)

            exp = cp.sum(Rb * M_f, axis=1) / denom
            win = cp.sum((Rb > 0) * M_f, axis=1) / denom

            er2 = cp.sum((Rb * Rb) * M_f, axis=1) / denom
            var = cp.maximum(er2 - exp * exp, 0.0)
            st = cp.sqrt(var)

            # worst5 approx
            if q_method == "hist":
                w5 = _worst5pct_hist_gpu(Rb, Mb, q=q, bins=Q_HIST_BINS, clip=Q_HIST_CLIP)
            else:
                w5 = _worst5pct_sample_gpu(Rb, Mb, q=q, k=Q_SAMPLE_K)

            # sortino GPU (vectorized)
            srt = _sortino_downside_gpu(Rb, Mb)

            # pf GPU
            if PNL_all is not None:
                PNLb = PNL_all[s:e].astype(cp.float32)
                pfb = _profit_factor_gpu(PNLb, Mb)
            else:
                # fallback: PF sobre r como proxy (igual que tu versión anterior)
                pfb = _profit_factor_gpu(Rb, Mb)

            # dd GPU (si hay EQ)
            if EQ_all is not None:
                EQb = EQ_all[s:e].astype(cp.float32)
                ddb = _max_drawdown_gpu(EQb, Mb)
            else:
                ddb = cp.full((Gb,), cp.nan, dtype=cp.float32)

            # scores GPU
            sv1 = _score_v1_gpu(exp, srt, win, ddb)
            sv2 = _score_v2_gpu(exp, srt, win, ddb, st, w5, nt)

            # store
            expectancy[s:e] = exp
            winrate[s:e] = win
            n_trades[s:e] = nt
            worst5[s:e] = w5
            std_r[s:e] = st
            sortino[s:e] = srt
            pf[s:e] = pfb
            max_dd[s:e] = ddb
            scores_v1[s:e] = sv1
            scores_v2[s:e] = sv2

        # Top-K exact refinement (GPU) -> reemplaza worst5/sortino y re-scorea solo esos
        worst5, sortino, scores_v1, scores_v2 = _refine_topk_exact_gpu(
            R_all=R_all,
            M_all=M_all.astype(cp.bool_),
            EQ_all=EQ_all,
            PNL_all=PNL_all,
            worst5=worst5,
            sortino=sortino,
            pf=pf,
            max_dd=max_dd,
            std_r=std_r,
            expectancy=expectancy,
            winrate=winrate,
            n_trades=n_trades,
            scores_v1=scores_v1,
            scores_v2=scores_v2,
            topk=TOPK_REFINE,
        )

        # move to CPU
        return {
            "expectancy": cp.asnumpy(expectancy).astype(np.float64),
            "winrate": cp.asnumpy(winrate).astype(np.float64),
            "n_trades": cp.asnumpy(n_trades).astype(np.int32),
            "worst5": cp.asnumpy(worst5).astype(np.float64),
            "sortino": cp.asnumpy(sortino).astype(np.float64),
            "pf": cp.asnumpy(pf).astype(np.float64),
            "max_dd": cp.asnumpy(max_dd).astype(np.float64),
            "std_r": cp.asnumpy(std_r).astype(np.float64),
            "scores_v1": cp.asnumpy(scores_v1).astype(np.float64),
            "scores_v2": cp.asnumpy(scores_v2).astype(np.float64),
        }

# ============================
# Multi-GPU orchestrator
# ============================
def _parse_devices(devices_str: str) -> List[int]:
    devices_str = (devices_str or "").strip()
    if not devices_str:
        return [0]
    out: List[int] = []
    for p in devices_str.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out if out else [0]

def _split_ranges(G: int, n_parts: int) -> List[Tuple[int, int]]:
    if n_parts <= 1:
        return [(0, G)]
    base = G // n_parts
    rem = G % n_parts
    ranges: List[Tuple[int, int]] = []
    s = 0
    for i in range(n_parts):
        add = base + (1 if i < rem else 0)
        e = s + add
        ranges.append((s, e))
        s = e
    return ranges

def _eval_multi_gpu(
    devices: List[int],
    R_path: str,
    M_path: str,
    EQ_path: Optional[str],
    PNL_path: Optional[str],
    batch_g: int,
    q_method: str,
    q: float,
) -> Dict[str, np.ndarray]:
    if not GPU_ENABLED:
        raise RuntimeError("CuPy not available but multi-GPU was requested.")

    with cp.cuda.Device(devices[0]):
        G = int(cp.load(R_path).shape[0])

    ranges = _split_ranges(G, len(devices))

    merged: Dict[str, np.ndarray] = {}
    keys = ["expectancy","winrate","n_trades","worst5","sortino","pf","max_dd","std_r","scores_v1","scores_v2"]
    for key in keys:
        merged[key] = np.full(G, np.nan if key != "n_trades" else 0, dtype=np.float64 if key != "n_trades" else np.int32)

    for dev, (s, e) in zip(devices, ranges):
        res = _eval_batches_on_device(dev, R_path, M_path, EQ_path, PNL_path, batch_g, q_method, q)
        for k, arr in res.items():
            merged[k][s:e] = arr[s:e]

    return merged

# ============================
# Main
# ============================
def main() -> None:
    ap = argparse.ArgumentParser("GPU Evaluator (CuPy batch + approx quantile + sortino gpu + topk exact refine)")
    ap.add_argument("--tensors", default=TENSORS_DIR, help="dir con R.npy, M.npy, groups.json (y opcional EQ.npy, PNL.npy)")
    ap.add_argument("--out", default=OUT_DIR, help="dir de salida")
    ap.add_argument("--top", type=int, default=TOP_N_JSON, help="top N a exportar a candidates.json")
    ap.add_argument("--batch-g", type=int, default=DEFAULT_BATCH_G, help="batch size en dimensión G (grupos) para GPU")
    ap.add_argument("--devices", default=DEFAULT_DEVICES, help='multi-gpu: "0" o "0,1,2"')
    ap.add_argument("--q-method", default=Q_METHOD, choices=["sample","hist"], help="aprox quantile method")
    args = ap.parse_args()

    tensors_dir = args.tensors
    out_dir = args.out
    _ensure_dir(out_dir)

    r_path = os.path.join(tensors_dir, "R.npy")
    m_path = os.path.join(tensors_dir, "M.npy")
    g_path = os.path.join(tensors_dir, "groups.json")

    if not (os.path.exists(r_path) and os.path.exists(m_path) and os.path.exists(g_path)):
        raise SystemExit(f"[gpu_evaluator] Missing inputs in {tensors_dir}: need R.npy, M.npy, groups.json")

    groups: List[Dict[str, Any]] = _read_json(g_path)

    eq_path = os.path.join(tensors_dir, "EQ.npy")
    pnl_path = os.path.join(tensors_dir, "PNL.npy")
    EQ_path = eq_path if os.path.exists(eq_path) else None
    PNL_path = pnl_path if os.path.exists(pnl_path) else None

    if not GPU_ENABLED:
        raise SystemExit("[gpu_evaluator] CuPy not available. Install cupy-cudaXX (matching your CUDA).")

    devices = _parse_devices(args.devices)
    res = _eval_multi_gpu(
        devices=devices,
        R_path=r_path,
        M_path=m_path,
        EQ_path=EQ_path,
        PNL_path=PNL_path,
        batch_g=int(args.batch_g),
        q_method=args.q_method,
        q=0.05,
    )

    expectancy = res["expectancy"]
    winrate = res["winrate"]
    n_trades = res["n_trades"]
    worst5 = res["worst5"]
    sortino = res["sortino"]
    pf = res["pf"]
    max_dd = res["max_dd"]
    std_r = res["std_r"]
    scores_v1 = res["scores_v1"]
    scores_v2 = res["scores_v2"]

    has_EQ = EQ_path is not None
    dd_ok = np.isfinite(max_dd) & (max_dd > TH_DD) if has_EQ else np.zeros_like(max_dd, dtype=bool)

    pass_mask = (
        (n_trades >= MIN_TRADES) &
        (n_trades >= MIN_R_OBS) &
        (expectancy > TH_EXPECTANCY) &
        (sortino >= TH_SORTINO) &
        (pf >= TH_PF) &
        dd_ok &
        (winrate >= TH_WINRATE) &
        (worst5 > TH_WORST5)
    )

    if not has_EQ:
        print("[gpu_evaluator][WARN] EQ.npy not found => max_dd=NaN => PASS will be 0 due to DD gate.")

    np.save(os.path.join(out_dir, "scores.npy"), scores_v2.astype(np.float32))
    np.save(os.path.join(out_dir, "scores_v1.npy"), scores_v1.astype(np.float32))

    metrics: List[Dict[str, Any]] = []
    G = len(groups)
    for i, meta in enumerate(groups):
        metrics.append({
            "group_index": i,
            "symbol": meta.get("symbol"),
            "regime": meta.get("regime"),
            "seed": meta.get("seed"),
            "window": meta.get("window"),
            "params_key": meta.get("params_key"),
            "n_trades": int(n_trades[i]),
            "expectancy_r": float(expectancy[i]) if np.isfinite(expectancy[i]) else None,
            "sortino": float(sortino[i]) if np.isfinite(sortino[i]) else None,
            "profit_factor": float(pf[i]) if np.isfinite(pf[i]) else None,
            "max_dd": float(max_dd[i]) if np.isfinite(max_dd[i]) else None,
            "winrate": float(winrate[i]) if np.isfinite(winrate[i]) else None,
            "worst_5pct_r": float(worst5[i]) if np.isfinite(worst5[i]) else None,
            "std_r": float(std_r[i]) if np.isfinite(std_r[i]) else None,
            "PASS_SHADOW": bool(pass_mask[i]),
            "score_v1": float(scores_v1[i]) if np.isfinite(scores_v1[i]) else None,
            "score_v2": float(scores_v2[i]) if np.isfinite(scores_v2[i]) else None,
            "params": meta.get("params", {}),
        })
    _write_json(os.path.join(out_dir, "metrics.json"), metrics)

    pass_idx = np.where(pass_mask)[0]
    if pass_idx.size:
        order = np.argsort(-scores_v2[pass_idx])
        top_idx = pass_idx[order][: int(args.top)]
        candidates = []
        for i in top_idx.tolist():
            candidates.append({
                "symbol": groups[i].get("symbol"),
                "regime": groups[i].get("regime"),
                "seed": groups[i].get("seed"),
                "window": groups[i].get("window"),
                "params": groups[i].get("params", {}),
                "metrics": {
                    "trades": int(n_trades[i]),
                    "expectancy_r": float(expectancy[i]),
                    "sortino": float(sortino[i]) if np.isfinite(sortino[i]) else float("inf"),
                    "profit_factor": float(pf[i]) if np.isfinite(pf[i]) else float("nan"),
                    "max_dd": float(max_dd[i]) if np.isfinite(max_dd[i]) else float("nan"),
                    "winrate": float(winrate[i]),
                    "worst_5pct_r": float(worst5[i]),
                    "std_r": float(std_r[i]) if np.isfinite(std_r[i]) else float("nan"),
                    "score_v1": float(scores_v1[i]),
                    "score_v2": float(scores_v2[i]),
                },
            })
    else:
        candidates = []
    _write_json(os.path.join(out_dir, "candidates.json"), candidates)

    csv_path = os.path.join(out_dir, "candidates.csv")
    header = [
        "group_index","symbol","regime","seed","window","n_trades",
        "expectancy_r","sortino","profit_factor","max_dd","winrate",
        "worst_5pct_r","std_r","PASS_SHADOW","score_v1","score_v2","params_key"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        order_all = np.argsort(-scores_v2)
        for i in order_all.tolist():
            meta = groups[i]
            row = [
                str(i),
                str(meta.get("symbol") or ""),
                str(meta.get("regime") or ""),
                str(meta.get("seed") or ""),
                str(meta.get("window") or ""),
                str(int(n_trades[i])),
                str(float(expectancy[i])) if np.isfinite(expectancy[i]) else "",
                str(float(sortino[i])) if np.isfinite(sortino[i]) else "",
                str(float(pf[i])) if np.isfinite(pf[i]) else "",
                str(float(max_dd[i])) if np.isfinite(max_dd[i]) else "",
                str(float(winrate[i])) if np.isfinite(winrate[i]) else "",
                str(float(worst5[i])) if np.isfinite(worst5[i]) else "",
                str(float(std_r[i])) if np.isfinite(std_r[i]) else "",
                "1" if bool(pass_mask[i]) else "0",
                str(float(scores_v1[i])) if np.isfinite(scores_v1[i]) else "",
                str(float(scores_v2[i])) if np.isfinite(scores_v2[i]) else "",
                str(meta.get("params_key") or ""),
            ]
            f.write(",".join(row) + "\n")

    summary = {
        "tensors_dir": tensors_dir,
        "out_dir": out_dir,
        "G": int(G),
        "MIN_TRADES": int(MIN_TRADES),
        "MIN_R_OBS": int(MIN_R_OBS),
        "thresholds": {
            "expectancy": TH_EXPECTANCY,
            "sortino": TH_SORTINO,
            "pf": TH_PF,
            "dd": TH_DD,
            "winrate": TH_WINRATE,
            "worst5": TH_WORST5,
        },
        "quantile": {
            "method": args.q_method,
            "sample_k": int(Q_SAMPLE_K),
            "hist_bins": int(Q_HIST_BINS),
            "hist_clip": float(Q_HIST_CLIP),
        },
        "batch_g": int(args.batch_g),
        "devices": devices,
        "topk_refine": int(TOPK_REFINE),
        "pass_count": int(len(candidates)),
        "has_EQ": bool(EQ_path is not None),
        "has_PNL": bool(PNL_path is not None),
        "outputs": ["scores.npy","scores_v1.npy","metrics.json","candidates.json","candidates.csv","summary.json"],
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)

    print(f"[gpu_evaluator] GPU=True devices={devices} batch_g={args.batch_g} topk_refine={TOPK_REFINE}")
    print(f"[gpu_evaluator] Saved -> {out_dir}")
    print(f"[gpu_evaluator] PASS={len(candidates)} / G={G}")

if __name__ == "__main__":
    main()





