from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import calendar
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from analysis.grid_metrics import (
    compute_metrics_from_trades,
    compute_score,
)

# =========================================================
# Window -> dates (reproducible)
# =========================================================

_WINDOW_RE1 = re.compile(r"^\d{4}-\d{2}[_-]\d{4}-\d{2}$")
_WINDOW_RE2 = re.compile(r"^\d{4}_\d{2}_\d{4}_\d{2}$")


def window_to_dates(window: str) -> Tuple[str, str]:
    """
    Acepta:
      - '2020-01_2021-12'
      - '2020-01-2021-12'
      - '2020_01_2021_12'
    -> ('2020-01-01', '2021-12-31')
    """
    s = str(window).strip()
    if not s or s.lower() == "none":
        raise ValueError("[GRID] --window vacío/None")

    # normalizar formatos
    if _WINDOW_RE2.match(s):
        # 2020_01_2021_12 -> 2020-01_2021-12
        a, b, c, d = s.split("_")
        s = f"{a}-{b}_{c}-{d}"

    if _WINDOW_RE1.match(s):
        # puede venir con '_' o con '-'
        if "_" in s:
            start, end = s.split("_", 1)
        else:
            # '2020-01-2021-12' -> partir por mitad fija
            start, end = s[:7], s[-7:]

        y1, m1 = map(int, start.split("-"))
        y2, m2 = map(int, end.split("-"))

        from_date = f"{y1:04d}-{m1:02d}-01"
        last_day = calendar.monthrange(y2, m2)[1]
        to_date = f"{y2:04d}-{m2:02d}-{last_day:02d}"
        return from_date, to_date

    raise ValueError(f"[GRID] Invalid --window format: {window!r}. Expected YYYY-MM_YYYY-MM (or compatible).")


# =========================================================
# Utils
# =========================================================

def _stable_key(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _ensure_dir(p: str) -> None:
    os.makedirs(p or ".", exist_ok=True)

def _load_json_any(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(p: str, data: Any) -> None:
    _ensure_dir(os.path.dirname(p) or ".")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _now_ts() -> int:
    return int(time.time())

# =========================================================
# Trade loading (best-effort)
# =========================================================

def _find_trades_file(output_dir: str) -> Optional[str]:
    candidates: List[str] = []
    for pat in ("*trades*.csv", "*trade*.csv", "*fills*.csv", "*executions*.csv"):
        candidates.extend(glob.glob(os.path.join(output_dir, pat)))

    for pat in ("**/*trades*.csv", "**/*trade*.csv"):
        candidates.extend(glob.glob(os.path.join(output_dir, pat), recursive=True))

    candidates = [c for c in candidates if "bar" not in os.path.basename(c).lower()]
    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0, reverse=True)
    return candidates[0]

def _load_trades_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)

    for t in out:
        for k in ("pnl_r", "holding_time_sec", "timestamp_ms", "timestamp"):
            if k in t and t[k] not in (None, ""):
                try:
                    t[k] = float(t[k]) if k not in ("timestamp_ms", "timestamp") else int(float(t[k]))
                except Exception:
                    pass
    return out

# =========================================================
# Backtest runner (subprocess)
# =========================================================

@dataclass
class RunResult:
    ok: bool
    params_key: str
    output_dir: Optional[str]
    trades: int
    equity_r: float
    metrics: Dict[str, float]
    score: float
    stderr_tail: str = ""
    stdout_tail: str = ""

OUTPUT_DIR_RE = re.compile(r"output_dir\s*=\s*(.+)$", re.IGNORECASE)
TRADES_EQUITY_RE = re.compile(r"trades\s*=\s*(\d+)\s+equity_r\s*=\s*([-\d\.]+)", re.IGNORECASE)

def _parse_stdout_for_summary(stdout: str) -> Tuple[int, float, Optional[str]]:
    trades = 0
    equity_r = 0.0
    outdir = None

    for line in stdout.splitlines():
        m = OUTPUT_DIR_RE.search(line.strip())
        if m:
            outdir = m.group(1).strip()

    for line in stdout.splitlines():
        m = TRADES_EQUITY_RE.search(line.strip())
        if m:
            trades = int(m.group(1))
            equity_r = float(m.group(2))

    return trades, equity_r, outdir

def _run_one_backtest(
    python_exe: str,
    base_cmd: List[str],
    tmp_config_path: str,
    quiet: bool = True,
) -> Tuple[int, str, str]:
    cmd = [python_exe] + base_cmd + ["--config", tmp_config_path]
    if quiet and "--quiet" not in cmd:
        cmd.append("--quiet")

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, p.stdout, p.stderr

# =========================================================
# Grid definition / robust candidates
# =========================================================

def _expand_grid_dict(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    items = list(grid.items())
    if not items:
        return [{}]

    combos = [{}]
    for k, vals in items:
        new_combos = []
        for c in combos:
            for v in vals:
                d = dict(c)
                d[k] = v
                new_combos.append(d)
        combos = new_combos
    return combos

def _grid_to_patches(grid_obj: Any) -> List[Dict[str, Any]]:
    """
    Acepta:
      - dict[str, list] (grid cartesiano)
      - list[dict] (candidatos)
      - list[{"params": {...}, ...}] (robust_candidates.json)
    """
    if isinstance(grid_obj, dict):
        # grid cartesiano
        return _expand_grid_dict(grid_obj)

    if isinstance(grid_obj, list):
        patches: List[Dict[str, Any]] = []
        for it in grid_obj:
            if isinstance(it, dict) and isinstance(it.get("params"), dict):
                patches.append(it["params"])
            elif isinstance(it, dict):
                patches.append(it)
        if not patches:
            raise ValueError("[GRID] --grid list vacío o sin dicts utilizables")
        return patches

    raise ValueError(f"[GRID] --grid tiene tipo no soportado: {type(grid_obj)}")

def _apply_patch_to_config(cfg: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(cfg))  # deep copy

    if isinstance(out.get("strategy_params"), dict):
        params = out["strategy_params"]
    elif isinstance(out.get("strategy"), dict) and isinstance(out["strategy"].get("params"), dict):
        params = out["strategy"]["params"]
    else:
        out["strategy_params"] = {}
        params = out["strategy_params"]

    for k, v in patch.items():
        params[k] = v

    return out

# =========================================================
# Resume store
# =========================================================

def _load_done_keys(done_jsonl: str) -> set:
    done = set()
    if not os.path.exists(done_jsonl):
        return done
    with open(done_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "params_key" in obj:
                    done.add(obj["params_key"])
            except Exception:
                continue
    return done

def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# =========================================================
# Worker
# =========================================================

def _worker_run(args: Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]) -> RunResult:
    idx, base_cfg, patch, runtime = args

    python_exe = runtime["python_exe"]
    base_cmd = runtime["base_cmd"]
    tmp_dir = runtime["tmp_dir"]

    params_key = _stable_key(patch)

    tmp_config_path = os.path.join(tmp_dir, f"cfg_{idx:05d}_{params_key}.json")
    cfg2 = _apply_patch_to_config(base_cfg, patch)
    _save_json(tmp_config_path, cfg2)

    rc, stdout, stderr = _run_one_backtest(
        python_exe=python_exe,
        base_cmd=base_cmd,
        tmp_config_path=tmp_config_path,
        quiet=bool(runtime.get("quiet", True)),
    )

    trades, equity_r, outdir = _parse_stdout_for_summary(stdout)

    metrics: Dict[str, float] = {}
    score = -1e9
    resolved_outdir: Optional[str] = None

    if outdir:
        resolved_outdir = outdir.strip().replace("\\", "/")
        if not os.path.isabs(resolved_outdir):
            resolved_outdir = os.path.abspath(resolved_outdir)

        trades_csv = _find_trades_file(resolved_outdir)
        if trades_csv:
            try:
                trades_list = _load_trades_from_csv(trades_csv)
                metrics = compute_metrics_from_trades(trades_list)
                score = compute_score(metrics)
                trades = int(metrics.get("trades", trades))
                equity_r = float(metrics.get("equity_r", equity_r))
            except Exception:
                metrics = {}
                score = -1e9

    ok = (rc == 0)

    stdout_tail = "\n".join(stdout.splitlines()[-12:])
    stderr_tail = "\n".join(stderr.splitlines()[-12:])

    return RunResult(
        ok=ok,
        params_key=params_key,
        output_dir=resolved_outdir,
        trades=int(trades),
        equity_r=float(equity_r),
        metrics=metrics if isinstance(metrics, dict) else {},
        score=float(score),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )

# =========================================================
# Main
# =========================================================

SUMMARY_FIELDNAMES = [
    "params_key",
    "ok",
    "trades",
    "equity_r",
    "max_drawdown_r",
    "sortino",
    "profit_factor",
    "expectancy",
    "winrate",
    "avg_trade_duration",
    "score",
    "output_dir",
    "params_json",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="config base (json) para backtest")
    ap.add_argument("--grid", required=True, help="grid json (dict grid OR robust_candidates list)")
    ap.add_argument("--results-dir", default="results", help="carpeta results/")
    ap.add_argument("--tmp-dir", default=None, help="carpeta temporal configs (default: <results-dir>/_grid_tmp)")
    ap.add_argument("--python", default=sys.executable, help="python exe a usar")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2), help="procesos")
    ap.add_argument("--quiet", action="store_true", help="pasa --quiet al backtest")
    ap.add_argument("--resume", action="store_true", help="salta configs ya corridas")

    ap.add_argument("--data", required=True, help="datasets/SYMBOL/1m (o lo que uses)")
    ap.add_argument("--window", default=None, help="YYYY-MM_YYYY-MM (ej: 2020-08_2021-12)")

    ap.add_argument("--from-date", dest="from_date", default=None, help="YYYY-MM-DD (compat)")
    ap.add_argument("--to-date", dest="to_date", default=None, help="YYYY-MM-DD (compat)")

    args = ap.parse_args()

    from_date = args.from_date
    to_date = args.to_date
    if args.window:
        from_date, to_date = window_to_dates(args.window)

    if not from_date or not to_date:
        raise SystemExit("[GRID] Missing date range. Provide --window OR (--from-date AND --to-date).")

    base_cfg = _load_json_any(args.base_config)
    grid_obj = _load_json_any(args.grid)

    results_dir = os.path.abspath(args.results_dir)
    _ensure_dir(results_dir)

    tmp_dir = args.tmp_dir
    if not tmp_dir:
        tmp_dir = os.path.join(results_dir, "_grid_tmp")
    tmp_dir = os.path.abspath(tmp_dir)
    _ensure_dir(tmp_dir)

    runs_jsonl = os.path.join(results_dir, "grid_runs.jsonl")
    summary_csv = os.path.join(results_dir, "grid_summary.csv")
    top_csv = os.path.join(results_dir, "grid_top20.csv")
    best_json = os.path.join(results_dir, "best_config.json")

    done_keys = _load_done_keys(runs_jsonl) if args.resume else set()

    # Convertimos grid a lista de patches (combos)
    combos = _grid_to_patches(grid_obj)

    # map O(1): params_key -> patch
    patch_by_key: Dict[str, Dict[str, Any]] = {}
    for c in combos:
        patch_by_key[_stable_key(c)] = c

    base_cmd = [
        "-m", "backtest.run_backtest",
        "--data", args.data,
        "--from", from_date,
        "--to", to_date,
    ]

    runtime = {
        "python_exe": args.python,
        "base_cmd": base_cmd,
        "tmp_dir": tmp_dir,
        "quiet": bool(args.quiet),
    }

    tasks: List[Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for i, patch in enumerate(combos):
        k = _stable_key(patch)
        if args.resume and k in done_keys:
            continue
        tasks.append((i, base_cfg, patch, runtime))

    print(f"[GRID] window={args.window!r} from={from_date} to={to_date}")
    print(f"[GRID] results_dir={results_dir}")
    print(f"[GRID] tmp_dir={tmp_dir}")
    print(f"[GRID] combos_total={len(combos)} to_run={len(tasks)} workers={args.workers} resume={args.resume}")

    if not os.path.exists(summary_csv):
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
            w.writeheader()

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    if tasks:
        with ctx.Pool(processes=args.workers, maxtasksperchild=20) as pool:
            for rr in pool.imap_unordered(_worker_run, tasks, chunksize=1):
                _append_jsonl(runs_jsonl, {
                    "ts": _now_ts(),
                    "ok": rr.ok,
                    "params_key": rr.params_key,
                    "trades": rr.trades,
                    "equity_r": rr.equity_r,
                    "score": rr.score,
                    "output_dir": rr.output_dir,
                    "stdout_tail": rr.stdout_tail,
                    "stderr_tail": rr.stderr_tail,
                })

                patch = patch_by_key.get(rr.params_key, {})
                row = {
                    "params_key": rr.params_key,
                    "ok": int(bool(rr.ok)),
                    "trades": rr.trades,
                    "equity_r": rr.equity_r,
                    "max_drawdown_r": float(rr.metrics.get("max_drawdown_r", 0.0)),
                    "sortino": float(rr.metrics.get("sortino", 0.0)),
                    "profit_factor": float(rr.metrics.get("profit_factor", 0.0)),
                    "expectancy": float(rr.metrics.get("expectancy", 0.0)),
                    "winrate": float(rr.metrics.get("winrate", 0.0)),
                    "avg_trade_duration": float(rr.metrics.get("avg_trade_duration", 0.0)),
                    "score": rr.score,
                    "output_dir": rr.output_dir or "",
                    "params_json": json.dumps(patch or {}, ensure_ascii=False, separators=(",", ":")),
                }

                with open(summary_csv, "a", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
                    w.writerow(row)

                print(
                    f"[GRID] ok={rr.ok} trades={rr.trades} equity_r={rr.equity_r:.6f} "
                    f"sortino={rr.metrics.get('sortino', 0.0):.3f} score={rr.score:.3f}"
                )

    # rank top20
    rows: List[Dict[str, Any]] = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["score"] = float(r.get("score", -1e9))
            except Exception:
                r["score"] = -1e9
            rows.append(r)

    rows.sort(key=lambda r: float(r.get("score", -1e9)), reverse=True)
    top = rows[:20]

    if top:
        with open(top_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(top[0].keys()))
            w.writeheader()
            w.writerows(top)

        best = top[0]
        best_patch = json.loads(best.get("params_json") or "{}")
        best_cfg = _apply_patch_to_config(base_cfg, best_patch)

        _save_json(best_json, {
            "score": best.get("score"),
            "summary": best,
            "best_patch": best_patch,
            "best_config": best_cfg,
        })

        print(f"[GRID] ✅ top20 -> {top_csv}")
        print(f"[GRID] ✅ best_config -> {best_json}")
    else:
        print("[GRID] No hay resultados para rankear (¿falló todo o no corrió nada?)")

if __name__ == "__main__":
    main()

