#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import requests


# =========================
# Endpoints
# =========================
SPOT_ENDPOINT = "https://api.binance.com/api/v3/klines"
FUTURES_ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"

MAX_LIMIT = 1000  # Binance max
DEFAULT_SLEEP = 0.25  # segundos entre requests
RETRY_SLEEP = 2.0
MAX_RETRIES = 7


# ============================================================
# Time helpers
# ============================================================
def _ym_to_month_range(ym: str) -> Tuple[int, int]:
    """
    "2023-01" -> (start_ms, end_ms_exclusive)
    """
    start = datetime.strptime(ym, "%Y-%m").replace(tzinfo=timezone.utc)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _month_iter(ym_from: str, ym_to: str) -> List[str]:
    """
    Inclusive range: ym_from..ym_to
    """
    a = datetime.strptime(ym_from, "%Y-%m").replace(tzinfo=timezone.utc)
    b = datetime.strptime(ym_to, "%Y-%m").replace(tzinfo=timezone.utc)
    if a > b:
        raise ValueError(f"--from {ym_from} must be <= --to {ym_to}")

    out: List[str] = []
    cur = a
    while cur <= b:
        out.append(cur.strftime("%Y-%m"))
        # next month
        y = cur.year + (1 if cur.month == 12 else 0)
        m = 1 if cur.month == 12 else cur.month + 1
        cur = cur.replace(year=y, month=m)
    return out


def _interval_to_ms(interval: str) -> int:
    """
    Binance intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    s = interval.strip()
    if not s:
        raise ValueError("interval empty")

    unit = s[-1]
    n_str = s[:-1]
    if not n_str.isdigit():
        raise ValueError(f"Unsupported interval: {interval!r}")
    n = int(n_str)

    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 60 * 60_000
    if unit == "d":
        return n * 24 * 60 * 60_000
    if unit == "w":
        return n * 7 * 24 * 60 * 60_000
    if unit == "M":
        # Mes calendario: no es fijo. Para paginar evitamos depender del step.
        # Igual Binance devuelve openTime, y avanzamos a last_open_time + 1ms para evitar loops.
        return -1

    raise ValueError(f"Unsupported interval unit: {interval!r}")


# ============================================================
# Binance fetch
# ============================================================
def fetch_klines(
    endpoint: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    sleep_s: float = DEFAULT_SLEEP,
    verbose: bool = False,
) -> List[list]:
    """
    Fetch klines in [start_ms, end_ms) handling pagination.
    """
    out: List[list] = []
    cur = int(start_ms)

    step_ms = _interval_to_ms(interval)
    # Para interval "1M" (mes calendario), step_ms=-1; avanzamos con guard de 1ms.
    fallback_advance_ms = 60_000 if step_ms <= 0 else step_ms

    sess = requests.Session()

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }

        data = None
        last_err: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                r = sess.get(endpoint, params=params, timeout=20)

                # rate limit / bans
                if r.status_code in (418, 429):
                    # backoff simple
                    wait = max(RETRY_SLEEP, (attempt + 1) * RETRY_SLEEP)
                    if verbose:
                        print(f"[WARN] HTTP {r.status_code} rate-limit, sleeping {wait:.1f}s")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                data = r.json()
                break

            except Exception as e:
                last_err = e
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = max(RETRY_SLEEP, (attempt + 1) * RETRY_SLEEP)
                print(f"[WARN] retry {attempt+1}/{MAX_RETRIES} after error: {e} (sleep {wait:.1f}s)")
                time.sleep(wait)

        if data is None:
            # por si algo rarísimo sucede
            raise RuntimeError(f"Failed to fetch klines: {last_err}")

        if not data:
            break

        out.extend(data)

        # avanzar al siguiente bloque
        last_open_time = int(data[-1][0])

        # Guard anti-loop: si no avanza, cortamos para no colgarnos.
        if last_open_time < cur:
            if verbose:
                print(f"[WARN] last_open_time went backwards ({last_open_time} < {cur}), stopping pagination.")
            break

        # Avance:
        # - si tenemos step_ms válido, sumamos step_ms
        # - si no, sumamos 1ms para salir del mismo candle
        if step_ms > 0:
            nxt = last_open_time + step_ms
        else:
            nxt = last_open_time + 1

        if nxt <= cur:
            if verbose:
                print(f"[WARN] pagination stalled (cur={cur}, nxt={nxt}), stopping.")
            break

        cur = nxt

        if sleep_s > 0:
            time.sleep(float(sleep_s))

    return out


# ============================================================
# CSV writer
# ============================================================
def write_csv(path: str, rows: List[list]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])

        for r in rows:
            w.writerow([
                int(r[0]),          # open time (ms)
                float(r[1]),        # open
                float(r[2]),        # high
                float(r[3]),        # low
                float(r[4]),        # close
                float(r[5]),        # volume
            ])


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("Binance Kline Downloader → CSV (mensual, reanudable)")
    ap.add_argument("symbol", help="Ej: SOLUSDT")
    ap.add_argument("interval", help="Ej: 1m")
    ap.add_argument("--out", default="datasets", help="Base output dir (default: datasets)")
    ap.add_argument("--endpoint", choices=["spot", "futures"], default="spot", help="spot|futures (default: spot)")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Sleep entre requests (default: 0.25)")
    ap.add_argument("--verbose", action="store_true", help="Logs extra")

    # Compat viejo (no lo rompemos)
    ap.add_argument("--months", nargs="+", help="Ej: 2023-01 2023-02")

    # Nuevo modo
    ap.add_argument("--from", dest="ym_from", default=None, help="YYYY-MM (ej: 2020-01)")
    ap.add_argument("--to", dest="ym_to", default=None, help="YYYY-MM (ej: 2025-12)")

    args = ap.parse_args()

    symbol = args.symbol.upper().strip()
    interval = args.interval.strip()
    base_out = args.out

    endpoint = SPOT_ENDPOINT if args.endpoint == "spot" else FUTURES_ENDPOINT

    # Resolver lista de meses
    months: List[str] = []
    if args.months:
        months = list(args.months)
    else:
        if not args.ym_from or not args.ym_to:
            raise SystemExit("Provide --months ... OR (--from YYYY-MM AND --to YYYY-MM).")
        months = _month_iter(args.ym_from, args.ym_to)

    # Descargar
    for ym in months:
        start_ms, end_ms = _ym_to_month_range(ym)

        out_path = os.path.join(
            base_out,
            symbol,
            interval,
            f"{symbol}_{interval}_{ym}.csv",
        )

        if os.path.exists(out_path):
            print(f"[SKIP] {ym} exists -> {out_path}")
            continue

        print(f"[FETCH] {symbol} {interval} {ym} ({args.endpoint})")
        rows = fetch_klines(
            endpoint=endpoint,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            sleep_s=float(args.sleep),
            verbose=bool(args.verbose),
        )
        print(f"[FETCH] {ym} rows={len(rows)}")

        if not rows:
            print(f"[WARN] no data for {ym}")
            continue

        write_csv(out_path, rows)
        print(f"[OK] wrote {out_path}")

    print("[DONE] download finished.")


if __name__ == "__main__":
    main()
