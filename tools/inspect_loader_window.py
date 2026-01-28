#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone

from analysis.robust_optimizer import load_candles_from_path, window_to_dates, _pick_ts_ms


def _fmt_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect candle loader window selection.")
    ap.add_argument("--data", default="datasets/SOLUSDT/1m", help="Dataset path (dir or file)")
    ap.add_argument("--window", default="2020-09_2020-12", help="Window YYYY-MM_YYYY-MM")
    ap.add_argument("--min-rows", type=int, default=170000, help="Minimum expected rows")
    args = ap.parse_args()

    date_from, date_to = window_to_dates(args.window)
    candles = load_candles_from_path(args.data, date_from=date_from, date_to=date_to, debug=False)
    if not candles:
        raise SystemExit("[INSPECT] No candles loaded.")

    first_ts = _pick_ts_ms(candles[0])
    last_ts = _pick_ts_ms(candles[-1])
    if first_ts is None or last_ts is None:
        raise SystemExit("[INSPECT] Missing timestamps in loaded candles.")

    print(f"[INSPECT] window={args.window} rows={len(candles)}")
    print(f"[INSPECT] first_ts={_fmt_ts(first_ts)} last_ts={_fmt_ts(last_ts)}")

    if len(candles) < args.min_rows:
        raise SystemExit(f"[INSPECT] Expected at least {args.min_rows} rows, got {len(candles)}")

    if args.window == "2020-09_2020-12":
        if _fmt_ts(first_ts) != "2020-09-01 00:00":
            raise SystemExit(f"[INSPECT] Unexpected first timestamp: {_fmt_ts(first_ts)}")
        if _fmt_ts(last_ts) != "2020-12-31 23:59":
            raise SystemExit(f"[INSPECT] Unexpected last timestamp: {_fmt_ts(last_ts)}")


if __name__ == "__main__":
    main()
