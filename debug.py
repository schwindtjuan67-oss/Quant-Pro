from analysis.robust_optimizer import _real_backtest_fn, load_candles_from_path

candles = load_candles_from_path(
    "datasets/SOLUSDT/1m",
    date_from="2019-01-01",
    date_to="2020-12-31"
)

params = {
  "ema_fast": 20,
  "ema_slow": 54,
  "atr_len": 11,
  "sl_atr_mult": 1.2,
  "tp_atr_mult": 1.68,
  "rr_min": 1.4,
  "delta_threshold": 80,
  "delta_rolling_sec": 60,
  "cooldown_sec": 15,
  "max_trades_day": 8,
  "use_time_filter": True,
  "hour_start": 2,
  "hour_end": 20
}

_real_backtest_fn(
    candles,
    params,
    base_cfg=...,   # el mismo del pipeline
    symbol="SOLUSDT",
    interval="1m",
    warmup=500
)
