import os, glob, pandas as pd

DATA_DIR = r"datasets\SOLUSDT\1m"
WINDOW  = "2020-09_2020-12"   # cambia acá

def parse_window(w):
    a,b = w.split("_")
    y1,m1 = map(int,a.split("-"))
    y2,m2 = map(int,b.split("-"))
    start = pd.Timestamp(year=y1, month=m1, day=1)
    end   = (pd.Timestamp(year=y2, month=m2, day=1) + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1))
    return start, end

start, end = parse_window(WINDOW)

files = []
for ext in ("*.parquet","*.csv","*.feather","*.pkl"):
    files += glob.glob(os.path.join(DATA_DIR, ext))

print(f"[INFO] DATA_DIR={DATA_DIR}")
print(f"[INFO] WINDOW={WINDOW} start={start} end(exclusive)={end}")
print(f"[INFO] files={len(files)} -> {files[:5]}{' ...' if len(files)>5 else ''}")

if not files:
    raise SystemExit("[FATAL] No data files found (csv/parquet/feather/pkl).")

fp = files[0]
print(f"[INFO] sample file={fp}")

if fp.endswith(".parquet"):
    df = pd.read_parquet(fp)
elif fp.endswith(".csv"):
    df = pd.read_csv(fp)
elif fp.endswith(".feather"):
    df = pd.read_feather(fp)
elif fp.endswith(".pkl"):
    df = pd.read_pickle(fp)
else:
    raise SystemExit("[FATAL] Unknown format.")

print("[INFO] columns:", list(df.columns)[:30])

cands = ["ts","timestamp","time","open_time","datetime","date","openTime","open_time_ms"]
tcol = None
for c in cands:
    if c in df.columns:
        tcol = c
        break
if tcol is None:
    tcol = df.columns[0]
print(f"[INFO] using time column: {tcol}")

s = df[tcol]
if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
    v = s.dropna().iloc[0]
    unit = "ms" if v > 10_000_000_000 else "s"
    ts = pd.to_datetime(s, unit=unit, errors="coerce")
else:
    ts = pd.to_datetime(s, errors="coerce")

df["_ts"] = ts
df = df.dropna(subset=["_ts"]).sort_values("_ts")

print(f"[INFO] total rows={len(df)} first={df._ts.iloc[0]} last={df._ts.iloc[-1]}")

wdf = df[(df["_ts"] >= start) & (df["_ts"] < end)]
print(f"[INFO] window rows={len(wdf)} first={wdf._ts.iloc[0] if len(wdf) else None} last={wdf._ts.iloc[-1] if len(wdf) else None}")

if len(wdf) > 3:
    d = wdf["_ts"].diff().dropna()
    gaps = d[d > pd.Timedelta(minutes=2)]
    print(f"[INFO] gaps>2m count={len(gaps)} max_gap={gaps.max() if len(gaps) else None}")
