import os, glob, re
import pandas as pd

DATA_DIR = r"datasets\SOLUSDT\1m"
WINDOW  = "2020-09_2020-12"

def parse_window(w):
    a,b = w.split("_")
    y1,m1 = map(int,a.split("-"))
    y2,m2 = map(int,b.split("-"))
    start = pd.Timestamp(year=y1, month=m1, day=1)
    end   = (pd.Timestamp(year=y2, month=m2, day=1) + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1))
    return start, end

def parse_ts(s: pd.Series) -> pd.Series:
    # numeric epoch?
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        v = s.dropna().iloc[0]
        unit = "ms" if v > 10_000_000_000 else "s"
        return pd.to_datetime(s, unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce")

start, end = parse_window(WINDOW)

rx = re.compile(r".*_(\d{4})-(\d{2})\.csv$", re.I)
paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

keep = []
for p in paths:
    m = rx.match(os.path.basename(p))
    if not m:
        continue
    y,mo = int(m.group(1)), int(m.group(2))
    month_start = pd.Timestamp(year=y, month=mo, day=1)
    month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    if month_end > start and month_start < end:
        keep.append(p)

print(f"[INFO] WINDOW={WINDOW} start={start} end={end}")
print(f"[INFO] total csv={len(paths)} keep={len(keep)}")
print("[INFO] keep:", keep)

dfs = []
for p in keep:
    df = pd.read_csv(p)
    df["timestamp"] = parse_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
wdf = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]

print(f"[INFO] concat rows={len(df)} first={df.timestamp.iloc[0]} last={df.timestamp.iloc[-1]}")
print(f"[INFO] window rows={len(wdf)} first={wdf.timestamp.iloc[0] if len(wdf) else None} last={wdf.timestamp.iloc[-1] if len(wdf) else None}")
