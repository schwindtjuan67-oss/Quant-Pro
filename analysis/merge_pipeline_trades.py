import argparse
import glob
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--parts-dir", default="results/pipeline_trades_parts")
    ap.add_argument("--out", default="results/pipeline_trades.csv")
    args = ap.parse_args()

    pattern = os.path.join(args.parts_dir, args.run_id, "**", "pipeline_trades.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise SystemExit(f"No parts found with pattern: {pattern}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source_file"] = f
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] failed reading {f}: {e}")

    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Merged {len(files)} parts -> {args.out} rows={len(out)}")   

if __name__ == "__main__":
    main()

