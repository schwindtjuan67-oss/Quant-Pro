import json
import glob
from collections import Counter

files = sorted(glob.glob("results/robust/robust_*_seed*.json"))
if not files:
    print("No robust files found")
    exit(1)

path = files[-1]
print("FILE:", path)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

passed = sum(1 for r in data if r.get("passed") is True)
agg_empty = sum(
    1 for r in data
    if isinstance(r.get("agg"), dict) and len(r.get("agg")) == 0
)

reasons = Counter(
    r.get("fail_reason", "OK")
    for r in data
    if not r.get("passed")
)

print(f"N={len(data)} passed={passed} agg_empty={agg_empty}")
print("Top fail_reason:")
for k, v in reasons.most_common(10):
    print(v, k[:120])
