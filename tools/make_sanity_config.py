import argparse, json
from pathlib import Path

TRUE_KEYS = {"enabled","enable","is_enabled","active","use","use_module","use_filter"}
FALSE_KEYS = {"disabled","disable","is_disabled","use_regime_filter","regime_filter","filter_regime","use_spread_filter","use_volume_filter"}

def loosen_key(k: str, v):
    kl = k.lower()

    # bool toggles
    if isinstance(v, bool):
        if kl in FALSE_KEYS or kl.startswith("disable") or kl.endswith("_disabled"):
            return False
        if kl in TRUE_KEYS or kl.startswith("enable") or kl.endswith("_enabled"):
            return True
        return v

    # numeric loosening
    if isinstance(v, (int, float)):
        if kl.startswith("min_") or kl.endswith("_min"):
            return 0
        if kl.startswith("max_") or kl.endswith("_max"):
            return 10**12
        if "cooldown" in kl:
            return 0
        if "fee" in kl or "slippage" in kl:
            return 0
        if "spread" in kl:
            # if it's a max spread, make it huge; if min spread, make it 0
            return 10**12 if ("max" in kl or kl.startswith("max_") or kl.endswith("_max")) else 0
        if "volume" in kl or "notional" in kl:
            return 0
        return v

    return v

def walk(obj):
    if isinstance(obj, dict):
        out = {}
        for k,v in obj.items():
            nv = walk(v)
            nv2 = loosen_key(k, nv)
            out[k] = nv2
        return out
    if isinstance(obj, list):
        return [walk(x) for x in obj]
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input base-config json")
    ap.add_argument("--out", dest="out", required=True, help="output sanity config json")
    args = ap.parse_args()

    inp = Path(args.inp)
    data = json.loads(inp.read_text(encoding="utf-8"))

    sanity = walk(data)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote sanity config -> {args.out}")

if __name__ == "__main__":
    main()
