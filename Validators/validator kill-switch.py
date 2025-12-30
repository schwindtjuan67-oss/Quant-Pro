from analysis.kill_switch import RollingDDKillSwitch
import csv

def load_trades(p):
    out = []
    with open(p) as f:
        for r in csv.DictReader(f):
            out.append({"pnl_r": float(r["pnl_r"])})
    return out

trades = load_trades("logs/trades_sample.csv")

kill = RollingDDKillSwitch(window_trades=20, dd_limit_r=6.0)
status = kill.evaluate(trades)

print(status)
