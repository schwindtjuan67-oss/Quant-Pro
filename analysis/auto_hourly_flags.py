# analysis/auto_hourly_flags.py
import time
import json
import subprocess
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_SCRIPT = os.path.join(ROOT, "analysis", "simulate_hourly_kill.py")
FLAGS_PATH = os.path.join(ROOT, "logs", "SOLUSDT_hourly_regime_flags.json")

REFRESH_HOURS = 3

def load_flags():
    if not os.path.exists(FLAGS_PATH):
        return {}
    with open(FLAGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

print("🧠 Auto Hourly Flags Runner iniciado")

prev_flags = load_flags()

while True:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Recalculando flags horarios…")
    subprocess.call(["python", SIM_SCRIPT])

    new_flags = load_flags()

    if new_flags != prev_flags:
        print("🚨 FLAGS CAMBIARON")
        try:
            from tools.telegram_alerts import send_regime_update
            send_regime_update(prev_flags, new_flags)
        except Exception as e:
            print("Telegram error:", e)

        prev_flags = new_flags

    time.sleep(REFRESH_HOURS * 3600)
