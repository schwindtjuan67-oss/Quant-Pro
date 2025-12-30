# tools/telegram_alerts.py
import requests
import json

TOKEN = "TU_BOT_TOKEN"
CHAT_ID = "TU_CHAT_ID"

def send(msg: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

def send_regime_update(old, new):
    def fmt(rg):
        return ", ".join(str(h) for h in new["regimes"][rg]["kill_hours"]) or "—"

    msg = (
        "🚨 *REGIME UPDATE*\n\n"
        f"TREND kill hours: {fmt('TREND')}\n"
        f"RANGE kill hours: {fmt('RANGE')}"
    )
    send(msg)
