import json
import requests
import os

BOT_TOKEN = "TU_TOKEN"
CHAT_ID = "TU_CHAT_ID"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGS = os.path.join(ROOT, "logs", "SOLUSDT_hourly_regime_flags.json")


def send(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": msg})


def notify():
    with open(FLAGS, "r", encoding="utf-8") as f:
        data = json.load(f)

    for rg, block in data["regimes"].items():
        if block["kill_hours"]:
            send(f"🔴 {rg} KILLED hours: {block['kill_hours']}")
        if block["risk_hours"]:
            send(f"🟡 {rg} RISK hours: {block['risk_hours']}")
        if block["promote_hours"]:
            send(f"🟢 {rg} PROMOTE hours: {block['promote_hours']}")


if __name__ == "__main__":
    notify()
