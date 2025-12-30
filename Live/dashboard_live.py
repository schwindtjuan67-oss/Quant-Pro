# Live/dashboard_live.py

import os
import json
import time
import datetime as dt
from typing import Any, Dict, Optional

import requests


# --------- Rutas de archivos ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "telegram_config.json")
# Ajusta si tu logger escribe en otra ruta:
# El TradeLogger por defecto escribe en <project_root>/logs/state_SYMBOL.json
# Mientras que este archivo está en Live/, así que buscamos primero en project_root/logs,
# y si no existe, en Live/logs (compatibilidad).
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATE_PATH_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "logs", "state_SOLUSDT.json"),
    os.path.join(BASE_DIR, "logs", "state_SOLUSDT.json"),
]

def _find_state_path():
    for p in STATE_PATH_CANDIDATES:
        if os.path.exists(p):
            return p
    # devolver la primera por defecto (donde el logger debería escribir)
    return STATE_PATH_CANDIDATES[0]

STATE_PATH = _find_state_path()


# --------- Helpers de archivo ---------

def load_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[DASH] No se encontró config en {path}")
        return {}
    except Exception as e:
        print(f"[DASH] Error leyendo config {path}: {e}")
        return {}


def load_state(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Aún no hay snapshot
        return None
    except json.JSONDecodeError:
        # Se está escribiendo el archivo justo ahora
        return None
    except Exception as e:
        print(f"[DASH] Error leyendo state: {e}")
        return None


# --------- Formato del panel ---------

def format_panel(state: Optional[Dict[str, Any]]) -> str:
    if not state:
        return (
            "📊 *Hybrid Scalper Dashboard — SOLUSDT*\n\n"
            "_Esperando primera actualización del bot..._\n"
            "Verifica que HybridScalperPRO esté corriendo "
            "y que logger_pro esté escribiendo el state JSON."
        )

    ts = state.get("timestamp")  # ms (ideal) o s
    if ts:
        # Normalizar tipos: aceptar int/float o string (ISO or numeric)
        try:
            if isinstance(ts, str):
                s = ts.strip()
                # si es un entero en string
                if s.isdigit():
                    ts = int(s)
                else:
                    # intentar parseo ISO (ej: 2025-11-26T00:00:00)
                    try:
                        dt_obj = dt.datetime.fromisoformat(s)
                        ts = int(dt_obj.timestamp() * 1000)
                    except Exception:
                        # intentar convertir a float
                        try:
                            ts = int(float(s))
                        except Exception:
                            ts = None

            if ts:
                if ts > 10_000_000_000:  # ms
                    dt_obj = dt.datetime.fromtimestamp(ts / 1000.0)
                else:  # s
                    dt_obj = dt.datetime.fromtimestamp(ts)
                ts_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = "N/A"
        except Exception:
            ts_str = "N/A"
    else:
        ts_str = "N/A"

    pos = state.get("position", {}) or {}
    risk = state.get("risk", {}) or {}
    delta = state.get("delta", {}) or {}
    filters = state.get("filters", {}) or {}
    meta = state.get("meta", {}) or {}

    side = pos.get("side") or "FLAT"
    qty = pos.get("qty") or 0.0
    entry_price = pos.get("entry_price")
    cur_sl = pos.get("current_sl")
    best_price = pos.get("best_price")

    equity = risk.get("equity")
    trades_today = risk.get("trades_today")
    max_loss_pct = risk.get("max_loss_pct")
    max_dd_pct = risk.get("max_dd_pct")

    d15 = delta.get("delta_rolling_15s")
    d60 = delta.get("delta_rolling_60s")
    dc = delta.get("delta_candle")
    trades_win = delta.get("trades_count_window")

    trend_long = filters.get("trend_long_ok")
    trend_short = filters.get("trend_short_ok")
    vwap_long = filters.get("vwap_long_ok")
    vwap_short = filters.get("vwap_short_ok")

    status = meta.get("status", "RUNNING")

    # ---- Construir texto ----
    lines = []
    lines.append("📊 *Hybrid Scalper Dashboard — SOLUSDT*")
    lines.append("")
    lines.append(f"`Estado Bot:` {status}")
    lines.append(f"`Último update:` {ts_str}")
    lines.append("")

    # Posición
    lines.append("🧱 *Posición actual*")
    lines.append(f"`Side:` {side}")
    lines.append(f"`Qty:` {qty:.4f}")

    if entry_price is not None:
        lines.append(f"`Entry:` {entry_price:.4f}")
    if cur_sl is not None:
        lines.append(f"`SL:` {cur_sl:.4f}")
    if best_price is not None:
        lines.append(f"`Best:` {best_price:.4f}")

    lines.append("")

    # Risk
    lines.append("🛡 *Risk Manager*")
    if equity is not None:
        lines.append(f"`Equity:` {equity:.4f} USDT")
    if trades_today is not None:
        lines.append(f"`Trades hoy:` {trades_today}")
    if max_loss_pct is not None:
        lines.append(f"`Max Loss:` {max_loss_pct*100:.2f}%")
    if max_dd_pct is not None:
        lines.append(f"`Max DD:` {max_dd_pct*100:.2f}%")

    lines.append("")

    # Delta
    lines.append("🔥 *Delta Flow*")
    if d15 is not None or d60 is not None or dc is not None:
        lines.append(f"`Δ15s:` {d15}  |  `Δ60s:` {d60}")
        lines.append(f"`ΔCandle:` {dc}")
    if trades_win is not None:
        lines.append(f"`Trades ventana:` {trades_win}")
    lines.append("")

    # Filtros
    lines.append("🧠 *Filtros Trend / VWAP*")
    lines.append(
        f"`Trend:` L={trend_long}  S={trend_short}\n"
        f"`VWAP:`  L={vwap_long}  S={vwap_short}"
    )

    return "\n".join(lines)


# --------- Telegram helpers ---------

def send_initial_message(base_url: str, chat_id: int, text: str) -> Optional[int]:
    url = f"{base_url}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=data, timeout=5)
        if not r.ok:
            print("[DASH] Error inicial Telegram:", r.text)
            return None
        js = r.json()
        return js["result"]["message_id"]
    except Exception as e:
        print("[DASH] Excepción send_initial_message:", e)
        return None


def edit_panel_message(base_url: str, chat_id: int, message_id: int, text: str) -> None:
    url = f"{base_url}/editMessageText"
    data = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=data, timeout=5)
        if not r.ok:
            print("[DASH] Error editMessageText:", r.text)
    except Exception as e:
        print("[DASH] Excepción edit_panel_message:", e)


# --------- main ---------

def main() -> None:
    cfg = load_config(CONFIG_PATH)
    # Permitimos fallback a variables de entorno (.env)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

    bot_token = cfg.get("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = cfg.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("[DASH] Falta bot_token o chat_id en telegram_config.json o en .env")
        print("Crea Live/telegram_config.json con:")
        print('{"bot_token": "123:ABC...", "chat_id": 123456789}')
        return

    chat_id = int(chat_id)
    base_url = f"https://api.telegram.org/bot{bot_token}"

    print("[DASH] Iniciando Telegram Dashboard...")
    print(f"[DASH] State path: {STATE_PATH}")

    # Mensaje inicial
    initial_text = format_panel(load_state(STATE_PATH))
    msg_id = send_initial_message(base_url, chat_id, initial_text)
    if msg_id is None:
        print("[DASH] No se pudo obtener message_id, abortando.")
        return

    print(f"[DASH] Dashboard live creado. message_id={msg_id}")

    # Loop de actualización
    while True:
        state = load_state(STATE_PATH)
        text = format_panel(state)
        edit_panel_message(base_url, chat_id, msg_id, text)
        time.sleep(3)  # cada 3 segundos


if __name__ == "__main__":
    main()
