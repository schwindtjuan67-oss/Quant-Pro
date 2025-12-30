import os
import sys
import time
import json
import requests
from dotenv import load_dotenv

# Asegurar ROOT en sys.path
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Live.dashboard_live import format_panel, load_state, STATE_PATH


CONFIG_PATH = os.path.join(BASE, "telegram_config.json")

# Cargar .env del root del proyecto si existe
load_dotenv(os.path.join(ROOT, ".env"))


def load_config(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def send_message(base_url: str, chat_id: int, text: str):
    url = f"{base_url}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=data, timeout=5)
        try:
            print(f"[TLC] send_message -> status={r.status_code} ok={r.ok} resp={r.text[:200]}")
        except Exception:
            print(f"[TLC] send_message -> status={r.status_code} ok={r.ok}")

        # Si Telegram devuelve error de parseo de entities, reintentar sin parse_mode
        try:
            js = r.json()
        except Exception:
            js = {}

        desc = js.get("description", "") if isinstance(js, dict) else ""
        if r.status_code == 400 and "can't parse entities" in desc.lower():
            print("[TLC] Markdown parse error detected, retrying without parse_mode...")
            data2 = {"chat_id": chat_id, "text": text}
            try:
                r2 = requests.post(url, json=data2, timeout=5)
                try:
                    print(f"[TLC] send_message(retry) -> status={r2.status_code} ok={r2.ok} resp={r2.text[:200]}")
                except Exception:
                    print(f"[TLC] send_message(retry) -> status={r2.status_code} ok={r2.ok}")
                return r2.ok
            except Exception:
                import traceback
                print("[TLC] Exception sending retry message:")
                traceback.print_exc()
                return False

        return r.ok
    except Exception:
        import traceback
        print("[TLC] Exception sending message:")
        traceback.print_exc()
        return False


def run():
    cfg = load_config(CONFIG_PATH)
    bot_token = cfg.get("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id_cfg = cfg.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token:
        print("[TLC] Falta bot_token: agrega Live/telegram_config.json o TELEGRAM_BOT_TOKEN en .env")
        return

    base_url = f"https://api.telegram.org/bot{bot_token}"
    allowed_chat = int(chat_id_cfg) if chat_id_cfg else None

    print("[TLC] Telegram Commands listener iniciado (polling).")

    offset = None
    try:
        while True:
            params = {"timeout": 20, "limit": 5}
            if offset:
                params["offset"] = offset

            try:
                r = requests.get(f"{base_url}/getUpdates", params=params, timeout=25)
            except Exception as e:
                print("[TLC] Error getUpdates:", e)
                time.sleep(2)
                continue

            if not r.ok:
                print("[TLC] getUpdates HTTP error:", r.text[:200])
                time.sleep(2)
                continue

            js = r.json()
            results = js.get("result", [])

            for upd in results:
                offset = upd.get("update_id", 0) + 1
                # DEBUG: mostrar resumen del update recibido
                try:
                    print(f"[TLC] update_id={upd.get('update_id')}")
                except Exception:
                    print("[TLC] update: (couldn't print update_id)")

                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    print(f"[TLC] update_id={upd.get('update_id')} - sin 'message' (keys={list(upd.keys())})")
                    continue

                chat = msg.get("chat", {})
                chat_id = chat.get("id")
                text = msg.get("text", "") or ""

                # DEBUG: mostrar chat_id y texto (acotado)
                print(f"[TLC] recv -> chat_id={chat_id} text={repr(text[:200])}")

                if allowed_chat and chat_id != allowed_chat:
                    print(f"[TLC] Ignorado: chat_id {chat_id} != allowed_chat {allowed_chat}")
                    # Ignorar mensajes de chats no configurados
                    continue

                if not text.startswith("/"):
                    print(f"[TLC] Ignorado: mensaje no es comando (no empieza con '/')")
                    continue

                cmd = text.split()[0].lower()

                if cmd == "/help":
                    help_txt = (
                        "Comandos disponibles:\n"
                        "/help - lista comandos\n"
                        "/status - estado rápido del bot/dashboard\n"
                        "/dashboard - envia el panel completo\n"
                    )
                    send_message(base_url, chat_id, help_txt)

                elif cmd == "/status":
                    state = load_state(STATE_PATH)
                    try:
                        panel = format_panel(state)
                    except Exception as e:
                        # No queremos que un fallo en format_panel detenga el listener
                        err = f"Error formateando panel: {e}"
                        print(f"[TLC] {err}")
                        send_message(base_url, chat_id, err)
                        continue
                    # Enviar sólo primeras 4000 chars para evitar límites
                    send_message(base_url, chat_id, panel[:3900])

                elif cmd == "/dashboard":
                    state = load_state(STATE_PATH)
                    panel = format_panel(state)
                    send_message(base_url, chat_id, panel[:3900])

                elif cmd == "/logs":
                    # Devuelve la ruta que usa el dashboard y si existe el archivo
                    try:
                        import Live.dashboard_live as dl
                        path = dl.STATE_PATH
                        exists = os.path.exists(path)
                        msg = f"STATE_PATH={path}\nexists={exists}"
                        if exists:
                            with open(path, 'r', encoding='utf-8') as f:
                                preview = f.read(800)
                            msg += "\npreview:\n" + preview
                    except Exception as e:
                        msg = f"Error leyendo STATE_PATH: {e}"
                    send_message(base_url, chat_id, msg)

                elif cmd == "/start":
                    send_message(base_url, chat_id, "Bot conectado. Usa /help para comandos.")

                else:
                    send_message(base_url, chat_id, "Comando no reconocido. Usa /help.")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[TLC] Interrumpido por usuario.")


if __name__ == "__main__":
    run()
