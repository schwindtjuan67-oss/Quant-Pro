# Live/telegram_monitor.py

import requests
import time
import threading

class TelegramMonitor:
    def __init__(self, token: str, chat_id: str, interval_sec: int = 300):
        self.token = token
        self.chat_id = chat_id
        self.interval_sec = interval_sec

        self._last_status = None
        self._status_callback = None
        self._running = False

    def set_status_callback(self, func):
        """
        func debe ser una función que devuelva un string con el estado actual del bot.
        """
        self._status_callback = func

    def start(self):
        if self._status_callback is None:
            print("[TELEGRAM] Error: No status_callback assigned.")
            return

        self._running = True
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        print(f"[TELEGRAM] Monitor iniciado (cada {self.interval_sec} segundos).")

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            if self._status_callback is not None:
                msg = self._status_callback()
                if msg != self._last_status:
                    self.send_message(msg)
                    self._last_status = msg
            time.sleep(self.interval_sec)

    def send_message(self, text: str):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": text}
        try:
            requests.post(url, data=data)
        except Exception as e:
            print(f"[TELEGRAM] Error enviando mensaje: {e}")
