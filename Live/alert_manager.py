import os
import time
import traceback
from typing import Any, Dict, Optional

import requests

from .event_bus import EventBus


class AlertManager:
    """
    Orquestador de alerts vía Telegram conectado al EventBus.

    - Usa variables de entorno por defecto:
        TELEGRAM_BOT_TOKEN
        TELEGRAM_CHAT_ID

      También podés pasar bot_token / chat_id explícitos si querés.

    - Se suscribe a eventos del EventBus:
        * trade_entry
        * trade_exit
        * risk_event
        * error
    """

    def __init__(
        self,
        symbol: str,
        event_bus: EventBus,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        env_prefix: str = "TELEGRAM_",
    ) -> None:
        self.symbol = symbol
        self.event_bus = event_bus
        self.env_prefix = env_prefix

        if bot_token is None:
            bot_token = os.getenv(f"{env_prefix}BOT_TOKEN")
        if chat_id is None:
            chat_id = os.getenv(f"{env_prefix}CHAT_ID")

        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            print(
                "[ALERT] Telegram deshabilitado: faltan "
                f"{env_prefix}BOT_TOKEN o {env_prefix}CHAT_ID en el entorno."
            )
            self.api_url = None
        else:
            self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            print(
                f"[ALERT] Telegram listo → chat_id={self.chat_id}, "
                f"symbol={self.symbol}"
            )

        # Suscripciones al EventBus
        event_bus.subscribe("trade_entry", self.on_trade_entry)
        event_bus.subscribe("trade_exit", self.on_trade_exit)
        event_bus.subscribe("risk_event", self.on_risk_event)
        event_bus.subscribe("error", self.on_error)

    # --------------------------------------------------------------
    #  Low-level sender
    # --------------------------------------------------------------
    def _send(self, text: str, disable_notification: bool = False) -> None:
        if not self.enabled or not self.api_url:
            return

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_notification": disable_notification,
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=5)
            if not resp.ok:
                print(
                    f"[ALERT] Error HTTP {resp.status_code}: {resp.text[:200]}"
                )
        except Exception as e:  # pragma: no cover - solo logging
            print("[ALERT] Excepción enviando mensaje Telegram:", e)
            traceback.print_exc()

    # --------------------------------------------------------------
    #  Handlers de eventos
    # --------------------------------------------------------------
    def on_trade_entry(self, data: Dict[str, Any]) -> None:
        side = data.get("side", "?")
        price = data.get("price")
        qty = data.get("qty")
        sl = data.get("sl")
        atr = data.get("atr")
        vol_target = data.get("vol_target")
        risk = data.get("risk_state") or {}

        trades_today = risk.get("trades_today")
        dd_pct = risk.get("drawdown_pct")

        lines = [
            f"🔔 *ENTRY {self.symbol}*",
            f"Side: *{side}*",
        ]

        if price is not None:
            lines.append(f"Precio: `{price}`")
        if qty is not None:
            lines.append(f"Qty: `{qty}`")
        if sl is not None:
            lines.append(f"SL inicial: `{sl}`")
        if atr is not None:
            lines.append(f"ATR: `{atr}`")
        if vol_target is not None:
            try:
                pct = float(vol_target) * 100.0
                lines.append(f"Riesgo por trade: `{pct:.2f}%`")
            except Exception:
                pass

        meta = []
        if trades_today is not None:
            meta.append(f"trades_día={trades_today}")
        if dd_pct is not None:
            try:
                meta.append(f"DD={dd_pct:.2%}")
            except Exception:
                pass

        if meta:
            lines.append("`" + " | ".join(meta) + "`")

        self._send("\n".join(lines))

    def on_trade_exit(self, data: Dict[str, Any]) -> None:
        side = data.get("side", "?")
        price = data.get("price")
        qty = data.get("qty")
        reason = data.get("reason", "-")
        pnl_abs = data.get("pnl_abs")
        pnl_pct = data.get("pnl_pct")
        risk = data.get("risk_state") or {}

        eq = risk.get("equity")
        trades_today = risk.get("trades_today")
        dd_pct = risk.get("drawdown_pct")

        lines = [
            f"✅ *EXIT {self.symbol}*",
            f"Side: *{side}*  |  Motivo: `{reason}`",
        ]

        if price is not None:
            lines.append(f"Precio cierre: `{price}`")
        if qty is not None:
            lines.append(f"Qty: `{qty}`")
        if pnl_abs is not None:
            lines.append(f"PnL: `{pnl_abs:.4f}`")
        if pnl_pct is not None:
            try:
                lines.append(f"PnL % equity: `{pnl_pct:.2%}`")
            except Exception:
                pass

        meta = []
        if eq is not None:
            meta.append(f"equity≈{eq:.2f}")
        if trades_today is not None:
            meta.append(f"trades_día={trades_today}")
        if dd_pct is not None:
            try:
                meta.append(f"DD={dd_pct:.2%}")
            except Exception:
                pass

        if meta:
            lines.append("`" + " | ".join(meta) + "`")

        self._send("\n".join(lines))

    def on_risk_event(self, data: Dict[str, Any]) -> None:
        reason = data.get("reason", "RISK_EVENT")
        risk = data.get("risk_state") or {}

        eq = risk.get("equity")
        trades_today = risk.get("trades_today")
        dd_pct = risk.get("drawdown_pct")

        lines = [
            f"🛑 *RISK {self.symbol}*",
            f"Motivo: `{reason}`",
        ]

        meta = []
        if eq is not None:
            meta.append(f"equity≈{eq:.2f}")
        if trades_today is not None:
            meta.append(f"trades_día={trades_today}")
        if dd_pct is not None:
            try:
                meta.append(f"DD={dd_pct:.2%}")
            except Exception:
                pass

        if meta:
            lines.append("`" + " | ".join(meta) + "`")

        self._send("\n".join(lines))

    def on_error(self, data: Dict[str, Any]) -> None:
        message = data.get("message", "Error no especificado")
        where = data.get("where", "unknown")
        lines = [
            f"⚠️ *ERROR {self.symbol}*",
            f"Lugar: `{where}`",
            "",
            f"`{message}`",
        ]
        self._send("\n".join(lines), disable_notification=False)
