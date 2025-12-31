:v

**QuantBot**

**Instalación**

1. Abrí el instalador de dependencias: `install_requirements.bat` — esto crea/activa un `.venv` y corre `pip install -r requirements.txt`.

   En PowerShell:

   `call install_requirements.bat`

2. Después de instalar, arrancá el bot principal:

   `call run_bot.bat`

   - `run_bot.bat` abre dos ventanas:
     - Una ejecuta `Live\hybrid_scalper_pro.py` (el bot principal / estrategia).
     - Otra ejecuta `Live\telegram_commands.py` (listener de comandos por Telegram).

3. Si sólo querés correr el listener de Telegram por separado, usá:

   `call run_telegram_commands.bat`

**Variables y configuración**

- El BOT puede usar variables de entorno en un archivo `.env` en la raíz (se usa `python-dotenv`).
- Opcionalmente, para Telegram podés crear `Live/telegram_config.json` con `{ "bot_token": "...", "chat_id": 12345 }`, o definir `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` en `.env`.
- Atención: en `Live/order_router.py` hay claves de ejemplo/placeholder: reemplazá `api_key` y `api_secret` por las tuyas antes de pasar a live.

**Qué hace cada cosa**

- `Live/hybrid_scalper_pro.py`: estrategia principal (EMA 9/21 + ATR, filtros Trend/VWAP, scoring, trailing SL, maneja entradas/salidas).
- `Live/ws_futures_1m.py`: conecta al WebSocket de Binance Futures y entrega velas 1m al bot.
- `Live/delta_router.py` y `Live/delta_live.py`: calculan un "delta" del flujo de órdenes para filtros institucionales.
- `Live/order_router.py`: wrapper que envía órdenes (usa `Live/binance_client.py`).
- `Live/risk_manager.py`: límites diarios, max trades, drawdown — bloquea operaciones si se superan.
- `Live/logger_pro.py`: genera `logs/bars_*.csv`, `logs/trades_*.csv` y `logs/state_SYMBOL.json` para dashboard/Telegram.
- `Live/alert_manager.py`: conecta eventos del bot al telegram (avisos de entry/exit/risk/errors).
- `Live/telegram_commands.py`: listener por polling que responde a `/help`, `/status`, `/dashboard`, `/logs`.

**Archivos útiles**

- `install_requirements.bat` — instala dependencias y crea/activa `.venv`.
- `run_bot.bat` — lanza la estrategia + listener de Telegram en dos ventanas.
- `run_telegram_commands.bat` — lanza sólo el listener de Telegram.
- `logs/` — aquí se guardan CSVs y snapshots JSON que usa el dashboard Telegram.
 
**Ejemplo `.env`**

Podés crear un archivo `.env` en la raíz con tus credenciales/IDs. Ejemplo 

```
TELEGRAM_BOT_TOKEN=8402398613:HASHASa-1Bbn3RsLFGidQk-1b2asg312ASD
TELEGRAM_CHAT_ID=-12512612612
```

Guardá el archivo como `.env` y luego ejecutá `call install_requirements.bat`.


