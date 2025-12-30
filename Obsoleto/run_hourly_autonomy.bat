@echo off
cd /d %~dp0

echo [1] Generando flags horarios...
python analysis/hourly_flags_generator.py

echo [2] Enviando alertas Telegram...
python analysis/telegram_alerts.py

echo [OK] Autonomía ejecutada.
pause
