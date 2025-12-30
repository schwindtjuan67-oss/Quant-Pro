@echo off
cd /d "%~dp0"

echo Activando entorno virtual...
call ".venv\Scripts\activate.bat"

echo Lanzando Telegram Commands listener...
python "Live\telegram_commands.py"

pause