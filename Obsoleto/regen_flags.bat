@echo off
cd /d "%~dp0"
set PYTHONUTF8=1

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

python analysis\generate_hourly_flags_v1.py --symbol SOLUSDT
pause
