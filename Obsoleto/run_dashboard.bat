@echo off
cd /d "%~dp0"
set PYTHONUTF8=1

REM Activar venv si existe
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

streamlit run analysis\dashboard_v2.py
pause
