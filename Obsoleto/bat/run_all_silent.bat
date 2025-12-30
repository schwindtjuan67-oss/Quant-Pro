@echo off
setlocal

cd /d "%~dp0"

call kill_ports.bat

call ".venv\Scripts\activate.bat"

set CONFIG=configs\shadow_debug.json

REM FastAPI - silencioso
start "" /B ^
uvicorn dashboard_rt.app:app --host 127.0.0.1 --port 8000

REM Streamlit - silencioso
start "" /B ^
streamlit run analysis/dashboard_v2.py --server.port 8501 --server.headless true

REM Abrir browser
timeout /t 3 >nul
start "" http://localhost:8501

REM Scalper visible
start "Shadow Runner" cmd /k ^
python run.py --mode shadow --config "%CONFIG%"

endlocal
