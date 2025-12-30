@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ==========================================
echo   ShadowBot - DEBUG Full Stack
echo ==========================================

cd /d "%~dp0"

REM ==============================
REM Activar venv
REM ==============================
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo [OK] .venv activado
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    echo [OK] venv activado
) else (
    echo [ERROR] No se encontró venv
    pause
    exit /b 1
)

REM ==============================
REM Config SHADOW
REM ==============================
set CONFIG=configs\shadow_debug.json
echo [INFO] Config SHADOW: %CONFIG%

REM ==============================
REM Dashboard Realtime (FastAPI)
REM ==============================
echo [START] Dashboard RT (FastAPI)...
start "Dashboard RT" cmd /k ^
uvicorn dashboard_rt.app:app --host 127.0.0.1 --port 8000

REM ==============================
REM Dashboard Analisis Profundo (Streamlit)
REM ==============================
echo [START] Streamlit Analisis Profundo...
start "Streamlit Analysis" cmd /k ^
streamlit run analysis/dashboard_v2.py --server.port 8501

REM ==============================
REM Shadow Runner (SCALPER)
REM ==============================
echo [START] Shadow Runner (SCALPER)...
start "Shadow Runner" cmd /k ^
python run.py --mode shadow --config "%CONFIG%"

echo.
echo ==========================================
echo STACK DEBUG LEVANTADO
echo.
echo Shadow Runner : visible
echo Dashboard RT  : http://127.0.0.1:8000
echo Streamlit     : http://localhost:8501
echo ==========================================
echo.

pause
