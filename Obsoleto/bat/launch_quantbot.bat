@echo off
title QuantBot Launcher PRO
color 0A

echo ============================================
echo     🚀 QuantBot Launcher - Nivel PRO
echo ============================================
echo.

:: Detectar venv
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] No se encontro el entorno virtual .venv
    pause
    exit /b
)

echo [OK] Activando entorno virtual...
call .venv\Scripts\activate.bat

echo [OK] Preparando logs...
if not exist logs mkdir logs
if not exist state mkdir state

echo Iniciando API FastAPI en puerto 8000...
start "API_SERVICE" cmd /k "color 0B && uvicorn api.main:app --reload --port 8000"

echo Iniciando Dashboard React en puerto 5173...
start "FRONTEND" cmd /k "color 0D && cd web_dashboard && npm run dev"

echo Iniciando Bot Shadow...
start "SHADOW_RUNNER" cmd /k "color 0A && python -m Live.shadow_runner"

echo.
echo ============================================
echo   ✔ Todos los procesos fueron iniciados
echo ============================================
echo.

pause
exit
