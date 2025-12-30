@echo off
title QuantBot Shutdown PRO
color 0C

echo ============================================
echo     🛑 QuantBot Shutdown Profesional
echo ============================================
echo.

echo Cerrando consolas...
taskkill /FI "WINDOWTITLE eq API_SERVICE*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq FRONTEND*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq SHADOW_RUNNER*" /T /F >nul 2>&1

echo Matando servicios persistentes...
taskkill /IM python.exe /F >nul 2>&1
taskkill /IM node.exe /F >nul 2>&1
taskkill /IM uvicorn.exe /F >nul 2>&1

echo Limpiando puertos (si quedo algo colgado)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173') do taskkill /PID %%a /F >nul 2>&1

echo.
echo ============================================
echo   ✔ Todos los procesos fueron detenidos
echo ============================================
echo.

pause
exit
