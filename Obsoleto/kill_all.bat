@echo off
echo Matando procesos Auror...
taskkill /F /IM uvicorn.exe >nul 2>&1
taskkill /F /IM python.exe  >nul 2>&1
taskkill /F /IM node.exe    >nul 2>&1
taskkill /F /IM npm.exe     >nul 2>&1
taskkill /F /IM tauri.exe   >nul 2>&1
echo Listo.
pause
