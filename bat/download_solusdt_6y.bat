@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =====================================================
REM ROOT DEL PROYECTO
REM =====================================================
cd /d %~dp0\..

REM =====================================================
REM PYTHON / PATH
REM =====================================================
set "PY=python"
set "PYTHONPATH=%cd%"

echo ROOT=%cd%
echo PYTHONPATH=%PYTHONPATH%
echo.

REM =====================================================
REM LOGGING
REM =====================================================
if not exist logs mkdir logs
set "LOG=logs\download_klines.log"

echo ===============================================>>"%LOG%"
echo START DOWNLOAD %DATE% %TIME%>>"%LOG%"
echo ===============================================>>"%LOG%"

REM =====================================================
REM CONFIG
REM =====================================================
set "SYMBOL=SOLUSDT"
set "INTERVAL=1m"
set "FROM=2020-01"
set "TO=2025-12"
set "OUT=datasets"

REM Rate-limit safe (si querés más rápido, bajá a 0.05)
set "SLEEP=0.25"

REM =====================================================
REM INFO
REM =====================================================
echo [BAT] Downloading %SYMBOL% %INTERVAL% from %FROM% to %TO%
echo [BAT] Output dir: %OUT%
echo [BAT] Sleep: %SLEEP%s
echo.

echo [BAT] Downloading %SYMBOL% %INTERVAL% from %FROM% to %TO%>>"%LOG%"
echo [BAT] Output dir: %OUT%>>"%LOG%"
echo [BAT] Sleep: %SLEEP%s>>"%LOG%"

REM =====================================================
REM RUN DOWNLOADER
REM =====================================================
%PY% tools\download_klines.py ^
  %SYMBOL% ^
  %INTERVAL% ^
  --from %FROM% ^
  --to %TO% ^
  --sleep %SLEEP% ^
  --out %OUT% >>"%LOG%" 2>&1

if errorlevel 1 (
  echo.
  echo ===============================================
  echo [BAT][ERROR] Downloader failed
  echo ===============================================
  echo Ver logs\download_klines.log
  echo.
  echo [BAT][ERROR] Downloader failed>>"%LOG%"
  pause
  exit /b 1
)

echo.
echo ===============================================
echo [BAT] Download finished OK
echo ===============================================
echo.

echo [BAT] Download finished OK>>"%LOG%"
echo END DOWNLOAD %DATE% %TIME%>>"%LOG%"
echo ===============================================>>"%LOG%"

pause
endlocal
