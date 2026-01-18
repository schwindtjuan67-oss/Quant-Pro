@echo off
setlocal ENABLEDELAYEDEXPANSION

REM =====================================================
REM SMOKE TEST — ROBUST ONLY
REM Objetivo:
REM  - 1 seed
REM  - 1 window
REM  - SOLO ROBUST
REM  - verificar que exista al menos 1 passed=true
REM =====================================================

REM ---------------------------
REM ROOT SAFE
REM ---------------------------
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR%"=="" (
  echo [FATAL] SCRIPT_DIR unresolved
  pause
  exit /b 1
)
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%\.." || (
  echo [FATAL] Cannot cd to repo root
  pause
  exit /b 1
)
set "ROOT=%CD%"

REM ---------------------------
REM PYTHON
REM ---------------------------
set "PYTHON=python"
set "PYTHONPATH=%ROOT%"
set "PYTHONIOENCODING=utf-8"

REM ---------------------------
REM CONFIG
REM ---------------------------
set "DATA=datasets\SOLUSDT\1m"
set "BASE_CFG=configs\pipeline_research_backtest.json"

set "ROBUST_OUT_DIR=results\robust"
set "LOG_DIR=logs"
set "LOG_FILE=%LOG_DIR%\smoketest.log"

REM ---------------------------
REM SMOKE PARAMETERS (FIXOS)
REM ---------------------------
set "WINDOW=2022-01_2023-06"
set "SEED=5946"

set "A_WORKERS=3"
set "A_BATCH_SIZE=4"
set "A_SAMPLES=300"

REM ---------------------------
REM INIT DIRS
REM ---------------------------
for %%D in ("%LOG_DIR%" "%ROBUST_OUT_DIR%") do (
  if not exist %%D mkdir %%D
)

REM ---------------------------
REM LOG BOOT
REM ---------------------------
echo =====================================================
echo [SMOKE TEST] %DATE% %TIME%
echo ROOT=%ROOT%
echo DATA=%DATA%
echo WINDOW=%WINDOW%
echo SEED=%SEED%
echo =====================================================

>>"%LOG_FILE%" echo =====================================================
>>"%LOG_FILE%" echo [SMOKE TEST] %DATE% %TIME%
>>"%LOG_FILE%" echo ROOT=%ROOT%
>>"%LOG_FILE%" echo DATA=%DATA%
>>"%LOG_FILE%" echo WINDOW=%WINDOW%
>>"%LOG_FILE%" echo SEED=%SEED%
>>"%LOG_FILE%" echo =====================================================

REM ---------------------------
REM RUN ROBUST (ONLY)
REM ---------------------------
set "OUT=%ROBUST_OUT_DIR%\robust_%WINDOW%_seed%SEED%.json"

echo [SMOKE TEST] Running ROBUST...
>>"%LOG_FILE%" echo [SMOKE TEST] Running ROBUST...

%PYTHON% -m analysis.robust_optimizer ^
  --data "%DATA%" ^
  --base-config "%BASE_CFG%" ^
  --window %WINDOW% ^
  --samples %A_SAMPLES% ^
  --seed %SEED% ^
  --workers %A_WORKERS% ^
  --batch-size %A_BATCH_SIZE% ^
  --parallel ^
  --out "%OUT%" >>"%LOG_FILE%" 2>&1

echo [SMOKE TEST] ROBUST finished
>>"%LOG_FILE%" echo [SMOKE TEST] ROBUST finished

echo.
echo =====================================================
echo [SMOKE TEST] Output file:
echo %OUT%
echo =====================================================
echo.
echo Abrilo y verificá:
echo   - exista al menos 1 "passed": true
echo   - "agg" NO vacio
echo.
pause
exit /b 0
