@echo off
setlocal ENABLEDELAYEDEXPANSION

REM =====================================================
REM ROOT
REM =====================================================
cd /d "%~dp0\.."
set "ROOT=%CD%"

REM =====================================================
REM PYTHON HARDENING
REM =====================================================
set "PYTHON=python"
set "PYTHONPATH=%ROOT%"
set "PYTHONIOENCODING=utf-8"

REM asegurar package analysis
if not exist "analysis\__init__.py" (
  type nul > "analysis\__init__.py"
)

REM =====================================================
REM CONFIG GENERAL (INTACTA)
REM =====================================================
set "DATA=datasets\SOLUSDT\1m"
set "BASE_CFG=configs\pipeline_research_backtest.json"

set "ROBUST_OUT_DIR=results\robust"
set "ROBUST_LEGACY_DIR=results\robust_legacy"
set "PROMO_DIR=results\promotions"
set "STAGEC_TRADES_DIR=results\pipeline_stageC_trades"
set "FROZEN_DIR=results\frozen"
set "LOG_DIR=logs"
set "LOG_FILE=%LOG_DIR%\autoloop_ABC.log"

set "HEARTBEAT_FILE=results\autoloop_heartbeat.txt"
set "LOCK_FILE=results\AUTOLOOP.lock"
set "LOCK_STALE_MIN=120"
set "STOP_FILE=results\STOP_AUTOLOOP.txt"

REM ---------------------------
REM A
REM ---------------------------
set "A_WORKERS=3"
set "A_BATCH_SIZE=4"
set "A_SAMPLES=300"

REM ---------------------------
REM B
REM ---------------------------
set "B_SAMPLES=200"
set "B_SEED=1337"

REM ---------------------------
REM C
REM ---------------------------
set "C_WARMUP=500"
set "C_INTERVAL=1m"

REM ---------------------------
REM WINDOWS
REM ---------------------------
set "WINDOWS=2019-01_2020-12 2021-01_2021-12 2022-01_2023-06"

REM ---------------------------
REM AUTOLOOP
REM ---------------------------
set "SEEDS_PER_CYCLE=3"
set "SEED_STATE_FILE=results\autoloop_seed_base.txt"
set "START_SEED=1000"
set "CYCLE_DELAY_SEC=30"
set "KEEP_RUNNING=0"
set "SHORTLIST_N=10"
set "SUPERVISOR_DELAY_SEC=10"
set "ARCHIVE_LEGACY=1"

REM =====================================================
REM INIT DIRS
REM =====================================================
for %%D in ("%LOG_DIR%" "%ROBUST_OUT_DIR%" "%ROBUST_LEGACY_DIR%" "%PROMO_DIR%" "%STAGEC_TRADES_DIR%" "%FROZEN_DIR%") do (
  if not exist %%D mkdir %%D
)

REM =====================================================
REM LOG BOOT
REM =====================================================
>>"%LOG_FILE%" echo =====================================================
>>"%LOG_FILE%" echo [BOOT] %DATE% %TIME%
>>"%LOG_FILE%" echo ROOT=%ROOT%
>>"%LOG_FILE%" echo PYTHONPATH=%PYTHONPATH%
>>"%LOG_FILE%" echo DATA=%DATA%
>>"%LOG_FILE%" echo BASE_CFG=%BASE_CFG%
>>"%LOG_FILE%" echo WINDOWS=%WINDOWS%
>>"%LOG_FILE%" echo =====================================================

REM =====================================================
REM LOCK + STALE RECOVERY (INTACTO)
REM =====================================================
if exist "%LOCK_FILE%" (
  powershell -NoProfile -Command ^
    "$p='%LOCK_FILE%';" ^
    "$age=(New-TimeSpan -Start (Get-Item $p).LastWriteTime -End (Get-Date)).TotalMinutes;" ^
    "if($age -gt %LOCK_STALE_MIN%){ Remove-Item -Force $p; exit 0 } else { exit 1 }"
  if errorlevel 1 goto END
)

echo %DATE% %TIME%>"%LOCK_FILE%"

if not exist "%SEED_STATE_FILE%" echo %START_SEED%>"%SEED_STATE_FILE%"

REM =====================================================
REM SUPERVISOR LOOP
REM =====================================================
:SUPERVISOR
if exist "%STOP_FILE%" goto END

call :ONE_CYCLE
set "RC=%ERRORLEVEL%"

>>"%LOG_FILE%" echo [SUPERVISOR] rc=%RC% %DATE% %TIME%

if "%RC%"=="10" goto END
if "%RC%"=="20" goto END

timeout /t %SUPERVISOR_DELAY_SEC% /nobreak >nul
goto SUPERVISOR

REM =====================================================
REM ONE CYCLE
REM =====================================================
:ONE_CYCLE
echo %DATE% %TIME%>"%LOCK_FILE%"
echo %DATE% %TIME%>"%HEARTBEAT_FILE%"

set /p SEED_BASE=<"%SEED_STATE_FILE%"
if "%SEED_BASE%"=="" set "SEED_BASE=%START_SEED%"

set "SEEDS=%SEED_BASE% %SEED_BASE%+1 %SEED_BASE%+2"
set /a NEXT_SEED_BASE=SEED_BASE+%SEEDS_PER_CYCLE%
echo %NEXT_SEED_BASE%>"%SEED_STATE_FILE%"

>>"%LOG_FILE%" echo [CYCLE] seed_base=%SEED_BASE% seeds=%SEEDS%

REM =========================
REM FASE A
REM =========================
set "PIPELINE_PHASE=A"

for %%W in (%WINDOWS%) do (
  for %%S in (%SEEDS%) do (
    set "OUT=%ROBUST_OUT_DIR%\robust_%%W_seed%%S.json"
    if not exist "!OUT!" (
      %PYTHON% -m analysis.robust_optimizer ^
        --data "%DATA%" ^
        --base-config "%BASE_CFG%" ^
        --window %%W ^
        --samples %A_SAMPLES% ^
        --seed %%S ^
        --workers %A_WORKERS% ^
        --batch-size %A_BATCH_SIZE% ^
        --parallel ^
        --out "!OUT!" >>"%LOG_FILE%" 2>&1
    )
  )
)

REM =========================
REM POST A
REM =========================
%PYTHON% -m analysis.analysis_post_robust >>"%LOG_FILE%" 2>&1

REM =========================
REM PROMOTER A→B
REM =========================
%PYTHON% -m analysis.promoter_faseAto_B >>"%LOG_FILE%" 2>&1

REM =========================
REM STAGE B
REM =========================
%PYTHON% -m analysis.stage_b_risk_calibration >>"%LOG_FILE%" 2>&1

REM =========================
REM STAGE C
REM =========================
%PYTHON% -m analysis.stage_c_pipeline_eval >>"%LOG_FILE%" 2>&1

timeout /t %CYCLE_DELAY_SEC% /nobreak >nul
exit /b 0

REM =====================================================
REM END
REM =====================================================
:END
del "%LOCK_FILE%" >nul 2>&1
>>"%LOG_FILE%" echo [END] %DATE% %TIME%
pause

