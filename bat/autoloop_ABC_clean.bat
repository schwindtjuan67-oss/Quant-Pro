@echo off
setlocal ENABLEDELAYEDEXPANSION

REM =====================================================
REM HARD FAIL SAFE
REM =====================================================
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

REM =====================================================
REM PYTHON HARDENING
REM =====================================================
set "PYTHON=python"
set "PYTHONPATH=%ROOT%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "RUN_MODE=PIPELINE"
REM Desactivar SOFT_MAX_TRADES en modo PIPELINE para no recortar risk_mult en research/backtest
set "PIPELINE_DISABLE_SOFT_MAX_TRADES=1"
set "PIPELINE_DISABLE_TRADE_LOG=1"
set "PIPELINE_VERBOSE_DIAGNOSTICS=0"
set "PIPELINE_VERBOSE_HYBRID=0"
set "PIPELINE_DISABLE_GPU=0"

if not exist "%ROOT%\analysis\__init__.py" (
  type nul > "%ROOT%\analysis\__init__.py"
)

REM =====================================================
REM CONFIG GENERAL
REM =====================================================
set "DATA=datasets\SOLUSDT\1m"
set "BASE_CFG=configs\pipeline_research_backtest.json"

set "ROBUST_OUT_DIR=results\robust"
set "PROMO_DIR=results\promotions"
set "STAGEC_TRADES_DIR=results\pipeline_stageC_trades"
set "HEALTH_DIR=results\health"
set "HEALTH_OUT=%HEALTH_DIR%\health_latest.json"
set "LOG_DIR=logs"
set "LOG_FILE=%LOG_DIR%\autoloop_ABC.log"

set "LOCK_FILE=results\AUTOLOOP.lock"
set "LOCK_STALE_MIN=120"
set "STOP_FILE=results\STOP_AUTOLOOP.txt"

set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1

REM ---------------------------
REM A
REM ---------------------------
set "A_WORKERS=2"
set "A_BATCH_SIZE=32"
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
set "WINDOWS=2020-09_2020-12 2021-01_2021-12 2022-01_2023-06"


REM ---------------------------
REM AUTOLOOP
REM ---------------------------
set "SEEDS_PER_CYCLE=3"
set "SEED_STATE_FILE=results\autoloop_seed_base.txt"
set "START_SEED=1000"
set "CYCLE_DELAY_SEC=30"
set "SUPERVISOR_DELAY_SEC=10"

REM =====================================================
REM INIT DIRS
REM =====================================================
for %%D in ("%LOG_DIR%" "%ROBUST_OUT_DIR%" "%PROMO_DIR%" "%STAGEC_TRADES_DIR%" "%HEALTH_DIR%") do (
  if not exist %%D mkdir %%D
)

REM =====================================================
REM LOG BOOT
REM =====================================================
echo =====================================================
echo [BOOT] %DATE% %TIME%
echo ROOT=%ROOT%
echo DATA=%DATA%
echo WINDOWS=%WINDOWS%
echo =====================================================

>>"%LOG_FILE%" echo =====================================================
>>"%LOG_FILE%" echo [BOOT] %DATE% %TIME%
>>"%LOG_FILE%" echo ROOT=%ROOT%
>>"%LOG_FILE%" echo DATA=%DATA%
>>"%LOG_FILE%" echo WINDOWS=%WINDOWS%
>>"%LOG_FILE%" echo =====================================================

REM =====================================================
REM LOCK
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
timeout /t %SUPERVISOR_DELAY_SEC% /nobreak >nul
goto SUPERVISOR

REM =====================================================
REM ONE CYCLE
REM =====================================================
:ONE_CYCLE
set /p SEED_BASE=<"%SEED_STATE_FILE%"
if "%SEED_BASE%"=="" set "SEED_BASE=%START_SEED%"

set /a S1=SEED_BASE
set /a S2=SEED_BASE+1
set /a S3=SEED_BASE+2
set "SEEDS=%S1% %S2% %S3%"

set /a NEXT_SEED_BASE=SEED_BASE+%SEEDS_PER_CYCLE%
echo %NEXT_SEED_BASE%>"%SEED_STATE_FILE%"

echo [CYCLE] seed_base=%SEED_BASE% seeds=%SEEDS%
>>"%LOG_FILE%" echo [CYCLE] seed_base=%SEED_BASE% seeds=%SEEDS%

REM =========================
REM FASE A Ã¢â‚¬â€ ROBUST
REM =========================
set "PIPELINE_PHASE=A"
set "PIPELINE_DISABLE_GPU=0"

for %%W in (%WINDOWS%) do (
  for %%S in (%SEEDS%) do (
    set "OUT=%ROBUST_OUT_DIR%\robust_%%W_seed%%S.json"
    if not exist "!OUT!" (
      set "PROGRESS=[ROBUST] window=%%W seed=%%S out=!OUT!"
      echo !PROGRESS!
      >>"%LOG_FILE%" echo !PROGRESS!
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

set "PIPELINE_DISABLE_GPU=0"

REM =========================
REM POST A Ã¢â‚¬â€ ÃƒÅ¡NICO PROMOTOR AÃ¢â€ â€™B
REM =========================
%PYTHON% -m analysis.analysis_post_robust >>"%LOG_FILE%" 2>&1

REM =========================
REM === HEALTH (POST-A) ===
REM =========================
echo [HEALTH][POST-A] running health check
>>"%LOG_FILE%" echo [HEALTH][POST-A] running health check

%PYTHON% -m analysis.pipeline_health ^
  --root "%ROOT%" ^
  --out "%HEALTH_OUT%" ^
  --stop-file "%STOP_FILE%" ^
  --stop-on-contract-fail ^
  >>"%LOG_FILE%" 2>&1

set "HC_RC=%ERRORLEVEL%"
echo [HEALTH][POST-A] rc=%HC_RC%
>>"%LOG_FILE%" echo [HEALTH][POST-A] rc=%HC_RC%

if %HC_RC%==1 (
  echo [HEALTH][POST-A] ABORTING (health execution error rc=1)
  >>"%LOG_FILE%" echo [HEALTH][POST-A] ABORTING (health execution error rc=1)
  echo HEALTH_EXEC_ERROR rc=1 %DATE% %TIME% > "%STOP_FILE%"
  exit /b 1
)

if %HC_RC% GEQ 2 (
  echo [HEALTH][POST-A] ABORTING (health gate failed rc=%HC_RC%)
  >>"%LOG_FILE%" echo [HEALTH][POST-A] ABORTING (health gate failed rc=%HC_RC%)
  exit /b %HC_RC%
)

REM =========================
REM STAGE B Ã¢â‚¬â€ RISK CALIBRATION
REM =========================
%PYTHON% -m analysis.stage_b_risk_calibration ^
  --data "%DATA%" ^
  --base-config "%BASE_CFG%" ^
  --fasea "%PROMO_DIR%\faseA_promoted.json" ^
  --out "%PROMO_DIR%\faseB_promoted.json" ^
  --report-csv "%PROMO_DIR%\faseB_report.csv" ^
  --samples %B_SAMPLES% ^
  --seed %B_SEED% >>"%LOG_FILE%" 2>&1

REM =========================
REM === HEALTH (POST-B) ===
REM =========================
echo [HEALTH][POST-B] running health check
>>"%LOG_FILE%" echo [HEALTH][POST-B] running health check
%PYTHON% -m analysis.pipeline_health ^
  --root "%ROOT%" ^
  --out "%HEALTH_OUT%" ^
  --stop-file "%STOP_FILE%" ^
  --stop-on-contract-fail ^
  >>"%LOG_FILE%" 2>&1

set "HC_RC=%ERRORLEVEL%"
echo [HEALTH][POST-B] rc=%HC_RC%
>>"%LOG_FILE%" echo [HEALTH][POST-B] rc=%HC_RC%

if %HC_RC%==1 (
  echo [HEALTH][POST-B] ABORTING (health execution error rc=1)
  >>"%LOG_FILE%" echo [HEALTH][POST-B] ABORTING (health execution error rc=1)
  echo HEALTH_EXEC_ERROR rc=1 %DATE% %TIME% > "%STOP_FILE%"
  exit /b 1
)

if %HC_RC% GEQ 2 (
  echo [HEALTH][POST-B] ABORTING (health gate failed rc=%HC_RC%)
  >>"%LOG_FILE%" echo [HEALTH][POST-B] ABORTING (health gate failed rc=%HC_RC%)
  exit /b %HC_RC%
)

REM =========================
REM STAGE C Ã¢â‚¬â€ SUPERVIVENCIA REAL
REM =========================
set PIPELINE_MIN_TRADES=400
set PIPELINE_MIN_R_OBS=300
set PIPELINE_TH_EXPECTANCY=0.08
set PIPELINE_TH_SORTINO=1.7
set PIPELINE_TH_PF=1.45
set PIPELINE_TH_DD=-0.15
set PIPELINE_TH_WINRATE=0.45
set PIPELINE_TH_WORST5=-1.2
set STAGEC_MIN_WINDOWS_OK=2
set STAGEC_REQUIRE_GLOBAL_PASS=1

%PYTHON% -m analysis.stage_c_pipeline_eval ^
  --faseb "%PROMO_DIR%\faseB_promoted.json" ^
  --data "%DATA%" ^
  --base-config "%BASE_CFG%" ^
  --trades-dir "%STAGEC_TRADES_DIR%" ^
  --interval %C_INTERVAL% ^
  --warmup %C_WARMUP% >>"%LOG_FILE%" 2>&1

REM =========================
REM === HEALTH (POST-C) ===
REM =========================
echo [HEALTH][POST-C] running health check
>>"%LOG_FILE%" echo [HEALTH][POST-C] running health check
%PYTHON% -m analysis.pipeline_health ^
  --root "%ROOT%" ^
  --out "%HEALTH_OUT%" ^
  --stop-file "%STOP_FILE%" ^
  --stop-on-contract-fail ^
  >>"%LOG_FILE%" 2>&1

set "HC_RC=%ERRORLEVEL%"
echo [HEALTH][POST-C] rc=%HC_RC%
>>"%LOG_FILE%" echo [HEALTH][POST-C] rc=%HC_RC%

if %HC_RC%==1 (
  echo [HEALTH][POST-C] ABORTING (health execution error rc=1)
  >>"%LOG_FILE%" echo [HEALTH][POST-C] ABORTING (health execution error rc=1)
  echo HEALTH_EXEC_ERROR rc=1 %DATE% %TIME% > "%STOP_FILE%"
  exit /b 1
)

if %HC_RC% GEQ 2 (
  echo [HEALTH][POST-C] ABORTING (health gate failed rc=%HC_RC%)
  >>"%LOG_FILE%" echo [HEALTH][POST-C] ABORTING (health gate failed rc=%HC_RC%)
  exit /b %HC_RC%
)

timeout /t %CYCLE_DELAY_SEC% /nobreak >nul
exit /b 0

REM =====================================================
REM END
REM =====================================================
:END
del "%LOCK_FILE%" >nul 2>&1
echo [END] %DATE% %TIME%
>>"%LOG_FILE%" echo [END] %DATE% %TIME%
pause
