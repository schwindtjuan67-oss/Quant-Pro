@echo off
setlocal ENABLEDELAYEDEXPANSION

REM =====================================================
REM HARD FAIL SAFE – detectar contexto inválido
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

if not exist "%ROOT%\analysis\__init__.py" (
  type nul > "%ROOT%\analysis\__init__.py"
)

REM =====================================================
REM CONFIG GENERAL
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
set "SUPERVISOR_DELAY_SEC=10"

REM =====================================================
REM INIT DIRS
REM =====================================================
for %%D in ("%LOG_DIR%" "%ROBUST_OUT_DIR%" "%ROBUST_LEGACY_DIR%" "%PROMO_DIR%" "%STAGEC_TRADES_DIR%" "%FROZEN_DIR%") do (
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
REM PROMOTER A → B  (FIX REAL)
REM =========================

  --rules configs/promotion_rules_A.json >>"%LOG_FILE%" 2>&1

REM =========================
REM STAGE B (CRITERIO DURO)
REM =========================
%PYTHON% -m analysis.stage_b_risk_calibration ^
  --samples %B_SAMPLES% ^
  --seed %B_SEED% ^
  --min-trades-holdout 200 ^
  --min-pf-holdout 1.25 ^
  --min-winrate-holdout 0.38 ^
  --max-dd-r-holdout 10 ^
  --min-expectancy-holdout 0.02 ^
  --exp-ratio-min 0.75 ^
  --pf-ratio-min 0.9 ^
  --dd-ratio-max 1.2 >>"%LOG_FILE%" 2>&1

REM =========================
REM STAGE C (LETAL / REAL)
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

%PYTHON% -m analysis.stage_c_pipeline_eval >>"%LOG_FILE%" 2>&1

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

