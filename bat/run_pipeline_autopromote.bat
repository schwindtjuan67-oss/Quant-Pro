@echo off
setlocal ENABLEDELAYEDEXPANSION

REM =====================================
REM MOVERSE AL ROOT DEL REPO
REM =====================================
cd /d "%~dp0\.."

REM =====================================================
REM CONFIG GENERAL
REM =====================================================

set PYTHON=python

REM --- Scripts ---
set ROBUST_SCRIPT=analysis\robust_optimizer.py
set POST_SCRIPT=analysis\analysis_post_robust.py
set PROMOTER_SCRIPT=analysis\promoter_faseA_to_B.py

REM --- Data / Config ---
set DATA=datasets\SOLUSDT\1m
set BASE_CFG=configs\pipeline_research_backtest.json

REM --- Outputs ---
set ROBUST_OUT_DIR=results\robust
set PROMO_OUT_DIR=results\promotions
set POST_OUT=results\post_analysis_summary.json
set PROMOTED_FILE=%PROMO_OUT_DIR%\faseA_promoted.json
set RULES_CFG=configs\promotion_rules_faseA_to_B.json

REM =====================================================
REM HARDWARE SAFE (8 GB RAM)
REM =====================================================
set WORKERS=3
set BATCH_SIZE=4
set SAMPLES=300

REM =====================================================
REM SEEDS A ROTAR
REM =====================================================
set SEEDS=1337 2024 777

REM =====================================================
REM VENTANAS TEMPORALES (FASE A)
REM =====================================================
set WINDOWS=2019-01_2020-12 2021-01_2021-12 2022-01_2023-06

REM =====================================================
REM SANITY CHECK
REM =====================================================
echo =========================================
echo [SANITY]
echo DATA=%DATA%
echo BASE_CFG=%BASE_CFG%
echo RULES=%RULES_CFG%
echo =========================================
echo.

if not exist "%ROBUST_OUT_DIR%" mkdir "%ROBUST_OUT_DIR%"
if not exist "%PROMO_OUT_DIR%" mkdir "%PROMO_OUT_DIR%"

:LOOP
REM =====================================================
REM RUN_ID (para no pisar outputs)
REM =====================================================
for /f "tokens=1-3 delims=/:. " %%a in ("%date% %time%") do (
  set RUN_ID=%%a%%b%%c
)
set RUN_ID=%RUN_ID: =%
set RUN_ID=%RUN_ID:-=%
set RUN_ID=%RUN_ID:/=%
set RUN_ID=%RUN_ID::=%

echo =========================================
echo [PIPELINE] START RUN_ID=%RUN_ID%
echo =========================================

REM =====================================================
REM FASE A — ROBUST OPTIMIZATION
REM =====================================================
for %%W in (%WINDOWS%) do (
  for %%S in (%SEEDS%) do (

    set OUT_FILE=%ROBUST_OUT_DIR%\robust_%%W_seed%%S_run%RUN_ID%.json

    echo =========================================
    echo [ROBUST] WINDOW=%%W  SEED=%%S  RUN=%RUN_ID%
    echo =========================================

    %PYTHON% %ROBUST_SCRIPT% ^
      --data "%DATA%" ^
      --base-config "%BASE_CFG%" ^
      --window %%W ^
      --samples %SAMPLES% ^
      --seed %%S ^
      --workers %WORKERS% ^
      --batch-size %BATCH_SIZE% ^
      --parallel ^
      --out "!OUT_FILE!"

    if errorlevel 1 (
      echo [ERROR] Robust optimizer failed (%%W seed %%S)
      goto END
    )

    echo [ROBUST] Finished %%W seed %%S
    echo.
  )
)

echo =========================================
echo [PIPELINE] FASE A COMPLETADA (RUN=%RUN_ID%)
echo =========================================

REM =====================================================
REM POST-ANALYSIS
REM =====================================================
echo =========================================
echo [POST] Running analysis_post_robust.py
echo =========================================
%PYTHON% %POST_SCRIPT% --results-dir "%ROBUST_OUT_DIR%" --out "%POST_OUT%"

if errorlevel 1 (
  echo [ERROR] Post-analysis failed
  goto END
)

REM =====================================================
REM PROMOTER A -> B
REM =====================================================
echo =========================================
echo [PROMOTER] Running promoter_faseA_to_B.py
echo =========================================
%PYTHON% %PROMOTER_SCRIPT% --in "%POST_OUT%" --rules "%RULES_CFG%" --out "%PROMOTED_FILE%"

if errorlevel 1 (
  echo [ERROR] Promoter failed
  goto END
)

REM =====================================================
REM CHECK PROMOTION RESULT
REM =====================================================
if exist "%PROMOTED_FILE%" (
  for %%A in ("%PROMOTED_FILE%") do (
    if %%~zA GTR 50 (
      echo =========================================
      echo [PROMOTER] PROMOTION OUTPUT EXISTS
      echo [PROMOTER] File: %PROMOTED_FILE%
      echo [PROMOTER] If selected_count>0 => ready for FASE B
      echo =========================================
    )
  )
)

REM Si querés “autostop” cuando haya candidatos:
REM (esto no depende del tamaño del archivo, depende de selected_count adentro;
REM lo dejamos simple para BAT: si existe, seguimos igual y vos revisás)
echo =========================================
echo [PIPELINE] LOOP CONTINUES (no human intervention)
echo =========================================
goto LOOP

:END
echo =========================================
echo [PIPELINE] FINISHED
echo =========================================
pause


