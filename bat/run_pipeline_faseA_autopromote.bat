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

REM --- Data / Config ---
set DATA=datasets\SOLUSDT\1m
set BASE_CFG=configs\pipeline_research_backtest.json

REM --- Outputs ---
set ROBUST_OUT_DIR=results\robust
set PROMO_OUT_DIR=results\promotions

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
REM SANITY CHECK (CRÍTICO)
REM =====================================================

echo =========================================
echo [SANITY]
echo DATA=%DATA%
echo BASE_CFG=%BASE_CFG%
echo =========================================
echo.

REM =====================================================
REM CREAR DIRECTORIOS
REM =====================================================

if not exist "%ROBUST_OUT_DIR%" mkdir "%ROBUST_OUT_DIR%"
if not exist "%PROMO_OUT_DIR%" mkdir "%PROMO_OUT_DIR%"

REM =====================================================
REM FASE A — ROBUST OPTIMIZATION
REM =====================================================

for %%W in (%WINDOWS%) do (
    for %%S in (%SEEDS%) do (

        set OUT_FILE=%ROBUST_OUT_DIR%\robust_%%W_seed%%S.json

        echo =========================================
        echo [ROBUST] WINDOW=%%W  SEED=%%S
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
            echo [ERROR] Robust optimizer failed
            goto END
        )

        echo [ROBUST] Finished %%W seed %%S
        echo.
    )
)

echo =========================================
echo [PIPELINE] FASE A COMPLETADA
echo =========================================

REM =====================================================
REM POST-ANALYSIS + AUTO-PROMOTION (A → B)
REM =====================================================

echo =========================================
echo [POST] Running analysis_post_robust.py
echo =========================================

%PYTHON% %POST_SCRIPT%

REM =====================================================
REM CHECK PROMOTION RESULT
REM =====================================================

set PROMOTED_FILE=%PROMO_OUT_DIR%\faseA_promoted.json

if exist "%PROMOTED_FILE%" (
    for %%A in ("%PROMOTED_FILE%") do (
        if %%~zA GTR 10 (
            echo =========================================
            echo [PROMOTER] PROMOTION SUCCESS
            echo [PROMOTER] Ready for FASE B
            echo =========================================
            goto END
        )
    )
)

echo =========================================
echo [PROMOTER] NO PROMOTION
echo [PROMOTER] Continuing FASE A exploration
echo =========================================

:END
echo =========================================
echo [PIPELINE] FINISHED
echo =========================================
pause


