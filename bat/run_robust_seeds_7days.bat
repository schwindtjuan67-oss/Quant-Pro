@echo off
setlocal ENABLEDELAYEDEXPANSION

REM =========================================
REM CONFIG GENERAL
REM =========================================

set PYTHON=python
set ROOT=%~dp0
set SCRIPT=analysis\robust_optimizer.py
set POST=analysis\analysis_post_robust.py

set DATA=datasets\SOLUSDT_1m
set BASE_CFG=configs\base_live.json
set OUT_DIR=results\robust

REM Hardware-safe (8 GB RAM)
set WORKERS=3
set BATCH_SIZE=4
set SAMPLES=300

REM =========================================
REM SEEDS
REM =========================================

set SEEDS=1337 2024 777

REM =========================================
REM VENTANAS (SECUENCIAL, NO PARALELO)
REM =========================================

set WINDOWS= ^
2019-01_2020-12 ^
2021-01_2021-12 ^
2022-01_2023-06

REM =========================================
REM FASE A — ROBUST SEARCH
REM =========================================

for %%W in (%WINDOWS%) do (
    for %%S in (%SEEDS%) do (

        set OUT_FILE=%OUT_DIR%\faseA_%%W_seed%%S.json

        echo =========================================
        echo FASE A  | WINDOW=%%W  SEED=%%S
        echo =========================================

        %PYTHON% %SCRIPT% ^
            --data %DATA% ^
            --base-config %BASE_CFG% ^
            --window %%W ^
            --samples %SAMPLES% ^
            --seed %%S ^
            --workers %WORKERS% ^
            --batch-size %BATCH_SIZE% ^
            --parallel ^
            --out !OUT_FILE!

        echo Finished %%W seed %%S
        echo.
    )
)

REM =========================================
REM POST-ANALYSIS + PROMOTION (AUTO)
REM =========================================

echo =========================================
echo POST-ANALYSIS + AUTO-PROMOTION
echo =========================================

%PYTHON% %POST%

echo =========================================
echo PIPELINE COMPLETO FINALIZADO
echo =========================================

pause

