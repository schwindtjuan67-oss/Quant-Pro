@echo off
setlocal enableextensions enabledelayedexpansion

REM =========================
REM Ajustes tuyos
REM =========================
set VENV_DIR=.venv

REM Path al dataset (carpeta con CSVs)
set DATA_DIR=datasets\SOLUSDT_1m

REM Config base (json) requerido para REAL in-memory backtest
set BASE_CONFIG=configs\base_config.json

REM Output dir
set OUT_DIR=results\robust

REM Ventana a correr (podés cambiarla)
set WINDOW=2020-01_2021-12

REM Samples por corrida (subilo/bajalo)
set SAMPLES=300

REM =========================
REM Memoria (8GB): recomendación
REM =========================
REM 1 worker = ultra seguro
REM 2 workers = ok a veces, depende del tamaño del dataset (ojo que se duplica data por proceso)
set WORKERS=1

REM batching reduce overhead
set BATCH_SIZE=8

REM =========================
REM Python / venv
REM =========================
if exist "%VENV_DIR%\Scripts\python.exe" (
  set PY="%VENV_DIR%\Scripts\python.exe"
) else (
  set PY=python
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM Vars de entorno que usa robust_optimizer.py
set ROBUST_PARALLEL=1
set ROBUST_WORKERS=%WORKERS%
set ROBUST_BATCH_SIZE=%BATCH_SIZE%
set ROBUST_PROGRESS_EVERY=25

echo.
echo =========================
echo [RUN] Robust seeds pipeline
echo [RUN] DATA_DIR   = %DATA_DIR%
echo [RUN] BASE_CONFIG= %BASE_CONFIG%
echo [RUN] WINDOW     = %WINDOW%
echo [RUN] SAMPLES    = %SAMPLES%
echo [RUN] WORKERS    = %WORKERS%
echo [RUN] BATCH_SIZE = %BATCH_SIZE%
echo =========================
echo.

for %%S in (1337 2024 777) do (
  set SEED=%%S
  set OUT_FILE=%OUT_DIR%\robust_seed_!SEED!__%WINDOW%.json
  set LOG_FILE=%OUT_DIR%\robust_seed_!SEED!__%WINDOW%.log

  echo [RUN] Seed=!SEED! -> !OUT_FILE!
  echo [RUN] Log = !LOG_FILE!

  %PY% robust_optimizer.py ^
    --data "%DATA_DIR%" ^
    --base-config "%BASE_CONFIG%" ^
    --window "%WINDOW%" ^
    --samples %SAMPLES% ^
    --seed !SEED! ^
    --workers %WORKERS% ^
    --parallel ^
    --batch-size %BATCH_SIZE% ^
    --out "!OUT_FILE!" ^
    1>> "!LOG_FILE!" 2>>&1

  echo [RUN] Finished seed=!SEED!
  echo.
)

echo [RUN] All seeds done. Running post-analysis...
%PY% post_analysis_robust.py

echo.
echo [RUN] Finished EVERYTHING.
pause

