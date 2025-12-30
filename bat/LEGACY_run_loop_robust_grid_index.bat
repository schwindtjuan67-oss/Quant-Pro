@@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =====================================================
REM ROOT DEL PROYECTO
REM =====================================================
cd /d %~dp0\..

REM =====================================================
REM RUN MODE (PIPELINE | SHADOW | LIVE)
REM =====================================================
set "RUN_MODE=PIPELINE"

REM =====================================================
REM RUN ID GLOBAL (una corrida = un ID) + SANITIZE
REM OJO: %DATE% depende del locale (puede traer /)
REM =====================================================
set "PIPELINE_RUN_ID=%DATE%_%TIME%"
set "PIPELINE_RUN_ID=%PIPELINE_RUN_ID: =_%"
set "PIPELINE_RUN_ID=%PIPELINE_RUN_ID::=%"
set "PIPELINE_RUN_ID=%PIPELINE_RUN_ID:/=-%"
set "PIPELINE_RUN_ID=%PIPELINE_RUN_ID:.=-%"

REM =====================================================
REM PYTHON / PATH (UNBUFFERED FIX)
REM =====================================================
set "PY=python -u"
set "PYTHONUNBUFFERED=1"
set "PYTHONPATH=%cd%"

echo ROOT=%cd%
echo PYTHONPATH=%PYTHONPATH%
echo RUN_MODE=%RUN_MODE%
echo PIPELINE_RUN_ID=%PIPELINE_RUN_ID%
echo.

REM =====================================================
REM LOGGING (SOLO ESTADO DEL PIPELINE)
REM =====================================================
if not exist logs mkdir logs
set "LOG=logs\robust_loop.log"

echo ===============================================>>"%LOG%"
echo START LOOP %DATE% %TIME%>>"%LOG%"
echo RUN_MODE=%RUN_MODE%>>"%LOG%"
echo PIPELINE_RUN_ID=%PIPELINE_RUN_ID%>>"%LOG%"
echo ===============================================>>"%LOG%"

REM =====================================================
REM CONFIG GLOBAL
REM =====================================================
set "SYMBOLS=SOLUSDT"
set "REGIMES=TREND"
set "SEEDS=1337 2027 4041"

REM SOL empieza en 2020-08
set "WINDOWS=2020-08_2021-12 2022-01_2022-12 2023-01_2023-12"

set "DATA_BASE=datasets"
set "CONFIG_BASE=configs"
set "LIB=results\library\top_k_library.json"

REM =====================================================
REM ASEGURAR DIRECTORIOS BASE
REM =====================================================
for %%D in (
  results
  results\robust
  results\grid_refine
  results\library
) do (
  if not exist "%%D" mkdir "%%D"
)

REM =====================================================
REM LOOP PRINCIPAL
REM =====================================================
:LOOP

for %%M in (%SYMBOLS%) do (
for %%R in (%REGIMES%) do (

  set "DATA=%DATA_BASE%\%%M\1m"
  set "BASECFG=%CONFIG_BASE%\backtest_%%R_only.json"

  REM ---- contexto fijo por símbolo / régimen
  set "PIPELINE_SYMBOL=%%M"
  set "PIPELINE_REGIME=%%R"

  for %%S in (%SEEDS%) do (
  for %%W in (%WINDOWS%) do (

    REM ---- contexto dinámico (CRÍTICO para el logger)
    set "PIPELINE_SEED=%%S"
    set "PIPELINE_WINDOW=%%W"

    set "ROBUST_OUT=results\robust\%%M\%%R\seed_%%S\%%W\robust_candidates.json"
    set "GRID_DIR=results\grid_refine\%%M\%%R\seed_%%S\%%W"
    set "BEST_CFG=!GRID_DIR!\best_config.json"

    echo ==================================================
    echo %%M %%R seed=%%S window=%%W
    echo ==================================================

    echo [CASE] %%M %%R seed=%%S window=%%W>>"%LOG%"

    REM ================= ROBUST OPTIMIZER =================
    %PY% -m analysis.robust_optimizer ^
      --data "!DATA!" ^
      --out "!ROBUST_OUT!" ^
      --seed %%S ^
      --window %%W ^
      --min-candles 1 ^
      --base-config "!BASECFG!" ^
      --symbol %%M ^
      --interval 1m ^
      --warmup 500 >>"%LOG%" 2>&1

    if errorlevel 1 (
      echo [BAT][ERROR] robust_optimizer fallo (%%M %%R seed=%%S %%W)>>"%LOG%"
      goto HANDLE_ERROR
    )

    REM ================= GRID REFINE =================
    %PY% -m backtest.grid_runner ^
      --base-config "!BASECFG!" ^
      --grid "!ROBUST_OUT!" ^
      --data "!DATA!" ^
      --window %%W ^
      --results-dir "!GRID_DIR!" ^
      --resume >>"%LOG%" 2>&1

    if errorlevel 1 (
      echo [BAT][ERROR] grid_runner fallo (%%M %%R seed=%%S %%W)>>"%LOG%"
      goto HANDLE_ERROR
    )

    REM ================= INDEXADOR =================
    if exist "!BEST_CFG!" (
      %PY% -m analysis.topk_indexer ^
        --input "!BEST_CFG!" ^
        --library "%LIB%" >>"%LOG%" 2>&1
    ) else (
      echo [BAT][WARN] best_config.json no encontrado: !BEST_CFG!>>"%LOG%"
    )

    echo [OK] %%M %%R seed=%%S window=%%W>>"%LOG%"
    echo.>>"%LOG%"

  )
  )

  REM =====================================================
  REM POST-CICLO: ANALYZE PIPELINE (actualiza candidates)
  REM =====================================================
  echo [BAT] Running analyze_pipeline...>>"%LOG%"
  %PY% analysis/analyze_pipeline.py >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo [BAT][WARN] analyze_pipeline devolvio error (no freno el loop).>>"%LOG%"
  ) else (
    echo [BAT] analyze_pipeline OK.>>"%LOG%"
  )

)
)

echo [BAT] Ciclo completo OK. Esperando 300s...>>"%LOG%"
timeout /t 300 >nul
goto LOOP

REM =====================================================
REM HANDLER DE ERROR GLOBAL (NO CORTA)
REM =====================================================
:HANDLE_ERROR
echo [BAT] ERROR DETECTADO - sigo tras 300s.>>"%LOG%"
timeout /t 300 >nul
goto LOOP

