@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ===========================
REM Settings
REM ===========================
set ROOT=%~dp0
cd /d "%ROOT%"

set DATA_INTERVAL=1m
set SAMPLES=200
set SEED_START=1
set SEED_END=50

REM Si tu robust_optimizer YA acepta fechas, dejalo en 1. Si no, ponelo en 0.
set ROBUST_SUPPORTS_DATES=1

REM Config base para grid_runner
set BASE_CONFIG=configs\backtest_trend_only.json

REM Inputs (listas)
set SYMBOLS_FILE=configs\symbols.txt
set REGIMES_FILE=configs\regimes.txt
set WINDOWS_FILE=configs\windows.csv

REM Output roots
set WF_ROOT=results\wf
set LIB_ROOT=results\library

if not exist "%WF_ROOT%" mkdir "%WF_ROOT%"
if not exist "%LIB_ROOT%" mkdir "%LIB_ROOT%"

echo [WF] ROOT=%ROOT%
echo [WF] SAMPLES=%SAMPLES%  SEEDS=%SEED_START%..%SEED_END%
echo.

REM ===========================
REM Loop: symbol x regime x window x seed
REM ===========================
for /f "usebackq delims=" %%S in ("%SYMBOLS_FILE%") do (
  set SYMBOL=%%S
  if "!SYMBOL!"=="" goto :continueSymbols

  for /f "usebackq delims=" %%R in ("%REGIMES_FILE%") do (
    set REGIME=%%R
    if "!REGIME!"=="" goto :continueRegimes

    for /f "usebackq tokens=1,2 delims=, skip=0" %%A in ("%WINDOWS_FILE%") do (
      set FROM_DATE=%%A
      set TO_DATE=%%B

      REM skip header if exists
      if /i "!FROM_DATE!"=="from" goto :continueWindows

      set FROM_TAG=!FROM_DATE:-=!
      set TO_TAG=!TO_DATE:-=!
      set WINDOW_TAG=!FROM_TAG!_!TO_TAG!

      for /l %%K in (%SEED_START%,1,%SEED_END%) do (
        set SEED=0000%%K
        set SEED=!SEED:~-4!

        set OUT_DIR=%WF_ROOT%\!SYMBOL!\!REGIME!\!WINDOW_TAG!\seed_!SEED!
        set ROBUST_OUT=!OUT_DIR!\robust_candidates.json
        set GRIDS_DIR=!OUT_DIR!\grids_refine
        set REFINE_DIR=!OUT_DIR!\refine

        if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"
        if not exist "!GRIDS_DIR!" mkdir "!GRIDS_DIR!"
        if not exist "!REFINE_DIR!" mkdir "!REFINE_DIR!"

        echo ==========================================================
        echo [WF] symbol=!SYMBOL! regime=!REGIME! window=!WINDOW_TAG! seed=!SEED!
        echo ==========================================================

        REM --- 1) Robust search ---
        if "!ROBUST_SUPPORTS_DATES!"=="1" (
          python -m analysis.robust_optimizer ^
            --data datasets\!SYMBOL!\%DATA_INTERVAL% ^
            --out "!ROBUST_OUT!" ^
            --samples %SAMPLES% ^
            --seed %%K ^
            --from-date "!FROM_DATE!" ^
            --to-date "!TO_DATE!"
        ) else (
          python -m analysis.robust_optimizer ^
            --data datasets\!SYMBOL!\%DATA_INTERVAL% ^
            --out "!ROBUST_OUT!" ^
            --samples %SAMPLES% ^
            --seed %%K
        )

        if errorlevel 1 (
          echo [WF][ERROR] robust_optimizer failed. Skipping seed !SEED!.
          echo.
          goto :continueSeed
        )

        REM --- 2) Robust -> local grids ---
        python analysis\robust_to_grid.py "!ROBUST_OUT!" "!GRIDS_DIR!"
        if errorlevel 1 (
          echo [WF][ERROR] robust_to_grid failed. Skipping seed !SEED!.
          echo.
          goto :continueSeed
        )

        REM --- 3) Refine each grid -> best_config.json ---
        for %%G in ("!GRIDS_DIR!\*.json") do (
          python -m backtest.grid_runner ^
            --base-config "%BASE_CONFIG%" ^
            --grid "%%~fG" ^
            --data datasets\!SYMBOL!\%DATA_INTERVAL% ^
            --from-date "!FROM_DATE!" ^
            --to-date "!TO_DATE!" ^
            --results-dir "!REFINE_DIR!" ^
            --resume
        )

        REM --- 4) Promote best_config.json to library with naming ---
        if exist "!REFINE_DIR!\best_config.json" (
          set LIB_DIR=%LIB_ROOT%\!SYMBOL!\!REGIME!\!WINDOW_TAG!
          if not exist "!LIB_DIR!" mkdir "!LIB_DIR!"

          copy /y "!REFINE_DIR!\best_config.json" "!LIB_DIR!\best__seed_!SEED!.json" >nul

          REM meta mínimo (sin python extra): guardo punteros de trazabilidad
          > "!LIB_DIR!\meta__seed_!SEED!.json" (
            echo {
            echo   "symbol": "!SYMBOL!",
            echo   "regime": "!REGIME!",
            echo   "window": "!WINDOW_TAG!",
            echo   "from_date": "!FROM_DATE!",
            echo   "to_date": "!TO_DATE!",
            echo   "seed": !SEED!,
            echo   "robust_out": "!ROBUST_OUT!",
            echo   "refine_dir": "!REFINE_DIR!"
            echo }
          )

          echo [WF] promoted -> "!LIB_DIR!\best__seed_!SEED!.json"
        ) else (
          echo [WF][WARN] best_config.json not found in "!REFINE_DIR!"
        )

        :continueSeed
        echo.
      )

      :continueWindows
    )

    :continueRegimes
  )

  :continueSymbols
)

echo [WF] DONE
endlocal
