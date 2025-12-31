@echo off
setlocal ENABLEDELAYEDEXPANSION

REM =====================================================
REM ROOT (moverse al root del repo)
REM =====================================================
cd /d "%~dp0\.."

REM =====================================================
REM HARDENING: imports + encoding
REM =====================================================
set "PYTHON=python"
set "PYTHONPATH=%CD%"
set "PYTHONIOENCODING=utf-8"

REM Asegurar que analysis sea package para -m
if not exist "analysis\__init__.py" ( type nul > "analysis\__init__.py" )

REM =====================================================
REM CONFIG GENERAL
REM =====================================================
set "DATA=datasets\SOLUSDT\1m"
set "BASE_CFG=configs\pipeline_research_backtest.json"

set "ROBUST_OUT_DIR=results\robust"
set "PROMO_DIR=results\promotions"
set "STAGEC_TRADES_DIR=results\pipeline_stageC_trades"
set "FROZEN_DIR=results\frozen"
set "LOG_DIR=logs"
set "LOG_FILE=%LOG_DIR%\autoloop_ABC.log"

REM ---------------------------
REM A: recursos
REM ---------------------------
set "A_WORKERS=3"
set "A_BATCH_SIZE=4"
set "A_SAMPLES=300"

REM ---------------------------
REM B: samples
REM ---------------------------
set "B_SAMPLES=200"
set "B_SEED=1337"

REM ---------------------------
REM C: options
REM ---------------------------
set "C_WARMUP=500"
set "C_INTERVAL=1m"
set "STAGEC_REQUIRE_GLOBAL_PASS=1"
set "STAGEC_MIN_WINDOWS_OK=2"

REM ---------------------------
REM Ventanas (FASE A/B/C)
REM ---------------------------
set "WINDOWS=2019-01_2020-12 2021-01_2021-12 2022-01_2023-06"

REM ---------------------------
REM Autoloop: seeds incremental por ciclo
REM ---------------------------
set "SEEDS_PER_CYCLE=3"
set "SEED_STATE_FILE=results\autoloop_seed_base.txt"
set "START_SEED=1000"

REM Delay entre ciclos
set "CYCLE_DELAY_SEC=30"

REM Si querés que NO se frene aunque haya promovidos en C:
set "KEEP_RUNNING=0"

REM Shortlist top-N
set "SHORTLIST_N=10"

REM Stop manual
set "STOP_FILE=results\STOP_AUTOLOOP.txt"

REM ---------------------------
REM Auto-recovery supervisor
REM ---------------------------
set "SUPERVISOR_DELAY_SEC=10"

REM Lock anti doble ejecución
set "LOCK_FILE=results\AUTOLOOP.lock"

REM Heartbeat
set "HEARTBEAT_FILE=results\autoloop_heartbeat.txt"

REM Archive legacy robust (sin meta/phase A)
set "ARCHIVE_LEGACY=1"
set "ROBUST_LEGACY_DIR=results\robust_legacy"

REM =====================================================
REM INIT DIRS
REM =====================================================
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%ROBUST_OUT_DIR%" mkdir "%ROBUST_OUT_DIR%"
if not exist "%PROMO_DIR%" mkdir "%PROMO_DIR%"
if not exist "%STAGEC_TRADES_DIR%" mkdir "%STAGEC_TRADES_DIR%"
if not exist "%FROZEN_DIR%" mkdir "%FROZEN_DIR%"

REM =====================================================
REM LOCK
REM =====================================================
if exist "%LOCK_FILE%" (
  echo [LOCK] %LOCK_FILE% exists. Another instance may be running.
  echo [LOCK] %LOCK_FILE% exists. Exiting.>>"%LOG_FILE%"
  goto END
)
echo %DATE% %TIME% > "%LOCK_FILE%"

REM Seed init
if not exist "%SEED_STATE_FILE%" (
  echo %START_SEED%>"%SEED_STATE_FILE%"
)

echo =====================================================>>"%LOG_FILE%"
echo [BOOT] %DATE% %TIME%>>"%LOG_FILE%"
echo ROOT=%CD%>>"%LOG_FILE%"
echo PYTHONPATH=%PYTHONPATH%>>"%LOG_FILE%"
echo DATA=%DATA%>>"%LOG_FILE%"
echo BASE_CFG=%BASE_CFG%>>"%LOG_FILE%"
echo WINDOWS=%WINDOWS%>>"%LOG_FILE%"
echo =====================================================>>"%LOG_FILE%"

REM =====================================================
REM Archive legacy robust once (evita WARN infinito)
REM =====================================================
if "%ARCHIVE_LEGACY%"=="1" (
  if not exist "%ROBUST_LEGACY_DIR%" mkdir "%ROBUST_LEGACY_DIR%"
  %PYTHON% -c "import os, json, glob, shutil; \
d=r'%ROBUST_OUT_DIR%'; ld=r'%ROBUST_LEGACY_DIR%'; moved=0; \
for p in glob.glob(os.path.join(d,'robust_*.json')): \
  try: \
    with open(p,'r',encoding='utf-8') as f: rec=json.load(f); \
    m=rec.get('meta',{}) if isinstance(rec,dict) else {}; \
    ph=m.get('pipeline_phase'); \
    if ph!='A': \
      shutil.move(p, os.path.join(ld, os.path.basename(p))); moved+=1; \
  except Exception: \
    try: shutil.move(p, os.path.join(ld, os.path.basename(p))); moved+=1; \
    except Exception: pass \
print('ARCHIVED',moved)" >>"%LOG_FILE%" 2>&1
)

REM =====================================================
REM SUPERVISOR LOOP (auto-recovery)
REM =====================================================
:SUPERVISOR
if exist "%STOP_FILE%" goto STOP

call :ONE_CYCLE
set "RC=%ERRORLEVEL%"

echo [SUPERVISOR] cycle_rc=%RC% at %DATE% %TIME%>>"%LOG_FILE%"

REM RC=10 stop requested, RC=20 promoted found + KEEP_RUNNING=0
if "%RC%"=="10" goto STOP
if "%RC%"=="20" goto STOP

REM cualquier otro exit code: esperar y seguir (auto recovery)
timeout /t %SUPERVISOR_DELAY_SEC% /nobreak >nul
goto SUPERVISOR

REM =====================================================
REM One cycle: A -> POSTA -> B -> C (o sleep)
REM Exit codes:
REM   0 = normal
REM  10 = stop requested
REM  20 = promoted found (y KEEP_RUNNING=0)
REM =====================================================
:ONE_CYCLE
if exist "%STOP_FILE%" exit /b 10

echo %DATE% %TIME% > "%HEARTBEAT_FILE%"

REM next seeds
set /p SEED_BASE=<"%SEED_STATE_FILE%"
if "%SEED_BASE%"=="" set "SEED_BASE=%START_SEED%"

set "SEED1=%SEED_BASE%"
set /a SEED2=SEED_BASE+1
set /a SEED3=SEED_BASE+2
set "SEEDS=%SEED1% %SEED2% %SEED3%"
set /a NEXT_SEED_BASE=SEED_BASE+%SEEDS_PER_CYCLE%
echo %NEXT_SEED_BASE%>"%SEED_STATE_FILE%"

echo ---------------------------------------------------->>"%LOG_FILE%"
echo [CYCLE] %DATE% %TIME%  seed_base=%SEED_BASE%  seeds=%SEEDS%>>"%LOG_FILE%"

REM =========================
REM FASE A (freezeada)
REM =========================
set "PIPELINE_PHASE=A"
echo %DATE% %TIME% > "%HEARTBEAT_FILE%"

for %%W in (%WINDOWS%) do (
  for %%S in (%SEEDS%) do (
    set "OUT_FILE=%ROBUST_OUT_DIR%\robust_%%W_seed%%S.json"

    if exist "!OUT_FILE!" (
      echo [A] SKIP exists !OUT_FILE!>>"%LOG_FILE%"
    ) else (
      echo [A] RUN window=%%W seed=%%S out=!OUT_FILE!>>"%LOG_FILE%"
      %PYTHON% -m analysis.robust_optimizer ^
        --data "%DATA%" ^
        --base-config "%BASE_CFG%" ^
        --window %%W ^
        --samples %A_SAMPLES% ^
        --seed %%S ^
        --workers %A_WORKERS% ^
        --batch-size %A_BATCH_SIZE% ^
        --parallel ^
        --out "!OUT_FILE!" >>"%LOG_FILE%" 2>&1

      if errorlevel 1 (
        echo [A] ERROR robust failed window=%%W seed=%%S>>"%LOG_FILE%"
      )
    )
    echo %DATE% %TIME% > "%HEARTBEAT_FILE%"
  )
)

REM =========================
REM POST A
REM =========================
echo [POSTA] Running analysis.analysis_post_robust>>"%LOG_FILE%"
%PYTHON% -m analysis.analysis_post_robust >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [POSTA] ERROR post failed>>"%LOG_FILE%"
  goto CYCLE_SLEEP
)

set "FASEA_FILE=%PROMO_DIR%\faseA_promoted.json"
if not exist "%FASEA_FILE%" (
  echo [POSTA] No %FASEA_FILE% yet.>>"%LOG_FILE%"
  goto CYCLE_SLEEP
)

for %%A in ("%FASEA_FILE%") do set "FASEA_SIZE=%%~zA"
if "!FASEA_SIZE!" LSS "20" (
  echo [POSTA] %FASEA_FILE% too small (!FASEA_SIZE!).>>"%LOG_FILE%"
  goto CYCLE_SLEEP
)

REM =========================
REM FASE B
REM =========================
set "PIPELINE_PHASE=B"
echo [B] Running Stage B>>"%LOG_FILE%"
%PYTHON% -m analysis.stage_b_risk_calibration ^
  --data "%DATA%" ^
  --base-config "%BASE_CFG%" ^
  --fasea "%FASEA_FILE%" ^
  --out "%PROMO_DIR%\faseB_promoted.json" ^
  --report-csv "%PROMO_DIR%\faseB_report.csv" ^
  --samples %B_SAMPLES% ^
  --seed %B_SEED% >>"%LOG_FILE%" 2>&1

if errorlevel 1 (
  echo [B] ERROR Stage B failed>>"%LOG_FILE%"
  goto CYCLE_SLEEP
)

set "FASEB_FILE=%PROMO_DIR%\faseB_promoted.json"
if not exist "%FASEB_FILE%" (
  echo [B] No %FASEB_FILE% produced.>>"%LOG_FILE%"
  goto CYCLE_SLEEP
)
for %%B in ("%FASEB_FILE%") do set "FASEB_SIZE=%%~zB"
if "!FASEB_SIZE!" LSS "20" (
  echo [B] %FASEB_FILE% too small (!FASEB_SIZE!).>>"%LOG_FILE%"
  goto CYCLE_SLEEP
)

REM =========================
REM FASE C
REM =========================
set "PIPELINE_PHASE=C"
echo [C] Running Stage C>>"%LOG_FILE%"
%PYTHON% -m analysis.stage_c_pipeline_eval ^
  --faseb "%FASEB_FILE%" ^
  --data "%DATA%" ^
  --base-config "%BASE_CFG%" ^
  --out "%PROMO_DIR%\faseC_promoted.json" ^
  --report-csv "%PROMO_DIR%\faseC_report.csv" ^
  --trades-dir "%STAGEC_TRADES_DIR%" ^
  --interval %C_INTERVAL% ^
  --warmup %C_WARMUP% >>"%LOG_FILE%" 2>&1

if errorlevel 1 (
  echo [C] ERROR Stage C failed>>"%LOG_FILE%"
  goto CYCLE_SLEEP
)

set "FASEC_FILE=%PROMO_DIR%\faseC_promoted.json"
if not exist "%FASEC_FILE%" (
  echo [C] No %FASEC_FILE% => keep exploring.>>"%LOG_FILE%"
  goto CYCLE_SLEEP
)

for %%C in ("%FASEC_FILE%") do set "FASEC_SIZE=%%~zC"
if "!FASEC_SIZE!" LSS "20" (
  echo [C] %FASEC_FILE% too small (!FASEC_SIZE!) => keep exploring.>>"%LOG_FILE%"
  goto CYCLE_SLEEP
)

REM =========================
REM FREEZE SNAPSHOT + SHORTLIST
REM =========================
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%t"
set "SNAP_DIR=%FROZEN_DIR%\%TS%"
mkdir "%SNAP_DIR%" >nul 2>&1

copy "%FASEA_FILE%" "%SNAP_DIR%\faseA_promoted.json" >nul
if exist "%PROMO_DIR%\faseA_debug_summary.json" copy "%PROMO_DIR%\faseA_debug_summary.json" "%SNAP_DIR%\" >nul
copy "%FASEB_FILE%" "%SNAP_DIR%\faseB_promoted.json" >nul
if exist "%PROMO_DIR%\faseB_report.csv" copy "%PROMO_DIR%\faseB_report.csv" "%SNAP_DIR%\" >nul
copy "%FASEC_FILE%" "%SNAP_DIR%\faseC_promoted.json" >nul
if exist "%PROMO_DIR%\faseC_report.csv" copy "%PROMO_DIR%\faseC_report.csv" "%SNAP_DIR%\" >nul

%PYTHON% -c "import json; p=r'%FASEC_FILE%'; n=int(r'%SHORTLIST_N%'); d=json.load(open(p,'r',encoding='utf-8')); \
def score(x): \
  for k in ('promotion_score','global_score','score','score_v2','score_v1'): \
    v=x.get(k,None); \
    if isinstance(v,(int,float)): return float(v); \
  g=x.get('global',{}); \
  for k in ('score','score_v2','score_v1'): \
    v=g.get(k,None) if isinstance(g,dict) else None; \
    if isinstance(v,(int,float)): return float(v); \
  return -1e18; \
d_sorted=sorted(d, key=score, reverse=True); \
out={'top_n':n,'count':len(d_sorted),'items':d_sorted[:n]}; \
json.dump(out, open(r'%SNAP_DIR%\\shortlist_topN.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)" >>"%LOG_FILE%" 2>&1

echo [C] PROMOTED FOUND. Snapshot frozen at %SNAP_DIR%>>"%LOG_FILE%"
echo =========================================
echo [AUTOLOOP] CANDIDATES FOUND (FASE C)
echo Snapshot: %SNAP_DIR%
echo =========================================

if "%KEEP_RUNNING%"=="1" (
  goto CYCLE_SLEEP
) else (
  exit /b 20
)

:CYCLE_SLEEP
echo [SLEEP] %CYCLE_DELAY_SEC%s>>"%LOG_FILE%"
timeout /t %CYCLE_DELAY_SEC% /nobreak >nul
exit /b 0

REM =====================================================
REM STOP / CLEANUP
REM =====================================================
:STOP
echo [STOP] Requested. Cleaning up.>>"%LOG_FILE%"
goto END

:END
del "%LOCK_FILE%" >nul 2>&1
echo [END] %DATE% %TIME%>>"%LOG_FILE%"
echo Done.
pause



