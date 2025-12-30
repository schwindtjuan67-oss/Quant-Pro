@echo off
cd /d %~dp0

REM ===============================
REM 1) Búsqueda robusta (anti-overfit)
REM ===============================
python -m analysis.robust_optimizer ^
  --data datasets/SOLUSDT/1m ^
  --out results/robust_candidates.json

REM ===============================
REM 2) Generar grids locales
REM ===============================
python -m analysis.robust_to_grid ^
  results/robust_candidates.json ^
  results/grids_refine

REM ===============================
REM 3) Refinamiento con grid_runner
REM ===============================
for %%G in (results\grids_refine\*.json) do (
  python -m backtest.grid_runner ^
    --base-config config/base.json ^
    --grid %%G ^
    --data datasets/SOLUSDT/1m ^
    --from-date 2023-01-01 ^
    --to-date 2023-12-31 ^
    --results-dir results/grid_refine ^
    --resume
)
