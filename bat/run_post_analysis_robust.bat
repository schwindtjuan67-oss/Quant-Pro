@echo off
setlocal enableextensions

REM === Ajustá esto si tu venv se llama distinto ===
set VENV_DIR=.venv

if exist "%VENV_DIR%\Scripts\python.exe" (
  set PY="%VENV_DIR%\Scripts\python.exe"
) else (
  set PY=python
)

echo [POST] Running post-analysis...
%PY% post_analysis_robust.py

echo.
echo [POST] Done.
pause

