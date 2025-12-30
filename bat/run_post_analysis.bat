@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ===============================
REM Post-analysis robust optimizer
REM ===============================

REM Activar venv
call .venv\Scripts\activate.bat

REM Ir a root del proyecto (ajustá si hace falta)
cd /d %~dp0

echo ==========================================
echo [POST] Running robust post-analysis
echo ==========================================

python analysis_post_robust.py

echo.
echo ==========================================
echo [POST] DONE
echo ==========================================
pause
