@echo off
setlocal ENABLEDELAYEDEXPANSION

REM =====================================
REM MOVERSE AL ROOT DEL REPO
REM =====================================
cd /d "%~dp0\.."

set PYTHON=python
set POST_SCRIPT=analysis\analysis_post_robust.py

echo =========================================
echo [POST] Running analysis_post_robust.py
echo =========================================

%PYTHON% %POST_SCRIPT% --verbose

echo.
echo =========================================
echo [POST] DONE
echo =========================================
pause
