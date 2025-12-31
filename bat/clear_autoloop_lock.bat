@echo off
setlocal
cd /d "%~dp0\.."
echo Cleaning autoloop artifacts...
if exist "results\AUTOLOOP.lock" del "results\AUTOLOOP.lock"
if exist "results\STOP_AUTOLOOP.txt" del "results\STOP_AUTOLOOP.txt"
echo Done.
pause
