@echo off
set BASE=C:\Users\PC\QuantBot-main

cd /d %BASE%

echo [INFO] Starting SHADOW from %BASE%

cmd /k python -m Live.run_shadow
