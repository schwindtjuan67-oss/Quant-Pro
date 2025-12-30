@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
uvicorn api.main:app --reload --port 8000
