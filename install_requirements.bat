@echo off
cd /d "%~dp0"

REM Crear entorno virtual si no existe
if not exist ".venv\Scripts\activate.bat" (
    echo Creando entorno virtual .venv...
    python -m venv .venv
)

echo Activando entorno virtual...
call ".venv\Scripts\activate.bat"

echo Actualizando pip e instalando requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Dependencias instaladas. Para ejecutar el bot:
echo   call run_bot.bat
pause