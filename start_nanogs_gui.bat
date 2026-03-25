@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment not found at .venv\Scripts\python.exe
    echo Recreate it or point the BAT file to the correct interpreter.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "nanogs_gui.py"

if errorlevel 1 (
    echo.
    echo The GUI exited with an error.
    pause
)