@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo NanoGS GUI GPU Support Installer
echo =================================
echo This script installs the source-mode GPU dependencies into .venv.
echo The compiled v1.0 Windows release stays CPU-only.
echo.

if not exist "requirements.txt" (
    echo requirements.txt was not found next to this script.
    echo Run this installer from the repository root.
    pause
    exit /b 1
)

set "PYTHON_CMD="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    echo Python was not found on PATH.
    echo Install Python 3.10+ first, then run this script again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    call %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

echo.
echo Select the CuPy package that matches your CUDA runtime:
echo   1. CUDA 12.x  ^(recommended for most current NVIDIA drivers^)
echo   2. CUDA 11.x
echo   3. Enter a custom CuPy package name
echo   4. Cancel
echo.
set /p GPU_CHOICE=Choice [1-4]: 

set "CUPY_PACKAGE="
if "%GPU_CHOICE%"=="1" set "CUPY_PACKAGE=cupy-cuda12x"
if "%GPU_CHOICE%"=="2" set "CUPY_PACKAGE=cupy-cuda11x"
if "%GPU_CHOICE%"=="3" (
    set /p CUPY_PACKAGE=Enter the package name ^(example: cupy-cuda12x^): 
)
if "%GPU_CHOICE%"=="4" (
    echo Installation cancelled.
    exit /b 0
)

if not defined CUPY_PACKAGE (
    echo No package was selected.
    pause
    exit /b 1
)

echo.
echo Upgrading pip ...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing base requirements ...
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.txt.
    pause
    exit /b 1
)

echo.
echo Installing !CUPY_PACKAGE! ...
call ".venv\Scripts\python.exe" -m pip install !CUPY_PACKAGE!
if errorlevel 1 (
    echo Failed to install !CUPY_PACKAGE!.
    echo Verify your CUDA runtime and try again with a matching package.
    pause
    exit /b 1
)

echo.
echo Verifying CuPy and CUDA device access ...
call ".venv\Scripts\python.exe" -c "import cupy as cp; print('CuPy', cp.__version__); print('CUDA devices', cp.cuda.runtime.getDeviceCount())"
if errorlevel 1 (
    echo CuPy installed, but the verification step failed.
    echo This usually means the CUDA runtime or NVIDIA driver does not match the package.
    pause
    exit /b 1
)

echo.
echo GPU support installation complete.
echo Start the GUI from source with:
echo   .venv\Scripts\python.exe nanogs_gui.py
echo Then set Device = gpu or auto in the GUI.
echo.
pause