@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3.12 -c "import sys" >nul 2>nul
        if errorlevel 1 (
            py -3.11 -m venv .venv
        ) else (
            py -3.12 -m venv .venv
        )
    )
    if errorlevel 1 goto :python_failed
)
".venv\Scripts\python.exe" check_environment.py >nul 2>nul
if errorlevel 1 (
    echo Installing point cloud libraries into this folder. Internet access is needed for first setup.
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --only-binary=:all: -r requirements.txt
    if errorlevel 1 goto :setup_failed
)
".venv\Scripts\python.exe" check_environment.py
if errorlevel 1 goto :setup_failed
if /i "%~1"=="--check-setup" exit /b 0
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto :runtime_failed
exit /b 0
:python_failed
echo.
echo Could not create the Python environment. Install 64-bit Python 3.11 or 3.12
echo from python.org with Tcl/Tk, then run this file again.
goto :failed
:setup_failed
echo.
echo Dependency setup failed. Check the pip error above and your Internet connection.
echo You do not need to reinstall Python just because dependency setup failed.
goto :failed
:runtime_failed
echo.
echo Hybrid Regrid could not start. The error above has the details.
:failed
if /i "%~1"=="--check-setup" exit /b 1
pause
exit /b 1
