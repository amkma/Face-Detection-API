@echo off
setlocal
cd /d "%~dp0"

rem First run on a fresh Windows box: create .venv, install deps, fetch the
rem YuNet blob, then hand over to start.bat (which only starts the server).
rem Safe to run again; if .venv already works this just calls start.bat.

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment in .venv ...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
        echo and tick "Add python.exe to PATH" in the installer, then run setup.bat again.
        pause
        exit /b 1
    )
    call ".venv\Scripts\activate.bat"
    python -m pip install --upgrade pip
)

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

".venv\Scripts\python.exe" -c "import fastapi, uvicorn, cv2" >nul 2>&1
if errorlevel 1 (
    echo Installing required Python packages. This can take several minutes...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Package installation failed. Check internet access and that the
        echo Python version is 3.10 or newer, then run setup.bat again.
        pause
        exit /b 1
    )
)

if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo Fetching the pinned YuNet detector into models\ ...
    ".venv\Scripts\python.exe" -m app.fetch_models
    if errorlevel 1 (
        echo.
        echo Fetching the detector failed. The server needs that file to start.
        pause
        exit /b 1
    )
)

echo.
echo Tests need pytest, which is not a runtime dependency. Installing it
echo into .venv only for this run, then running the suite.
".venv\Scripts\python.exe" -m pip install --quiet pytest
if errorlevel 1 goto :test_prep_failed

echo.
echo Running the test suite...
".venv\Scripts\python.exe" -m pytest tests -q
if errorlevel 1 (
    echo.
    echo Tests failed. See the output above.
    pause
    exit /b 1
)
echo.
echo Tests passed.
pause
exit /b 0

:test_prep_failed
echo.
echo Could not install pytest. Check internet access and try again.
pause
exit /b 1
