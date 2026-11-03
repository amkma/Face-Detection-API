@echo off
setlocal
cd /d "%~dp0"

rem Create the virtual environment and install dependencies, once.
rem Does nothing if .venv already imports what the app needs.
rem First run needs internet; after that the server runs offline.

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment in .venv ...
    py -3 -m venv .venv
    if errorlevel 1 goto :no_python
)

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

".venv\Scripts\python.exe" -c "import fastapi, uvicorn, cv2" >nul 2>&1
if not errorlevel 1 goto :have_deps

echo Installing required Python packages. This can take several minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed
goto :have_deps

:no_python
echo.
echo Could not create the virtual environment: "py -3" was not found.
echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
echo and tick "Add python.exe to PATH" in the installer.
pause
exit /b 1

:pip_failed
echo.
echo Package installation failed. Check internet access and try again.
pause
exit /b 1

:have_deps
rem The detector blob is fetched once, sha256-checked, and then left alone.
if exist "models\face_detection_yunet_2023mar.onnx" goto :start
echo Fetching the pinned YuNet detector into models\ ...
".venv\Scripts\python.exe" -m app.fetch_models
if errorlevel 1 (
    echo.
    echo Fetching the detector failed. Check internet access;
    echo the server will not start without that file.
    pause
    exit /b 1
)

:start
echo ========================================================
echo Starting Clear Face Detection Server...
echo Directory: %CD%
echo Web page:  http://127.0.0.1:8011/
echo JSON API:  POST http://127.0.0.1:8011/api/verify
echo ========================================================
echo Press Ctrl+C to stop.
echo.

".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8011
echo.
echo The server has stopped.
pause
