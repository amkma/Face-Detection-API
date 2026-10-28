@echo off
setlocal
cd /d "%~dp0"
if exist "check image\app\main.py" cd /d "check image"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :setup_failed
)

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -c "import importlib.metadata as m; import fastapi, uvicorn, insightface, onnxruntime; assert m.version('insightface') == '2.1'" >nul 2>&1
if errorlevel 1 (
    echo Installing required Python packages. This can take several minutes...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :setup_failed
)

echo ========================================================
echo Starting Clear Face Detection Server...
echo Directory: %CD%
echo JSON API: POST http://127.0.0.1:8000/api/verify
echo ========================================================

".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
if errorlevel 1 goto :startup_failed
goto :eof

:setup_failed
echo.
echo Setup failed. Ensure Python 3 and internet access are available, then review the messages above.
pause
exit /b 1

:startup_failed
echo.
echo The service stopped with an error. Review the messages above.
pause
exit /b 1
