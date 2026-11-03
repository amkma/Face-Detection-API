@echo off
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set PATH=%CD%\.venv\Scripts;%PATH%
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8011
echo.
echo The server has stopped.
pause
