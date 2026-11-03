@echo off
setlocal
echo ========================================================
echo Stopping Face Detection Server on port 8011...
echo ========================================================

for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8011 ^| findstr LISTENING') do (
    echo Terminating PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

taskkill /F /IM uvicorn.exe >nul 2>&1
echo Done.
timeout /t 2 >nul
