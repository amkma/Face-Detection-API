$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
if (Test-Path "$PSScriptRoot\check image\app\main.py") {
    Set-Location "$PSScriptRoot\check image"
} else {
    Set-Location $PSScriptRoot
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    & py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Could not create the virtual environment. Install Python 3 and try again."
        exit 1
    }
}

& ".\.venv\Scripts\python.exe" -c "import importlib.metadata as m; import fastapi, uvicorn, insightface, onnxruntime; assert m.version('insightface') == '2.1'" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required Python packages. This can take several minutes..." -ForegroundColor Yellow
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Package installation failed. Check that internet access is available, then try again."
        exit 1
    }
}

Write-Host "Starting Clear Face Detection Server..." -ForegroundColor Cyan
Write-Host "Directory: $(Get-Location)" -ForegroundColor Gray
Write-Host "JSON API: POST http://127.0.0.1:8000/api/verify" -ForegroundColor Gray
& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) {
    Write-Error "The service stopped with an error. Review the messages above."
    exit $LASTEXITCODE
}
