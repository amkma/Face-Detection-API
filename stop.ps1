Write-Host "Stopping Face Detection Server on port 8011..." -ForegroundColor Cyan

$stopped = $false

$connections = Get-NetTCPConnection -LocalPort 8011 -ErrorAction SilentlyContinue
if ($connections) {
    foreach ($conn in $connections) {
        if ($conn.OwningProcess -gt 0) {
            Write-Host "Stopping process PID $($conn.OwningProcess)..." -ForegroundColor Yellow
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            $stopped = $true
        }
    }
}

$uvicorns = Get-Process uvicorn* -ErrorAction SilentlyContinue
if ($uvicorns) {
    $uvicorns | Stop-Process -Force -ErrorAction SilentlyContinue
    $stopped = $true
}

if ($stopped) {
    Write-Host "Server stopped successfully." -ForegroundColor Green
} else {
    Write-Host "No running server was found on port 8011." -ForegroundColor Gray
}

Start-Sleep -Seconds 2
