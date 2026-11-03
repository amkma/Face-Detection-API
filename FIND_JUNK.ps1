# Paste into the ROOT folder of the repo on Windows (PowerShell, no venv).
# Pure ASCII on purpose: some Windows layers mangle non-ASCII in scripts.
# Copies the sources into probe\ byte for byte (no re-encode), normalises
# them to CRLF + UTF-8 BOM like Windows writes them, and reports lines that
# no longer look like code.
$ProgressPreference = "SilentlyContinue"
$ErrorActionPreference = "Stop"
$dir = Join-Path $PWD "probe"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$files = @{
  "app/__init__.py"                 = "probe/__init__.py"
  "app/main.py"                     = "probe/main.py"
  "app/face_service.py"             = "probe/face_service.py"
  "app/quality.py"                  = "probe/quality.py"
  "app/_detector.py"                = "probe/_detector.py"
  "app/_align.py"                   = "probe/_align.py"
  "app/fetch_models.py"             = "probe/fetch_models.py"
  "static/index.html"               = "probe/index.html"
  "static/overlay.js"               = "probe/overlay.js"
  "tests/test_align.py"             = "probe/test_align.py"
  "tests/test_detector.py"          = "probe/test_detector.py"
  "tests/test_endpoint_contract.py" = "probe/test_endpoint_contract.py"
  "tests/assets/portrait.png"       = "probe/portrait.png"
}

foreach ($src in $files.Keys) {
  if (-not (Test-Path $src)) { Write-Host "MISSING in repo: $src" -ForegroundColor Red; continue }
  [IO.File]::WriteAllBytes((Join-Path $dir $files[$src]), [IO.File]::ReadAllBytes($src))
}
$count = (Get-ChildItem $dir -Recurse -File | Measure-Object).Count
Write-Host ("probe file count: " + $count)

$utf8bom = New-Object System.Text.UTF8Encoding($true)
Get-ChildItem $dir -Filter *.py -Recurse -File | ForEach-Object {
  $t = [IO.File]::ReadAllText($_.FullName)
  $t = $t -replace "`r`n", "`n"
  $t = $t -replace "`n", "`r`n"
  [IO.File]::WriteAllText($_.FullName, $t, $utf8bom)
}

$patterns = @(
  '[|]\s*`',
  '[|]\s*[A-Za-z]+[()]',
  '[:=]\s+[A-Za-z_]\w*\s+:=',
  '[),\]]\s+[A-Za-z_]\w*\s+:=',
  '[A-Za-z0-9_]\s+[A-Za-z_]\w*\s*:=\s*(0|1|False|True|None)\s*$',
  '[0-9]\s+e\s*:=',
  'float(32|64)[0-9]{2}',
  '\ditems'
)

$hits = @()
Get-ChildItem $dir -Filter *.py -Recurse -File | ForEach-Object {
  $name = $_.FullName
  $lines = [IO.File]::ReadAllLines($_.FullName)
  $i = 0
  foreach ($line in $lines) {
    $i++
    $nonAscii = $line -match '[^\u0000-\u007F]'
    if ($nonAscii -and $i -gt 1200) { continue }
    $found = $nonAscii
    if (-not $found) {
      foreach ($pattern in $patterns) {
        if ($line -cmatch $pattern) { $found = $true; break }
      }
    }
    if ($found) { $hits += ('{0}:{1}  {2}' -f $name, $i, $line.Trim()) }
  }
}

if ($hits.Count -eq 0) {
  Write-Host "SCAN: clean, no non-ASCII or junk-token lines" -ForegroundColor Green
} else {
  Write-Host ('SCAN: ' + $hits.Count + ' suspicious line(s)') -ForegroundColor Yellow
  $hits | Select-Object -First 40 | ForEach-Object { Write-Host $_ }
}

