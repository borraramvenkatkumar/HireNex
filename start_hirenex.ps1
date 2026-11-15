$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$vite = Join-Path $root "frontend\node_modules\vite\bin\vite.js"

if (-not (Test-Path $python)) {
    Write-Host "Python environment not found. Run the first-time setup in README.md:" -ForegroundColor Yellow
    Write-Host "py -3.11 -m venv .venv"
    Write-Host ".\.venv\Scripts\Activate.ps1"
    Write-Host "pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path $vite)) {
    Write-Host "Frontend dependencies are missing. Run 'npm install' inside the frontend folder first." -ForegroundColor Yellow
    exit 1
}

$backendCommand = "Set-Location -LiteralPath '$root'; & '$python' -m uvicorn backend.main:app --reload"
$frontendDir = Join-Path $root "frontend"
$frontendCommand = "Set-Location -LiteralPath '$frontendDir'; npm run dev"

Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand)
Start-Sleep -Seconds 2
Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand)

Write-Host "HireNex services launched. Open http://localhost:5173" -ForegroundColor Green
