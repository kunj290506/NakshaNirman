<#
.SYNOPSIS
    One-command startup for CAD Floor Plan Generator (local development).
.DESCRIPTION
    Installs dependencies and starts both backend (FastAPI) and frontend (Vite)
    in parallel. Access the app at http://localhost:5173
.EXAMPLE
    .\start.ps1
#>

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CAD Floor Plan Generator - Starting...    " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Check prerequisites ──────────────────────────────────────
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

# Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "  ERROR: Python not found. Install Python 3.11+ from python.org" -ForegroundColor Red
    exit 1
}
$pyVer = python --version 2>&1
Write-Host "  Python: $pyVer" -ForegroundColor DarkGray

# Node
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
    Write-Host "  ERROR: Node.js not found. Install Node 20+ from nodejs.org" -ForegroundColor Red
    exit 1
}
$nodeVer = node --version 2>&1
Write-Host "  Node:   $nodeVer" -ForegroundColor DarkGray

# ── 2. Create .env if missing ────────────────────────────────────
if (-not (Test-Path "$ROOT\.env")) {
    if (Test-Path "$ROOT\.env.example") {
        Copy-Item "$ROOT\.env.example" "$ROOT\.env"
        Write-Host ""
        Write-Host "  Created .env from .env.example" -ForegroundColor Yellow
        Write-Host "  >> Edit .env to add your API keys (GROK_API_KEY, GROQ_API_KEY)" -ForegroundColor Yellow
        Write-Host ""
    }
}

# ── 3. Backend setup ────────────────────────────────────────────
Write-Host "[2/5] Setting up backend..." -ForegroundColor Yellow

$backendDir = "$ROOT\backend"
$venvDir = "$backendDir\.venv"

# Create virtual environment if needed
if (-not (Test-Path "$venvDir\Scripts\python.exe")) {
    Write-Host "  Creating Python virtual environment..." -ForegroundColor DarkGray
    python -m venv $venvDir
}

# Activate and install deps
$venvPython = "$venvDir\Scripts\python.exe"
$venvPip = "$venvDir\Scripts\pip.exe"

Write-Host "  Installing Python dependencies..." -ForegroundColor DarkGray
& $venvPip install -r "$backendDir\requirements.txt" --quiet 2>&1 | Out-Null
Write-Host "  Backend dependencies ready" -ForegroundColor Green

# ── 4. Frontend setup ───────────────────────────────────────────
Write-Host "[3/5] Setting up frontend..." -ForegroundColor Yellow

$frontendDir = "$ROOT\frontend"

if (-not (Test-Path "$frontendDir\node_modules")) {
    Write-Host "  Installing Node dependencies..." -ForegroundColor DarkGray
    Push-Location $frontendDir
    npm install --silent 2>&1 | Out-Null
    Pop-Location
}
Write-Host "  Frontend dependencies ready" -ForegroundColor Green

# ── 5. Copy .env to backend ─────────────────────────────────────
if (Test-Path "$ROOT\.env") {
    Copy-Item "$ROOT\.env" "$backendDir\.env" -Force
}

# ── 6. Start both servers ───────────────────────────────────────
Write-Host "[4/5] Starting servers..." -ForegroundColor Yellow

# Start backend in background
$backendJob = Start-Job -ScriptBlock {
    param($venvPython, $backendDir)
    Set-Location $backendDir
    & $venvPython -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
} -ArgumentList $venvPython, $backendDir

Write-Host "  Backend  -> http://localhost:8000  (API + docs at /docs)" -ForegroundColor Green

# Start frontend in background
$frontendJob = Start-Job -ScriptBlock {
    param($frontendDir)
    Set-Location $frontendDir
    npm run dev
} -ArgumentList $frontendDir

Write-Host "  Frontend -> http://localhost:5173  (Vite dev server)" -ForegroundColor Green

# ── 7. Ready ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "[5/5] Ready!" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  App running at: http://localhost:5173     " -ForegroundColor White
Write-Host "  API docs at:    http://localhost:8000/docs " -ForegroundColor White
Write-Host "  Press Ctrl+C to stop all servers          " -ForegroundColor DarkGray
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Wait and stream output
try {
    while ($true) {
        # Stream backend output
        $backendOutput = Receive-Job -Job $backendJob -ErrorAction SilentlyContinue
        if ($backendOutput) {
            $backendOutput | ForEach-Object { Write-Host "[backend] $_" -ForegroundColor DarkGray }
        }
        # Stream frontend output
        $frontendOutput = Receive-Job -Job $frontendJob -ErrorAction SilentlyContinue
        if ($frontendOutput) {
            $frontendOutput | ForEach-Object { Write-Host "[frontend] $_" -ForegroundColor DarkCyan }
        }
        # Check if either job failed
        if ($backendJob.State -eq 'Failed') {
            Write-Host "Backend crashed! Check errors above." -ForegroundColor Red
            Receive-Job -Job $backendJob
            break
        }
        if ($frontendJob.State -eq 'Failed') {
            Write-Host "Frontend crashed! Check errors above." -ForegroundColor Red
            Receive-Job -Job $frontendJob
            break
        }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    Write-Host ""
    Write-Host "Stopping servers..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job -Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob -Force -ErrorAction SilentlyContinue
    Remove-Job -Job $frontendJob -Force -ErrorAction SilentlyContinue
    Write-Host "All servers stopped." -ForegroundColor Green
}
