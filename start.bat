@echo off
title CAD Floor Plan Generator
echo.
echo ============================================
echo   CAD Floor Plan Generator - Starting...
echo ============================================
echo.

cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Install Node 20+ from nodejs.org
    pause
    exit /b 1
)

:: Create .env if missing
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo   Created .env from .env.example
        echo   Edit .env to add your API keys
        echo.
    )
)

:: Backend venv setup
if not exist "backend\.venv\Scripts\python.exe" (
    echo [1/4] Creating Python virtual environment...
    python -m venv backend\.venv
)

echo [2/4] Installing backend dependencies...
backend\.venv\Scripts\pip.exe install -r backend\requirements.txt -q >nul 2>&1

:: Copy .env to backend
if exist ".env" copy ".env" "backend\.env" >nul

:: Frontend setup
if not exist "frontend\node_modules" (
    echo [3/4] Installing frontend dependencies...
    cd frontend
    call npm install --silent >nul 2>&1
    cd ..
)

echo [4/4] Starting servers...
echo.

:: Start backend in a new window
start "CAD-Backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

:: Give backend a moment
timeout /t 2 /nobreak >nul

:: Start frontend in a new window
start "CAD-Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo ============================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo ============================================
echo.
echo Close this window and the server windows to stop.
echo.

:: Open browser after a short delay
timeout /t 3 /nobreak >nul
start http://localhost:5173

pause
