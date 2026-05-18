@echo off
title Orbital Watch — Setup
color 0B

echo.
echo  ============================================================
echo    ORBITAL WATCH  ^|  AI Collision Avoidance System
echo    One-click setup for Windows
echo  ============================================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  [1/4] Python found.

REM --- Create virtual environment ---
if not exist "venv" (
    echo  [2/4] Creating virtual environment...
    python -m venv venv
) else (
    echo  [2/4] Virtual environment already exists, skipping.
)

REM --- Activate and install ---
echo  [3/4] Installing dependencies (this may take 2-4 minutes)...
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo.
    echo  [ERROR] Dependency install failed.
    echo  Try running: pip install -r requirements.txt
    pause
    exit /b 1
)

echo  [4/4] All dependencies installed successfully.
echo.
echo  ============================================================
echo    Starting Orbital Watch on http://localhost:5000
echo    Open your browser and go to: http://localhost:5000
echo    Press CTRL+C to stop the server
echo  ============================================================
echo.

python app.py
pause
