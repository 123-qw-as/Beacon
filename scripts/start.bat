@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Beacon Math Agent - Start (Windows)
echo ============================================
echo.

cd /d "%~dp0.."

REM --- Check Node.js (hard dependency) ---
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install: https://nodejs.org
    pause
    exit /b 1
)
for /f "delims=" %%v in ('node --version') do set "node_version=%%v"
echo [OK] Node.js %node_version%

REM --- Check .env ---
if not exist ".env" (
    if exist ".env.example" (
        echo [INIT] Copy .env.example to .env
        copy ".env.example" ".env" >nul
        echo       Configure API key via onboarding wizard after launch
    )
)

REM --- Install Node.js deps ---
if not exist "node_modules" (
    echo.
    echo [INSTALL] npm install...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed
        pause
        exit /b 1
    )
)

REM --- Check uv (non-blocking, onboarding handles install) ---
where uv >nul 2>nul
if errorlevel 1 (
    echo [HINT] uv not found, Python backend unavailable
    echo        Install via onboarding wizard after launch
) else (
    echo [OK] uv ready
    echo [CHECK] Python deps ^(uv sync^)...
    uv sync >nul 2>nul
    if errorlevel 1 (
        echo [WARN] uv sync failed, Python backend may not start
    )
)

REM --- Start ---
echo.
echo ============================================
echo   Starting server...
echo   Open http://localhost:5173 in browser
echo   First run: onboarding wizard will appear
echo   Press Ctrl+C to stop
echo ============================================
echo.

REM Open browser after 2 seconds
start "" /b cmd /c "timeout /t 2 >nul && start http://localhost:5173"

call npm start
pause
