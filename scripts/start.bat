@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================
echo   Beacon Math Agent — 一键启动 (Windows)
echo ============================================
echo.

cd /d "%~dp0.."

REM --- 检查 Node.js ---
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 Node.js，请先安装：https://nodejs.org
    pause
    exit /b 1
)
echo [OK] Node.js %node_version% 已就绪

REM --- 检查 uv（Python 包管理器）---
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [警告] 未找到 uv，Python 后端可能无法启动
    echo        安装指引：https://docs.astral.sh/uv/getting-started/installation/
) else (
    echo [OK] uv 已就绪
)

REM --- 检查 .env ---
if not exist ".env" (
    echo [警告] 未找到 .env 文件，将从 .env.example 复制
    copy ".env.example" ".env" >nul
    echo        请编辑 .env 填入你的 LLM API 配置
)

REM --- 安装 Node.js 依赖 ---
if not exist "node_modules" (
    echo.
    echo [安装] Node.js 依赖...
    call npm install
    if %errorlevel% neq 0 (
        echo [错误] npm install 失败
        pause
        exit /b 1
    )
)

REM --- 安装 Python 依赖 ---
echo.
echo [检查] Python 依赖...
uv sync 2>nul
if %errorlevel% neq 0 (
    echo [警告] uv sync 失败，请确认 uv 已安装并配置正确
)

REM --- 启动 ---
echo.
echo ============================================
echo   正在启动服务...
echo   浏览器访问 http://localhost:5173
echo   按 Ctrl+C 停止
echo ============================================
echo.

REM 等待 1 秒后自动打开浏览器
start "" /b cmd /c "timeout /t 2 >nul && start http://localhost:5173"

call npm start
pause
