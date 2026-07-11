@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================
echo   Beacon Math Agent - 一键启动 (Windows)
echo ============================================
echo.

cd /d "%~dp0.."

REM --- 检查 Node.js（唯一硬依赖，没有它无法启动服务器）---
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 Node.js，请先安装：https://nodejs.org
    echo        安装后重新运行此脚本。
    pause
    exit /b 1
)
for /f "delims=" %%v in ('node --version') do set "node_version=%%v"
echo [OK] Node.js %node_version% 已就绪

REM --- 检查 .env ---
if not exist ".env" (
    if exist ".env.example" (
        echo [初始化] 从 .env.example 创建 .env
        copy ".env.example" ".env" >nul
        echo           启动后可通过引导向导配置 API 密钥
    )
)

REM --- 安装 Node.js 依赖 ---
if not exist "node_modules" (
    echo.
    echo [安装] Node.js 依赖...
    call npm install
    if !errorlevel! neq 0 (
        echo [错误] npm install 失败
        pause
        exit /b 1
    )
)

REM --- 检查 uv（不阻断启动，引导向导会处理安装）---
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [提示] 未找到 uv，Python 后端暂不可用
    echo        启动后可在引导向导中一键安装 uv
) else (
    echo [OK] uv 已就绪
    REM --- 安装 Python 依赖 ---
    echo [检查] Python 依赖...
    uv sync 2>nul
    if !errorlevel! neq 0 (
        echo [警告] uv sync 失败，Python 后端可能无法启动
    )
)

REM --- 启动 ---
echo.
echo ============================================
echo   正在启动服务...
echo   浏览器访问 http://localhost:5173
echo   首次使用会自动进入配置引导
echo   按 Ctrl+C 停止
echo ============================================
echo.

REM 等待 2 秒后自动打开浏览器
start "" /b cmd /c "timeout /t 2 >nul && start http://localhost:5173"

call npm start
pause
