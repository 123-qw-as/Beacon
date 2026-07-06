#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "============================================"
echo "  Beacon Math Agent — 一键启动 (Unix/macOS)"
echo "============================================"
echo ""

# --- 检查 Node.js ---
if ! command -v node &>/dev/null; then
    echo "[错误] 未找到 Node.js，请先安装：https://nodejs.org"
    exit 1
fi
echo "[OK] Node.js $(node --version) 已就绪"

# --- 检查 uv ---
if ! command -v uv &>/dev/null; then
    echo "[警告] 未找到 uv，Python 后端可能无法启动"
    echo "       安装：curl -LsSf https://astral.sh/uv/install.sh | sh"
else
    echo "[OK] uv 已就绪"
fi

# --- 检查 .env ---
if [ ! -f ".env" ]; then
    echo "[警告] 未找到 .env，将从 .env.example 复制"
    cp .env.example .env
    echo "       请编辑 .env 填入你的 LLM API 配置"
fi

# --- 安装依赖 ---
if [ ! -d "node_modules" ]; then
    echo ""
    echo "[安装] Node.js 依赖..."
    npm install
fi

echo ""
echo "[检查] Python 依赖..."
uv sync 2>/dev/null || echo "[警告] uv sync 失败，请确认 uv 已安装"

# --- 启动 ---
echo ""
echo "============================================"
echo "  正在启动服务..."
echo "  浏览器访问 http://localhost:5173"
echo "  按 Ctrl+C 停止"
echo "============================================"
echo ""

# 自动打开浏览器（macOS / Linux）
if command -v open &>/dev/null; then
    (sleep 2 && open http://localhost:5173) &
elif command -v xdg-open &>/dev/null; then
    (sleep 2 && xdg-open http://localhost:5173) &
fi

exec npm start
