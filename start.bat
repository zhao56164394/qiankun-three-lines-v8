@echo off
chcp 65001 >nul
title 乾坤三线量化交易系统 v8.0

cd /d "%~dp0"

echo ============================================
echo   乾坤三线 v8.0 — 量化交易系统
echo ============================================
echo.

if not exist ".venv311\Scripts\activate.bat" (
    echo [错误] 未找到虚拟环境 .venv311
    pause
    exit /b 1
)

call .venv311\Scripts\activate.bat

echo 正在启动 Web 控制台...
echo 浏览器将自动打开，如未打开请访问 http://localhost:8501
echo 按 Ctrl+C 停止服务
echo.

streamlit run dashboard/app.py --server.headless false
