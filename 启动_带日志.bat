
@echo off
chcp 65001 >nul
title 一建机电教材 RAG 问答系统 - 日志窗口
echo.
echo ========================================
echo    一建机电教材 RAG 问答系统
echo ========================================
echo.
echo 正在启动程序...
echo.
echo [系统信息]
echo - Python: %PYTHON%
echo - 当前目录: %CD%
echo - Ollama URL: http://localhost:11434
echo.
echo ========================================
echo.
cd /d "%~dp0"
python desktop_app.py
if errorlevel 1 (
    echo.
    echo ========================================
    echo    程序异常退出！
    echo ========================================
    echo.
    echo 请检查错误信息后，按任意键退出...
    pause >nul
)
