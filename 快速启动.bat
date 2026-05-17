
@echo off
title 启动一建机电RAG（最简版）
cd /d "%~dp0"
python desktop_app.py
if %errorlevel% neq 0 (
    echo 启动失败，请检查是否已安装Python
    echo 运行: pip install -r requirements.txt
    pause
)

