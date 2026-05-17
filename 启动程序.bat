
@echo off
chcp 65001 &gt;nul
title 一建机电教材 RAG 问答系统

echo ========================================
echo   一建机电教材 RAG 问答系统
echo ========================================
echo.
echo [1/3] 检查Ollama服务...
timeout /t 1 &gt;nul

tasklist /FI "IMAGENAME eq ollama.exe" 2&gt;nul | find /I /N "ollama.exe" &gt;nul
if %ERRORLEVEL% NEQ 0 (
    echo [提示] 正在启动Ollama服务...
    start "" ollama serve
    timeout /t 3 /nobreak &gt;nul
)

echo.
echo [2/3] 检查模型...
timeout /t 1 &gt;nul

python -c "import sys; sys.path.insert(0, r'd:\Claude Code\一建机电RAG'); from desktop_app import LLM_MODEL, EMBED_MODEL; import requests; r = requests.get('http://localhost:11434/api/tags', timeout=5); m = [x['name'] for x in r.json()['models']]; print(f'✅ 已安装模型: {m}')" 2&gt;nul

if %ERRORLEVEL% NEQ 0 (
    echo [警告] 模型检查失败，但继续启动...
)

echo.
echo [3/3] 启动桌面应用...
timeout /t 1 &gt;nul

cd /d "d:\Claude Code\一建机电RAG"
python desktop_app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 应用程序启动失败！
    echo 请确保已安装Python依赖: pip install -r requirements.txt
    pause
)

