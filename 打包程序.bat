
@echo off
chcp 65001 &gt;nul
echo ========================================
echo   打包一建机电RAG为可执行程序
echo ========================================
echo.

echo [1/5] 检查PyInstaller...
pip show pyinstaller &gt;nul 2&gt;&amp;1
if %ERRORLEVEL% NEQ 0 (
    echo 正在安装PyInstaller...
    pip install pyinstaller
)

echo.
echo [2/5] 创建打包脚本...

cd /d "d:\Claude Code\一建机电RAG"

echo.
echo [3/5] 开始打包...
pyinstaller --windowed --onefile --name "一建机电RAG" --icon=NONE desktop_app.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [4/5] 打包成功！
    echo 可执行文件位置: dist\一建机电RAG.exe
    echo.
    echo [5/5] 复制数据目录...
    if not exist "dist\data" mkdir dist\data
    if not exist "dist\data\uploads" mkdir dist\data\uploads
    echo.
    echo ========================================
    echo   🎉 打包完成！
    echo ========================================
    echo 可执行文件: dist\一建机电RAG.exe
    echo.
    echo 注意：运行程序前请确保：
    echo   1. Ollama服务已启动
    echo   2. 模型 qwen2.5:7b 和 mxbai-embed-large 已下载
    echo.
) else (
    echo.
    echo [错误] 打包失败，请检查错误信息！
)

pause

