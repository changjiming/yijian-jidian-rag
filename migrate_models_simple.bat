@echo off
setlocal enabledelayedexpansion

echo ================================================
echo  Ollama 模型迁移工具
echo ================================================
echo.

set "SOURCE=%USERPROFILE%\.ollama\models"
set "TARGET=D:\OllamaModels"

echo 源路径: %SOURCE%
echo 目标路径: %TARGET%
echo.

if not exist "%SOURCE%" (
    echo ❌ 源目录不存在
    pause
    exit /b 1
)

if not exist "%TARGET%" (
    mkdir "%TARGET%" 2>nul
    if exist "%TARGET%" (
        echo ✅ 目标目录已创建
    )
)

echo.
echo 📊 正在计算文件大小...
for /f "tokens=3" %%a in ('dir /s "%SOURCE%" ^| find "File(s)"') do set "SIZE=%%a"
echo 待迁移大小: %SIZE%
echo.

echo ================================================
echo  开始迁移文件...
echo ================================================
echo.

xcopy "%SOURCE%\*" "%TARGET%\" /E /H /Y /Q

if %errorlevel% equ 0 (
    echo.
    echo ✅ 文件复制完成！
) else (
    echo.
    echo ❌ 复制失败
    pause
    exit /b 1
)

echo.
echo ================================================
echo  🎉 迁移完成！
echo ================================================
echo.
echo 📋 验证步骤：
echo    1. 关闭所有 Ollama 终端窗口
echo    2. 重新打开终端
echo    3. 运行: ollama list
echo.
echo 💡 现在可以重新下载新模型到 D 盘了
echo.

pause
