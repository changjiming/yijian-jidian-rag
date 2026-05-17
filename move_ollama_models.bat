@echo off
echo ================================================
echo  Ollama 模型存储位置迁移脚本
echo ================================================
echo.

REM 检查管理员权限
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  提示: 建议以管理员身份运行此脚本
    echo.
)

REM 设置新的模型存储路径
set NEW_MODEL_PATH=D:\OllamaModels

echo 📁 新的模型存储路径: %NEW_MODEL_PATH%
echo.

REM 检查目标目录是否存在
if not exist "%NEW_MODEL_PATH%" (
    echo ➕ 创建目标目录...
    mkdir "%NEW_MODEL_PATH%" 2>nul
    if exist "%NEW_MODEL_PATH%" (
        echo ✅ 目录创建成功
    ) else (
        echo ❌ 目录创建失败，请手动创建
        pause
        exit /b 1
    )
)

echo.
echo ⚙️  配置 Ollama 使用新的模型路径...
echo.

REM 设置环境变量（仅当前会话有效）
set OLLAMA_MODELS=%NEW_MODEL_PATH%

REM 创建系统环境变量的注册表设置
echo 📝 添加系统环境变量...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v OLLAMA_MODELS /t REG_SZ /d "%NEW_MODEL_PATH%" /f >nul 2>&1

if %errorLevel% equ 0 (
    echo ✅ 系统环境变量设置成功
) else (
    echo ⚠️  系统环境变量设置失败（可能需要管理员权限）
)

echo.
echo ================================================
echo  配置完成！
echo ================================================
echo.
echo 📋 后续步骤：
echo    1. 关闭所有正在运行的 Ollama 终端窗口
echo    2. 删除 C 盘的临时下载文件（可选）
echo    3. 重新打开终端，运行以下命令：
echo.
echo       ollama pull qwen2.5:7b
echo       ollama pull mxbai-embed-large
echo.
echo 💡 提示：环境变量修改后需要重启终端才能生效
echo.

pause
