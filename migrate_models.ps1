# Ollama 模型迁移脚本
# 将C盘模型迁移到D盘

Write-Host "============================================---"
Write-Host " Ollama 模型迁移脚本"
Write-Host "============================================---"
Write-Host ""

# 配置
$sourcePath = "$env:USERPROFILE\.ollama\models"
$targetPath = "D:\OllamaModels"
$backupPath = "D:\OllamaModels_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

Write-Host "源路径: $sourcePath"
Write-Host "目标路径: $targetPath"
Write-Host ""

# 检查源目录
if (-not (Test-Path $sourcePath)) {
    Write-Host "❌ 源目录不存在: $sourcePath" -ForegroundColor Red
    exit 1
}

# 检查目标目录
if (-not (Test-Path $targetPath)) {
    New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
    Write-Host "✅ 目标目录已创建: $targetPath" -ForegroundColor Green
}

# 计算大小
$size = (Get-ChildItem -Path $sourcePath -Recurse -File | Measure-Object -Property Length -Sum).Sum
$sizeGB = [math]::Round($size / 1GB, 2)
Write-Host "待迁移文件总大小: $sizeGB GB"
Write-Host ""

# 检查D盘可用空间
$drive = Get-PSDrive -Name D
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "D盘可用空间: $freeSpaceGB GB"

if ($freeSpaceGB -lt $sizeGB) {
    Write-Host "❌ D盘空间不足！需要至少 $sizeGB GB" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================---"
Write-Host " 开始迁移..."
Write-Host "============================================---"
Write-Host ""

# 步骤1：复制文件到目标位置（保留原有结构）
Write-Host "步骤1/4: 复制文件到D盘..." -ForegroundColor Cyan
Write-Host "这可能需要几分钟，请耐心等待..."
Write-Host ""

try {
    # 使用 robocopy 复制（更快更可靠）
    $robocopyResult = robocopy $sourcePath $targetPath /E /COPYALL /R:3 /W:5 /MT:8 /NP /NFL /NDL

    if ($LASTEXITCODE -lt 8) {
        Write-Host "✅ 文件复制完成！" -ForegroundColor Green
    } else {
        Write-Host "❌ 复制过程中出现错误" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "❌ 复制失败: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 步骤2：验证文件
Write-Host "步骤2/4: 验证复制文件..." -ForegroundColor Cyan

$sourceFileCount = (Get-ChildItem -Path $sourcePath -Recurse -File).Count
$targetFileCount = (Get-ChildItem -Path $targetPath -Recurse -File).Count

Write-Host "源文件数: $sourceFileCount"
Write-Host "目标文件数: $targetFileCount"

if ($sourceFileCount -ne $targetFileCount) {
    Write-Host "⚠️  文件数量不一致，停止后续操作" -ForegroundColor Yellow
    Write-Host "请检查目标目录并手动处理" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ 文件验证通过！" -ForegroundColor Green
Write-Host ""

# 步骤3：创建符号链接
Write-Host "步骤3/4: 创建符号链接..." -ForegroundColor Cyan

# 先删除原目录（备份一下）
Write-Host "备份原目录..."
Move-Item -Path $sourcePath -Destination "$sourcePath.bak" -Force

# 创建符号链接
Write-Host "创建符号链接..."
New-Item -ItemType SymbolicLink -Path $sourcePath -Target $targetPath -Force | Out-Null

if (Test-Path $sourcePath) {
    $link = Get-Item $sourcePath
    if ($link.LinkType -eq "SymbolicLink") {
        Write-Host "✅ 符号链接创建成功！" -ForegroundColor Green
        Write-Host "   $sourcePath -> $targetPath" -ForegroundColor Gray
    } else {
        Write-Host "❌ 符号链接创建失败" -ForegroundColor Red
        # 恢复备份
        Move-Item -Path "$sourcePath.bak" -Destination $sourcePath -Force
        exit 1
    }
}

Write-Host ""

# 步骤4：清理备份
Write-Host "步骤4/4: 清理备份..." -ForegroundColor Cyan
Remove-Item -Path "$sourcePath.bak" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "✅ 清理完成！" -ForegroundColor Green

Write-Host ""
Write-Host "============================================---"
Write-Host " 🎉 迁移完成！"
Write-Host "============================================---"
Write-Host ""
Write-Host "📋 验证方法："
Write-Host "   1. 运行: ollama list"
Write-Host "   2. 应该能看到所有模型"
Write-Host "   3. 运行: ollama run qwen2.5:7b 测试"
Write-Host ""
Write-Host "💾 空间释放："
Write-Host "   原C盘占用: $sizeGB GB"
Write-Host "   D盘新占用: $sizeGB GB"
Write-Host ""
Write-Host "⚠️  重要提示："
Write-Host "   如果 Ollama 服务正在运行，请重启 Ollama"
Write-Host "   （关闭当前终端，重新打开即可）"
Write-Host ""

# 保持窗口
Write-Host "按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
