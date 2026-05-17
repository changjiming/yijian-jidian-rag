# Ollama 模型迁移脚本（管理员版）
# 需要以管理员身份运行

Write-Host "============================================"
Write-Host " Ollama 模型迁移到 D 盘"
Write-Host "============================================"
Write-Host ""

$Source = "$env:USERPROFILE\.ollama\models"
$Target = "D:\OllamaModels"

Write-Host "源路径: $Source"
Write-Host "目标: $Target"
Write-Host ""

# 检查管理员权限
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "⚠️  请以管理员身份运行此脚本！" -ForegroundColor Yellow
    Write-Host "右键点击 -> 以管理员身份运行" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "或者在PowerShell中运行：" -ForegroundColor Cyan
    Write-Host "Start-Process powershell -Verb RunAs -ArgumentList '-File `"$PSCommandPath`"'" -ForegroundColor Gray
    Write-Host ""
    Read-Host "按 Enter 键退出"
    exit
}

Write-Host "✅ 检测到管理员权限" -ForegroundColor Green
Write-Host ""

# 创建目标目录
if (-not (Test-Path $Target)) {
    New-Item -Path $Target -ItemType Directory -Force | Out-Null
    Write-Host "✅ 已创建目标目录: $Target" -ForegroundColor Green
}

# 计算源文件大小
Write-Host "📊 正在计算文件大小..."
$totalSize = (Get-ChildItem -Path $Source -Recurse -File | Measure-Object -Property Length -Sum).Sum
$totalSizeGB = [math]::Round($totalSize / 1GB, 2)
Write-Host "总大小: $totalSizeGB GB" -ForegroundColor Cyan
Write-Host ""

# 确认操作
Write-Host "⚠️  即将开始迁移文件..." -ForegroundColor Yellow
$confirm = Read-Host "继续？(Y/N)"

if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "已取消"
    exit
}

# 执行复制
Write-Host ""
Write-Host "🚀 开始复制文件..."
Write-Host "(这可能需要几分钟，请耐心等待...)"
Write-Host ""

try {
    Copy-Item -Path "$Source\*" -Destination $Target -Recurse -Force -ErrorAction Stop
    Write-Host ""
    Write-Host "✅ 复制完成！" -ForegroundColor Green
} catch {
    Write-Host "❌ 复制失败: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "请手动复制文件：" -ForegroundColor Yellow
    Write-Host "从: $Source" -ForegroundColor Gray
    Write-Host "到: $Target" -ForegroundColor Gray
    Read-Host "按 Enter 键退出"
    exit 1
}

# 验证
Write-Host ""
Write-Host "🔍 验证文件..."
$sourceFiles = (Get-ChildItem -Path $Source -Recurse -File).Count
$targetFiles = (Get-ChildItem -Path $Target -Recurse -File).Count

if ($sourceFiles -eq $targetFiles) {
    Write-Host "✅ 验证通过！文件数量一致 ($sourceFiles 个)" -ForegroundColor Green
} else {
    Write-Host "⚠️  文件数量不一致: 源=$sourceFiles, 目标=$targetFiles" -ForegroundColor Yellow
}

# 创建符号链接
Write-Host ""
Write-Host "🔗 创建符号链接..."
$symlinkPath = $Source

# 备份原目录
$backupPath = "$Source.backup"
if (Test-Path $backupPath) {
    Remove-Item -Path $backupPath -Recurse -Force
}
Rename-Item -Path $Source -NewName "models.backup"

# 创建符号链接
try {
    New-Item -ItemType SymbolicLink -Path $Source -Target $Target -Force | Out-Null
    Write-Host "✅ 符号链接创建成功！" -ForegroundColor Green
    Write-Host "   $Source -> $Target" -ForegroundColor Gray
} catch {
    Write-Host "⚠️  符号链接创建失败" -ForegroundColor Yellow
    Write-Host "   手动恢复: 重命名 $Source.backup 为 models" -ForegroundColor Gray
}

# 完成
Write-Host ""
Write-Host "============================================"
Write-Host " 🎉 迁移完成！"
Write-Host "============================================"
Write-Host ""
Write-Host "📋 下一步：" -ForegroundColor Cyan
Write-Host "   1. 关闭所有 Ollama 相关终端"
Write-Host "   2. 重新打开 PowerShell"
Write-Host "   3. 运行: ollama list 验证"
Write-Host ""
Write-Host "💾 释放空间：" -ForegroundColor Cyan
Write-Host "   原 C 盘占用: $totalSizeGB GB"
Write-Host "   D 盘新占用: $totalSizeGB GB"
Write-Host ""
Write-Host "🗑️  释放更多空间（可选）：" -ForegroundColor Yellow
Write-Host "   如果一切正常，可以删除备份：" -ForegroundColor Gray
Write-Host "   Remove-Item -Path `"$Source.backup`" -Recurse -Force" -ForegroundColor Gray
Write-Host ""

Read-Host "按 Enter 键退出"
