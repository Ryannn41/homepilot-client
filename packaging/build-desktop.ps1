# 在已激活 homepilot_client 环境的 PowerShell 中，于仓库根目录执行：
#   .\packaging\build-desktop.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm packaging/homepilot-desktop.spec
Write-Host "输出目录: $Root\dist\HomePilot"
Write-Host "请将 models\ 与 .env 放在与 HomePilot.exe 同级或按 README 配置。"