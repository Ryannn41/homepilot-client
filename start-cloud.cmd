@echo off
REM 连接阿里云 ECS；绕过 PowerShell 执行策略，UNC/WSL 路径下也可直接双击或运行。
setlocal
pushd "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-cloud.ps1" %*
set ERR=%ERRORLEVEL%
popd
exit /b %ERR%