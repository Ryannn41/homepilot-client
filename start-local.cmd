@echo off
REM 连接本机服务端；绕过 PowerShell 执行策略，UNC/WSL 路径下也可直接双击或运行。
setlocal
pushd "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-local.ps1" %*
set ERR=%ERRORLEVEL%
popd
exit /b %ERR%