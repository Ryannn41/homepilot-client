# 连接本机服务端（homepilot-agent 或 tools.mock_server），本地开发测试用。
# 用法：.\start-local.ps1 [app.main 的参数，如 --ws-test 3]
$ErrorActionPreference = "Stop"

# 脚本在当前会话中运行，结束后恢复工作目录和环境变量，避免影响之后的命令
$previous = $env:HOMEPILOT_SERVER_URL
$env:HOMEPILOT_SERVER_URL = "ws://127.0.0.1:8000/ws/voice/"
Push-Location $PSScriptRoot
try {
    python -m app.main @args
} finally {
    Pop-Location
    $env:HOMEPILOT_SERVER_URL = $previous
}
