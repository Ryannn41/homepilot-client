# homepilot-client
A local voice client for HomePilot, providing wake-word detection, voice activity detection, audio streaming, and continuous voice interaction with the cloud AI assistant.

HomePilot Client 运行在家庭 Windows 电脑上，负责本地唤醒、语音采集、与云端 Agent 的 WebSocket 通信以及播放云端语音。ASR、LLM、TTS 和业务逻辑都在云端 `homepilot-agent` 中完成。

> 当前进度：第一阶段，项目基础环境（配置管理 + 启动入口）。音频、唤醒词、VAD、WebSocket 尚未实现。

## 环境要求

- Windows 10/11 x64
- Anaconda 或 Miniconda
- Python 3.11（由 conda 安装）

## 创建环境

在 Anaconda Prompt 或已初始化 conda 的 PowerShell 中执行：

```powershell
conda create -y -n homepilot_client python=3.11
conda activate homepilot_client
```

## 安装依赖

在仓库根目录执行：

```powershell
pip install -r requirements.txt
```

如果默认镜像不可用（例如出现 HTTP 403），可以临时指定其他镜像：

```powershell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

当前依赖：

| 包 | 用途 |
| --- | --- |
| pydantic-settings | 从环境变量和 `.env` 读取配置 |
| numpy | 音频数据处理 |
| scipy | 采样率转换（`resample_poly`） |
| sounddevice | 音频采集和播放（Windows wheel 自带 PortAudio） |
| pytest | 单元测试 |

## 配置

所有配置都有默认值，不创建 `.env` 也能启动。需要修改时：

```powershell
copy .env.example .env
```

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOMEPILOT_SERVER_URL` | `ws://127.0.0.1:8000/ws/voice/` | 云端 Agent 的 WebSocket 地址 |
| `HOMEPILOT_DEVICE_ID` | `homepilot-dev` | 本设备的唯一标识 |
| `HOMEPILOT_AUTH_TOKEN` | 空 | 云端认证令牌，日志中不会打印明文 |
| `HOMEPILOT_INPUT_DEVICE` | 空（系统默认） | 输入设备索引或名称子串 |
| `HOMEPILOT_OUTPUT_DEVICE` | 空（系统默认） | 输出设备索引或名称子串 |
| `HOMEPILOT_AUDIO_HOSTAPI` | `wasapi` | 优先使用的 Host API：`wasapi` / `mme` / `directsound` |
| `HOMEPILOT_FRAME_MS` | `20` | 音频分片时长，只能是 20 或 40 毫秒 |
| `HOMEPILOT_SILENCE_TIMEOUT_S` | `15` | 对话中用户静默多少秒后退出会话 |
| `HOMEPILOT_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

优先级：系统环境变量 > `.env` > 默认值。`.env` 从当前工作目录读取，请在仓库根目录启动。

内部音频格式固定为 16kHz、单声道、PCM16，不可配置。

## 启动

```powershell
python -m app.main
```

启动后会打印版本和当前生效的配置；配置不合法时打印错误并以退出码 2 结束。

## 测试

```powershell
python -m pytest -v
```

使用 `python -m pytest` 而不是 `pytest`，这样仓库根目录会自动加入 `sys.path`，测试才能导入 `app` 包。
