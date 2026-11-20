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

## 音频设备检查

```powershell
python -m app.main --list-devices   # 按 Host API 分组列出设备（首选 HOMEPILOT_AUDIO_HOSTAPI 排最前）
python -m app.main --probe          # 探测每个设备在 16000/44100/48000 Hz、单/双声道下是否可用
```

`--list-devices` 每行依次为：设备索引、输入/输出通道数、默认采样率、该 Host API 下的默认设备标记、设备名。

`--probe` 使用 `sounddevice.check_input_settings` / `check_output_settings`（PCM16）逐项检查，结果分三类：

- **支持**：PortAudio 接受该格式。
- **不支持**：PortAudio 明确拒绝（采样率、通道数或采样格式无效）。
- **探测失败**：其他主机错误或驱动异常，括号中给出原因。

注意：这一步只检查格式是否被接受，不实际打开音频流；最终能否正常采集和播放，在后续录音、播放测试中验证。WASAPI 共享模式通常只接受系统混音采样率（常见为 48000 Hz），因此客户端会在本地重采样到 16kHz。

## 录音测试

```powershell
python -m app.main --record-test 5                       # 录 5 秒，保存为 record_test.wav
python -m app.main --record-test 5 --output D:\tmp\a.wav
```

使用 `HOMEPILOT_INPUT_DEVICE` 指定的设备（留空为 `HOMEPILOT_AUDIO_HOSTAPI` 下的默认输入），按设备实际接受的原生采样率采集，转换为 16kHz 单声道 PCM16，每秒打印一次 RMS（dBFS）和峰值。结束时打印帧数、丢帧和输入溢出次数。WAV 文件已被 `.gitignore` 忽略。

`AudioCapture` 的队列默认容纳 50 帧（1 秒），满时丢弃最旧的帧并计数。

## 播放测试

```powershell
python -m app.main --play-test 5                          # 播放 5 秒 440 Hz 测试音（约 -20 dBFS）
$env:HOMEPILOT_OUTPUT_DEVICE="HyperX"; python -m app.main --play-test 5
```

`AudioPlayer` 接收 16kHz 单声道 PCM16（`bytes` 或一维 `np.int16`），按设备实际能打开的格式播放：采样率依次尝试设备默认值、48000、44100、16000，声道优先双声道（单声道复制到左右），每个候选先经 `check_output_settings` 过滤，再实际打开并启动输出流验证，失败则换下一个。

```python
async with AudioPlayer(settings) as player:
    await player.write(chunk)     # 可多次写入，缓冲超过 500ms 时异步等待
    done = await player.drain()   # 输出尾部并等待播完；被 stop() 打断返回 False
# 或一次性：await AudioPlayer(settings).play(pcm)
```

`stop()` 立即中止并释放输出流，可重复调用；异常、任务取消时同样会释放。

## 重采样

`app/audio/resample.py` 中的 `StreamResampler` 把连续的单声道 PCM16（一维 `np.int16`）从设备采样率流式转换为 16kHz：

```python
r = StreamResampler(48000, 16000)
for block in blocks:          # 任意长度的连续 int16 块
    out = r.process(block)    # 返回当前可确定的输出，可能为空
out_tail = r.flush()          # 流结束时取出尾部，之后不能再 process
```

分块处理的结果与对整段音频调用 `scipy.signal.resample_poly` 一致（误差不超过 1 LSB）；采样率之比化简后分子或分母超过 1000 时会拒绝。

## 测试

```powershell
python -m pytest -v
```

使用 `python -m pytest` 而不是 `pytest`，这样仓库根目录会自动加入 `sys.path`，测试才能导入 `app` 包。
