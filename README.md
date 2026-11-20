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
| sherpa-onnx | 本地唤醒词检测（KWS）和语音活动检测（VAD） |
| websockets | 与云端的 WebSocket 通信 |
| psutil | 运行统计中的内存和 CPU 占用 |
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
| `HOMEPILOT_RECONNECT_MIN_S` | `1` | 断线重连的首次等待秒数，之后翻倍 |
| `HOMEPILOT_RECONNECT_MAX_S` | `30` | 断线重连的最长等待秒数 |
| `HOMEPILOT_PING_INTERVAL_S` | `20` | WebSocket 心跳间隔，同时作为心跳超时 |
| `HOMEPILOT_INPUT_DEVICE` | 空（系统默认） | 输入设备索引或名称子串 |
| `HOMEPILOT_OUTPUT_DEVICE` | 空（系统默认） | 输出设备索引或名称子串 |
| `HOMEPILOT_AUDIO_HOSTAPI` | `wasapi` | 优先使用的 Host API：`wasapi` / `mme` / `directsound` |
| `HOMEPILOT_FRAME_MS` | `20` | 音频分片时长，只能是 20 或 40 毫秒 |
| `HOMEPILOT_SILENCE_TIMEOUT_S` | `15` | 唤醒后或追问时多少秒没人说话就结束会话 |
| `HOMEPILOT_RESPONSE_TIMEOUT_S` | `20` | 说完后多少秒内没有收到回复就取消本轮 |
| `HOMEPILOT_PLAYBACK_TAIL_MS` | `300` | 回复播放完后再等多少毫秒才开始听 |
| `HOMEPILOT_WAKE_SOUND` | `true` | 唤醒时播放「我在」应答 |
| `HOMEPILOT_SILENCE_FAREWELL_SOUND` | `true` | 静默超时结束时播放「有需要再叫我」 |
| `HOMEPILOT_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `HOMEPILOT_LOG_DIR` | `logs` | 日志文件目录，每天零点轮转；留空只输出到控制台 |
| `HOMEPILOT_LOG_KEEP_DAYS` | `14` | 日志文件保留天数 |
| `HOMEPILOT_STATS_INTERVAL_S` | `3600` | 运行统计写入日志的间隔（秒） |

优先级：系统环境变量 > `.env` > 默认值。`.env` 从当前工作目录读取，请在仓库根目录启动。

内部音频格式固定为 16kHz、单声道、PCM16，不可配置。

## 启动

```powershell
python -m app.main
```

启动后打印版本和当前生效的配置，然后连接服务端、打开麦克风，等待唤醒词「你好小余」。Ctrl+C 退出（有进行中的会话时会先发送 `session.end`）。配置不合法时打印错误并以退出码 2 结束；模型文件缺失、设备打不开时以退出码 1 结束。

服务端尚未上线时，先在另一个窗口运行 `python -m tools.mock_server`（见「WebSocket 通信 / 联调」）。

根目录的两个脚本只覆盖服务端地址，其余配置仍取自 `.env`，额外参数原样传给 `app.main`：

```powershell
.\start-local.cmd                # 推荐：不受执行策略限制（含 \\wsl.localhost\... 路径）
.\start-cloud.cmd
.\start-cloud.cmd --ws-test 3    # 带参数
# 若已设置执行策略，也可直接 .\start-local.ps1 / .\start-cloud.ps1
```

需要先 `conda activate homepilot_client`。`.cmd` 会调用 `powershell -ExecutionPolicy Bypass`，无需改系统策略；仍要用 `.ps1` 时，可执行一次 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。云端开启令牌校验时，在 `.env` 中配置与服务端一致的 `HOMEPILOT_AUTH_TOKEN`。

## 对话流程

`app/conversation.py` 中的 `Conversation` 按状态决定每一帧麦克风音频的去向（半双工）：

| 状态 | 麦克风 | 离开条件 |
| --- | --- | --- |
| `IDLE` 等待唤醒 | 只送唤醒词检测 | 唤醒：开始会话、播放「我在」（期间忽略麦克风）→ `LISTENING`；未连接服务端时忽略唤醒 |
| `LISTENING` 聆听 | 送 VAD（含预缓冲）；开始说话后同时上传 | 说完 → `THINKING`；`SILENCE_TIMEOUT_S` 内没开口 → 结束会话、播放「有需要再叫我」、`IDLE` |
| `THINKING` 等待回复 | 丢弃 | `response.audio.start` → `SPEAKING`；只有文本时 `response.done` → `FOLLOW_UP`；`RESPONSE_TIMEOUT_S` 超时 → `turn.cancel`、`FOLLOW_UP` |
| `SPEAKING` 播放回复 | 丢弃 | 收到 `response.done` 且播放完毕，再等 `PLAYBACK_TAIL_MS` → `FOLLOW_UP` |
| `FOLLOW_UP` 追问 | 同 `LISTENING`，无需唤醒词 | 开口 → 新一轮 `LISTENING`；`SILENCE_TIMEOUT_S` 内没开口 → 结束会话、播放「有需要再叫我」、`IDLE` |

任何状态下断线都立即停止播放并回到 `IDLE`；服务端 `error` 会结束当前轮并进入 `FOLLOW_UP`。

## 定时提醒

服务端到点后会主动推送提醒（`notify.start`、若干音频帧、`notify.end`，见 [docs/protocol.md](docs/protocol.md)）。提醒不属于任何会话，客户端在任何状态下都会接收，收完整条后再播放：

| 收到提醒时的状态 | 处理 |
| --- | --- |
| `IDLE` | 先发 `session.start` 开启会话，再播放 |
| `LISTENING` / `FOLLOW_UP`，用户还没开口 | 直接在当前会话中播放 |
| 用户正在说话、`THINKING`、`SPEAKING`，或正在播唤醒应答 | 排队，等这一轮回复播完进入 `FOLLOW_UP` 后再播 |

播放时先放提示音，再放提醒语音；播完进入 `FOLLOW_UP`，用户不说唤醒词就能直接回应（如「好的知道了」）。没人回应、静默超时回到 `IDLE` 时，不计为疑似误唤醒。多条提醒按到达顺序逐条播放。

断线时，没收完的提醒直接丢弃；已收完的提醒仍会播放，但播完后没有会话，需要唤醒才能继续对话。正在播放的提醒会随断线一起停止。

手动联调：

```powershell
# 窗口 1：每 60 秒推送一条模拟提醒
python -m tools.mock_server --notify-every 60
# 窗口 2
python -m app.main
```

## 长期运行与故障恢复

`python -m app.main` 设计为常驻运行：

- **音频故障自动恢复**：采集 2 秒没有数据、PortAudio 报错、找不到设备、播放设备打不开，或麦克风连续 10 秒输出全零（真实麦克风总有底噪，全零通常意味着设备失效或被静音）时，结束当前会话、关闭音频，重新初始化 PortAudio 并重新查找设备，然后回到等待唤醒。重试间隔 1、2、5、10 秒，之后每 10 秒一次；音频正常运行 60 秒后间隔重新从 1 秒开始。拔出耳机再插回，或者开机时设备还没就绪，都会自动恢复。唤醒词和 VAD 模型、WebSocket 连接不受影响。
- **其他异常**：状态机中未预料的异常会连同调用栈写入日志，然后按同样的方式恢复，不会退出程序。只有启动时模型文件缺失、配置错误才会退出。
- **日志文件**：同时写入 `logs/homepilot.log`（UTF-8），每天零点轮转，保留 `HOMEPILOT_LOG_KEEP_DAYS` 天。
- **运行统计**：每隔 `HOMEPILOT_STATS_INTERVAL_S` 秒及退出时写一行，例如：

```
运行统计: 运行 24.00 h, 唤醒 37 次, 对话 52 轮, 疑似误唤醒 2 次, 断线 0 次, 音频重启 1 次, 丢帧 0, 内存 152.3 MB, CPU 1.2%
```

「疑似误唤醒」指唤醒后一句话没说就因静默超时结束的会话；数量偏多时调高 `HOMEPILOT_KWS_KEYWORDS_THRESHOLD`。CPU 为占全部核心的百分比。状态变化、识别结果和回复文本都会打印在日志中。所有状态变化在同一个事件循环中按顺序处理；播放在独立任务中进行，不阻塞状态机。

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

## 唤醒词（KWS）

唤醒词为「你好，小余」，使用 [sherpa-onnx 开放词表 KWS](https://k2-fsa.github.io/sherpa/onnx/kws/index.html)，无需训练。

### 下载模型

模型不入库，解压到 `models/`（已被 `.gitignore` 忽略）。默认使用中文模型 wenetspeech-3.3M：

```powershell
cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
```

备选中英文模型 zh-en-3M（chunk-8 延迟约 160ms，默认模型约 320ms），下载后在 `.env` 中切换：

```powershell
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
tar xf sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
```

```ini
HOMEPILOT_KWS_MODEL_DIR=models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20
HOMEPILOT_KWS_ENCODER=encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx
HOMEPILOT_KWS_DECODER=decoder-epoch-13-avg-2-chunk-8-left-64.onnx
HOMEPILOT_KWS_JOINER=joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx
```

两个模型的「你好小余」token 序列相同，共用仓库根目录的 `keywords.txt`。

### 关键词文件

`keywords.txt` 每行一个关键词：声母/韵母拼音 token，`@` 后为原文（不能含空格），可追加 `:加分` 和 `#阈值` 单独调节：

```
n ǐ h ǎo x iǎo y ú @你好小余
```

修改唤醒词时用 sherpa-onnx 自带工具生成（需额外 `pip install pypinyin sentencepiece`，仅生成时需要）：

```powershell
python -c "import sherpa_onnx; print(sherpa_onnx.text2token(['你好小余'], tokens='models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt', tokens_type='ppinyin'))"
```

启动时会校验关键词中的 token 都在模型 `tokens.txt` 中。

### 测试

```powershell
python -m app.main --kws-test        # 持续检测，Ctrl+C 退出
python -m app.main --kws-test 30     # 检测 30 秒
```

检测到唤醒词时打印 `>>> 唤醒: 你好小余  帧 #N  t=...s`，每 3 秒打印一次麦克风电平。误唤醒多时调高 `HOMEPILOT_KWS_KEYWORDS_THRESHOLD`，漏检多时调低。

## 语音活动检测（VAD）与预缓冲

使用 sherpa-onnx 内置的 [Silero VAD](https://github.com/snakers4/silero-vad)，直接处理 `AudioCapture` 输出的 16kHz PCM16 帧，不再重采样。

### 下载模型

```powershell
cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

约 630 KB，放在 `models/` 下（已被 `.gitignore` 忽略）。

### 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOMEPILOT_VAD_MODEL` | `models/silero_vad.onnx` | 模型路径 |
| `HOMEPILOT_VAD_THRESHOLD` | `0.5` | 语音概率阈值（0~1），噪声误触发多时调高 |
| `HOMEPILOT_VAD_MIN_SPEECH_MS` | `250` | 连续语音达到该时长才判定开始 |
| `HOMEPILOT_VAD_MIN_SILENCE_MS` | `700` | 静音持续该时长才判定结束，不会因单个静音窗口提前结束 |
| `HOMEPILOT_VAD_MAX_SPEECH_S` | `30` | 单个片段最长时长，超过后强制结束 |
| `HOMEPILOT_PREROLL_MS` | `500` | 预缓冲时长（200~1000），20ms 分片时为 25 帧 |

### 工作方式

```python
with VADDetector(settings) as vad:
    for event in vad.process(frame):   # 每次一帧 20ms / 640 字节
        ...                            # SpeechStart / SpeechAudio / SpeechEnd
```

- 未说话时每帧进入有界预缓冲（`deque`，满时淘汰最旧帧）。
- VAD 判定开始时返回 `SpeechStart`，`frames` 为预缓冲中的帧（含当前帧），随后清空预缓冲；之后每帧返回 `SpeechAudio`。每一帧只交付一次，上层可以边收边上传。
- 静音持续 `VAD_MIN_SILENCE_MS` 后返回 `SpeechEnd`，`frames` 为完整片段（含预缓冲和结尾静音），附起止帧序号和时长。
- 达到最长时长时返回 `truncated=True` 的 `SpeechEnd`，并等待用户停顿后才开始新片段，避免把截断后的语音混入下一段。
- `reset()` 丢弃预缓冲、未结束的片段和模型状态；`close()` 后再调用 `process()` 会抛 `RuntimeError`。

实测（Windows，真实模型）：VAD 判定开始比实际发声晚约 300ms，主要来自 `VAD_MIN_SPEECH_MS`，因此预缓冲需大于 `VAD_MIN_SPEECH_MS` + 约 60ms；默认 500ms 下片段会在发声前额外保留约 150~200ms。单帧推理平均约 0.06ms（实时采集时约 0.2ms），远小于 20ms 帧长，因此直接在事件循环中同步调用，不使用线程。

### 测试

```powershell
python -m app.main --vad-test                          # 持续检测，Ctrl+C 退出
python -m app.main --vad-test 60 --vad-save vad_segments   # 检测 60 秒并把每个片段保存为 WAV
```

开始时打印 `>>> 语音开始`（判定时间、片段起点、预缓冲帧数），结束时打印 `<<< 语音结束`（时长、帧数、电平），退出时打印片段数量、总时长和 VAD 平均/最大耗时。保存的 WAV 可用来检查开头和结尾是否完整（`*.wav` 已被 `.gitignore` 忽略）。

## WebSocket 通信

协议说明见 [docs/protocol.md](docs/protocol.md)：24 字节头部（版本、类型、turn_id、seq）+ 16kHz PCM16 的二进制音频帧，加上 `session.start` / `input.start` / `input.end` / `transcript` / `response.*` 等 JSON 事件。

`app/client.py` 中的 `VoiceClient` 启动后常驻长连接，断线自动重连：

```python
async with VoiceClient(settings) as client:
    await client.start_session()         # 生成 session_id
    await client.start_input()           # 生成 turn_id，seq 从 0 开始
    await client.send_audio(frame)       # 自动编号 seq；离线时返回 False 并丢弃
    await client.end_input()
    item = await client.recv()           # ServerEvent / AudioFrame / Connected / Disconnected
```

- 只放行当前会话、当前轮次的事件和 TTS 音频；重复或乱序的帧丢弃，跳号记为缺口。
- 断线时发出 `Disconnected`（附被作废的 turn_id），会话随之作废，重连后需重新 `start_session()`。
- 连接时不走系统代理，也不启用 permessage-deflate 压缩。

### 联调

服务端尚未上线时，用仓库自带的模拟服务端：收到一句话后回一段模拟识别文本，并把这句话原样作为 TTS 发回。

```powershell
# 窗口 1：启动模拟服务端（默认 127.0.0.1:8000，与 HOMEPILOT_SERVER_URL 默认值一致）
python -m tools.mock_server
# 窗口 2：说一句话 -> VAD -> 上传 -> 接收并播放响应
python -m app.main --ws-test 3     # 3 轮，省略轮数则持续到 Ctrl+C
```

每轮打印 turn_id、上传帧数、收到的事件及相对 `input.end` 的耗时、播放帧数，结束时打印上传、离线丢弃、过期丢弃、重复、缺帧和非法消息的统计。运行中关闭再启动模拟服务端，可以验证自动重连。`--ws-test` 只是联调工具，不包含唤醒词和对话状态机。

模拟服务端选项：`--port`、`--token xxx`（要求客户端携带对应的 `HOMEPILOT_AUTH_TOKEN`）、`--realtime`（按实时速度发送 TTS，默认尽快发送）、`--notify-every S`（客户端连上后每隔 S 秒推送一条模拟提醒，语音为三声提示音，见「定时提醒」）。

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
