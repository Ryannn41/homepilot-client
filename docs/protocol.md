# HomePilot 语音 WebSocket 协议 v1

客户端（homepilot-client，家庭 Windows PC）与服务端（homepilot-agent，云端）之间的实时语音协议。客户端实现见 `app/protocol.py`、`app/client.py`，参考服务端见 `tools/mock_server.py`。

## 1. 连接

- 地址：`ws://<host>:<port>/ws/voice/`（生产环境建议 `wss://`），由客户端 `HOMEPILOT_SERVER_URL` 配置。
- 请求头：
  - `Authorization: Bearer <token>`：配置了令牌时携带。服务端认证失败应在握手阶段返回 HTTP 401/403，客户端会按退避策略重试。
  - `X-Device-Id: <device_id>`：设备标识。
- 不使用 permessage-deflate 压缩。
- 心跳：客户端每 20 秒发送 WebSocket ping，20 秒内未收到 pong 视为断线。服务端无需额外实现应用层心跳。
- 单条消息最大 1 MiB。
- 客户端常驻长连接，断线后自动重连：等待 1、2、4……秒，最长 30 秒，每次加 ±20% 随机抖动。

文本消息（text frame）为 JSON 事件，二进制消息（binary frame）为音频帧。

## 2. 标识

| 标识 | 生成方 | 格式 | 生命周期 |
| --- | --- | --- | --- |
| `session_id` | 客户端 | UUID4 | 一次会话（唤醒到退出对话）；断线即作废 |
| `turn_id` | 客户端 | UUID4 | 一轮用户输入及其响应 |
| `seq` | 发送方 | uint32 | 每个 turn 内按方向各自从 0 递增 |

JSON 中 UUID 为 36 字符小写带连字符字符串，如 `"77dcc792-312f-4c1e-a2cd-ba263a747646"`。

## 3. 音频格式

所有音频（上行麦克风和下行 TTS）统一为 **16000 Hz、单声道、PCM16 小端**。客户端按 20ms 一帧发送（320 采样点、640 字节）；服务端下行帧长不限，但必须是完整采样（偶数字节），建议 20~100ms。

## 4. 二进制音频帧

固定 24 字节头部 + PCM 数据，整数字段均为小端：

| 偏移 | 长度 | 字段 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 0 | 1 | `version` | uint8 | 当前为 `1` |
| 1 | 1 | `kind` | uint8 | `1` 麦克风音频（上行），`2` TTS 音频（下行） |
| 2 | 2 | `reserved` | uint16 | 必须为 `0` |
| 4 | 16 | `turn_id` | bytes | `uuid.UUID.bytes`，RFC 4122 网络字节序（大端） |
| 20 | 4 | `seq` | uint32 | 小端 |
| 24 | N | PCM | bytes | PCM16 小端，N 为偶数，可以为 0 |

Python：`struct.Struct("<BBH16sI")`。

```python
header = struct.pack("<BBH16sI", 1, 2, 0, turn_id.bytes, seq)
version, kind, reserved, turn, seq = struct.unpack_from("<BBH16sI", data)
turn_id = uuid.UUID(bytes=turn)
```

接收方遇到版本未知、`kind` 未知、`reserved` 非 0、长度不足 24 字节或 PCM 长度为奇数的帧，丢弃并记录日志，不断开连接。

## 5. JSON 事件

每个事件都有 `type` 和 `session_id`；与某一轮相关的事件还有 `turn_id`。未列出的额外字段接收方应忽略。

### 5.1 客户端 → 服务端

| type | 字段 | 说明 |
| --- | --- | --- |
| `session.start` | `device_id`、`client_version` | 开始会话（唤醒后发送） |
| `input.start` | `turn_id`、`sample_rate`=16000、`channels`=1、`format`=`"pcm16"` | 开始一轮语音输入，随后发送该 turn 的麦克风音频帧 |
| `input.end` | `turn_id`、`frames`、`truncated` | 本轮输入结束。`frames` 为已发送的音频帧数，服务端可据此判断是否丢帧；`truncated` 为 true 表示达到最长时长被截断 |
| `turn.cancel` | `turn_id`、`reason` | 放弃本轮（如响应超时），服务端应停止生成并不再发送该 turn 的数据 |
| `session.end` | `reason` | 结束会话（如静默超时） |

### 5.2 服务端 → 客户端

| type | 必需字段 | 说明 |
| --- | --- | --- |
| `transcript` | `turn_id`、`text`（string）、`final`（bool） | 语音识别结果，可先发中间结果再发最终结果 |
| `response.text` | `turn_id`、`text`（string） | 回复文本 |
| `response.audio.start` | `turn_id`、`sample_rate`（必须为 16000） | 开始发送 TTS 音频，下行 `seq` 从 0 开始 |
| `response.audio.end` | `turn_id`，可选 `frames` | 该段 TTS 音频发送完毕 |
| `response.done` | `turn_id` | 本轮回复全部结束（文本、所有语音段）。**每一轮必须发送**，包括只有文本、没有语音的回复 |
| `error` | `code`、`message`，可选 `session_id`、`turn_id` | 错误。与会话无关的错误可以不带 `session_id` |

客户端对字段缺失、类型错误、`type` 未知或 `sample_rate` 不是 16000 的事件，丢弃并记录日志。

## 6. 时序

```
客户端                                   服务端
  | session.start {session_id}             |
  |--------------------------------------->|
  | input.start {turn_id}                  |
  |--------------------------------------->|
  | [kind=1 turn_id seq=0] 预缓冲帧         |
  | [kind=1 turn_id seq=1..] 实时帧         |
  |--------------------------------------->|
  | input.end {turn_id, frames}            |
  |--------------------------------------->|
  |                  transcript {turn_id}  |
  |              response.text {turn_id}   |
  |       response.audio.start {turn_id}   |
  |        [kind=2 turn_id seq=0..]        |
  |         response.audio.end {turn_id}   |
  |              response.done {turn_id}   |
  |<---------------------------------------|
  | input.start {新 turn_id}  ...  (连续对话) |
  | session.end                            |
  |--------------------------------------->|
```

- 检测到用户开始说话时，客户端先发送预缓冲中的帧（约 500ms，包含语音开头），再发送实时帧。两部分属于同一 turn，`seq` 连续。
- 客户端半双工：播放 TTS 期间不上传麦克风音频。
- 同一 turn 内可以有多段 `response.audio.start` … `response.audio.end`，每段的下行 `seq` 都从 0 开始；客户端把它们连续播放。
- 客户端收到 `response.done` 并播放完所有语音后，进入连续对话（追问）状态。如果 `input.end` 后 20 秒内没有收到回复，或 `response.audio.end` 后 20 秒内没有收到 `response.done`，客户端分别发送 `turn.cancel` 或按回复结束处理。
- 服务端对某一轮发送 `error` 后，客户端发送 `turn.cancel` 并进入追问状态。

## 7. 接收方规则

客户端：

- 只接受 `session_id` 等于当前会话、`turn_id` 等于当前轮次的事件和音频；其余视为过期消息丢弃。发出新的 `input.start`、`turn.cancel` 或 `session.end` 后，旧轮次的所有消息都会被丢弃。
- 下行音频 `seq` 小于或等于上一帧的视为重复或乱序，丢弃；出现跳号时记录缺口，但照常播放后续帧。
- 下行音频 `kind` 必须为 2。

服务端建议：

- 只处理当前活动 turn 的音频；收到新的 `input.start` 时，放弃此前未完成的 turn。
- 校验上行 `seq` 连续，并与 `input.end.frames` 对照，发现丢帧时记录日志。
- 收到 `turn.cancel` 后停止该 turn 的识别、生成和 TTS 发送。

## 8. 断线

- 断线后当前会话和轮次立即作废；客户端不缓存、不补发断线期间的音频。
- 重连后，客户端需要发送新的 `session.start`（新的 `session_id`）才能继续对话；协议 v1 不支持恢复旧会话。
- 服务端在连接关闭时应清理该连接上的会话状态。
