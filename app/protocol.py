"""HomePilot 语音 WebSocket 协议：二进制音频帧与 JSON 事件，详见 docs/protocol.md。"""

import json
import struct
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import SAMPLE_RATE

PROTOCOL_VERSION = 1
HEADER = struct.Struct("<BBH16sI")  # version, kind, reserved, turn_id, seq
KIND_MIC = 1
KIND_TTS = 2
MAX_SEQ = 2**32 - 1


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class AudioFrame:
    kind: int
    turn_id: uuid.UUID
    seq: int
    pcm: bytes


@dataclass(frozen=True)
class ServerEvent:
    type: str
    session_id: uuid.UUID | None
    turn_id: uuid.UUID | None
    data: dict[str, Any]


def pack_audio(kind: int, turn_id: uuid.UUID, seq: int, pcm: bytes) -> bytes:
    if kind not in (KIND_MIC, KIND_TTS):
        raise ValueError(f"未知音频类型: {kind}")
    if not isinstance(turn_id, uuid.UUID):
        raise TypeError("turn_id 必须是 uuid.UUID")
    if not 0 <= seq <= MAX_SEQ:
        raise ValueError(f"seq 超出 uint32 范围: {seq}")
    if not isinstance(pcm, (bytes, bytearray)) or len(pcm) % 2:
        raise ValueError("PCM16 数据必须是偶数长度的 bytes")
    return HEADER.pack(PROTOCOL_VERSION, kind, 0, turn_id.bytes, seq) + bytes(pcm)


def unpack_audio(data: bytes) -> AudioFrame:
    if len(data) < HEADER.size:
        raise ProtocolError(f"音频帧过短: {len(data)} 字节")
    version, kind, reserved, turn, seq = HEADER.unpack_from(data)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"不支持的协议版本: {version}")
    if kind not in (KIND_MIC, KIND_TTS):
        raise ProtocolError(f"未知音频类型: {kind}")
    if reserved != 0:
        raise ProtocolError("保留字段必须为 0")
    pcm = bytes(data[HEADER.size :])
    if len(pcm) % 2:
        raise ProtocolError("PCM16 数据长度必须为偶数")
    return AudioFrame(kind, uuid.UUID(bytes=turn), seq, pcm)


# ---- 客户端 -> 服务端 JSON 事件 ----

def _event(type_: str, session_id: uuid.UUID, **fields: Any) -> str:
    msg: dict[str, Any] = {"type": type_, "session_id": str(session_id)}
    for key, value in fields.items():
        msg[key] = str(value) if isinstance(value, uuid.UUID) else value
    return json.dumps(msg, ensure_ascii=False)


def session_start(session_id: uuid.UUID, device_id: str, client_version: str) -> str:
    return _event("session.start", session_id, device_id=device_id, client_version=client_version)


def input_start(session_id: uuid.UUID, turn_id: uuid.UUID) -> str:
    return _event("input.start", session_id, turn_id=turn_id, sample_rate=SAMPLE_RATE, channels=1, format="pcm16")


def input_end(session_id: uuid.UUID, turn_id: uuid.UUID, frames: int, truncated: bool = False) -> str:
    return _event("input.end", session_id, turn_id=turn_id, frames=frames, truncated=truncated)


def turn_cancel(session_id: uuid.UUID, turn_id: uuid.UUID, reason: str) -> str:
    return _event("turn.cancel", session_id, turn_id=turn_id, reason=reason)


def session_end(session_id: uuid.UUID, reason: str) -> str:
    return _event("session.end", session_id, reason=reason)


# ---- 服务端 -> 客户端 JSON 事件 ----

# 事件类型 -> 必需字段及类型（session_id、turn_id 另行校验）
SERVER_EVENTS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "transcript": {"text": str, "final": bool},
    "response.text": {"text": str},
    "response.audio.start": {"sample_rate": int},
    "response.audio.end": {},
    "response.done": {},
    "error": {"code": str, "message": str},
}
_TURN_OPTIONAL = {"error"}


def _parse_uuid(msg: dict, key: str, required: bool) -> uuid.UUID | None:
    value = msg.get(key)
    if value is None:
        if required:
            raise ProtocolError(f"缺少字段 {key}")
        return None
    try:
        return uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        raise ProtocolError(f"{key} 不是合法的 UUID: {value!r}") from None


def decode_event(text: str) -> ServerEvent:
    try:
        msg = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"JSON 解析失败: {e}") from None
    if not isinstance(msg, dict):
        raise ProtocolError("事件必须是 JSON 对象")
    type_ = msg.get("type")
    if type_ not in SERVER_EVENTS:
        raise ProtocolError(f"未知事件类型: {type_!r}")
    optional = type_ in _TURN_OPTIONAL
    session_id = _parse_uuid(msg, "session_id", required=not optional)
    turn_id = _parse_uuid(msg, "turn_id", required=not optional)
    for key, expected in SERVER_EVENTS[type_].items():
        value = msg.get(key)
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ProtocolError(f"{type_} 字段 {key} 缺失或类型错误: {value!r}")
    if type_ == "response.audio.start" and msg["sample_rate"] != SAMPLE_RATE:
        raise ProtocolError(f"不支持的 TTS 采样率: {msg['sample_rate']}")
    return ServerEvent(type_, session_id, turn_id, msg)
