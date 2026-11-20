import json
import struct
import uuid

import pytest

from app import protocol
from app.protocol import (
    HEADER,
    KIND_MIC,
    KIND_TTS,
    ProtocolError,
    decode_event,
    pack_audio,
    unpack_audio,
)

TURN = uuid.UUID("00112233-4455-6677-8899-aabbccddeeff")
SESSION = uuid.UUID("12345678-1234-4234-8234-123456789abc")
PCM = bytes(range(256)) * 2 + b"\x01\x02"


# ---- 二进制帧 ----

def test_header_layout():
    data = pack_audio(KIND_MIC, TURN, 0x01020304, PCM)
    assert HEADER.size == 24
    assert data[0] == 1 and data[1] == KIND_MIC
    assert data[2:4] == b"\x00\x00"
    assert data[4:20] == bytes.fromhex("00112233445566778899aabbccddeeff")  # RFC 4122 大端
    assert data[20:24] == b"\x04\x03\x02\x01"  # seq 小端
    assert data[24:] == PCM


@pytest.mark.parametrize("kind", [KIND_MIC, KIND_TTS])
@pytest.mark.parametrize("seq", [0, 1, 2**32 - 1])
def test_roundtrip(kind, seq):
    frame = unpack_audio(pack_audio(kind, TURN, seq, PCM))
    assert (frame.kind, frame.turn_id, frame.seq, frame.pcm) == (kind, TURN, seq, PCM)


def test_empty_pcm_allowed():
    assert unpack_audio(pack_audio(KIND_TTS, TURN, 3, b"")).pcm == b""


@pytest.mark.parametrize("args,exc", [
    ((3, TURN, 0, PCM), ValueError),
    ((KIND_MIC, str(TURN), 0, PCM), TypeError),
    ((KIND_MIC, TURN, -1, PCM), ValueError),
    ((KIND_MIC, TURN, 2**32, PCM), ValueError),
    ((KIND_MIC, TURN, 0, b"\x00"), ValueError),
    ((KIND_MIC, TURN, 0, "pcm"), ValueError),
])
def test_pack_rejects(args, exc):
    with pytest.raises(exc):
        pack_audio(*args)


@pytest.mark.parametrize("data,match", [
    (b"\x01" * 23, "过短"),
    (struct.pack("<BBH16sI", 2, 1, 0, TURN.bytes, 0), "版本"),
    (struct.pack("<BBH16sI", 1, 9, 0, TURN.bytes, 0), "类型"),
    (struct.pack("<BBH16sI", 1, 1, 7, TURN.bytes, 0), "保留"),
    (struct.pack("<BBH16sI", 1, 1, 0, TURN.bytes, 0) + b"\x00", "偶数"),
])
def test_unpack_rejects(data, match):
    with pytest.raises(ProtocolError, match=match):
        unpack_audio(data)


# ---- 客户端事件 ----

def test_client_events():
    assert json.loads(protocol.session_start(SESSION, "dev1", "0.1.0")) == {
        "type": "session.start", "session_id": str(SESSION), "device_id": "dev1", "client_version": "0.1.0"}
    assert json.loads(protocol.input_start(SESSION, TURN)) == {
        "type": "input.start", "session_id": str(SESSION), "turn_id": str(TURN),
        "sample_rate": 16000, "channels": 1, "format": "pcm16"}
    assert json.loads(protocol.input_end(SESSION, TURN, 42, True)) == {
        "type": "input.end", "session_id": str(SESSION), "turn_id": str(TURN), "frames": 42, "truncated": True}
    assert json.loads(protocol.turn_cancel(SESSION, TURN, "timeout"))["reason"] == "timeout"
    assert json.loads(protocol.session_end(SESSION, "idle")) == {
        "type": "session.end", "session_id": str(SESSION), "reason": "idle"}


def test_client_event_ids_are_lowercase_hyphenated():
    msg = json.loads(protocol.input_start(SESSION, uuid.uuid4()))
    assert len(msg["turn_id"]) == 36 and msg["turn_id"] == msg["turn_id"].lower()


# ---- 服务端事件 ----

def server(type_, **fields):
    return json.dumps({"type": type_, "session_id": str(SESSION), "turn_id": str(TURN), **fields})


@pytest.mark.parametrize("text,type_", [
    (server("transcript", text="你好", final=True), "transcript"),
    (server("response.text", text="好的"), "response.text"),
    (server("response.audio.start", sample_rate=16000), "response.audio.start"),
    (server("response.audio.end", frames=10), "response.audio.end"),
    (server("response.audio.end"), "response.audio.end"),
])
def test_decode_valid(text, type_):
    ev = decode_event(text)
    assert ev.type == type_ and ev.session_id == SESSION and ev.turn_id == TURN


def test_decode_error_without_ids():
    ev = decode_event(json.dumps({"type": "error", "code": "auth", "message": "bad token"}))
    assert ev.session_id is None and ev.turn_id is None and ev.data["code"] == "auth"


@pytest.mark.parametrize("text,match", [
    ("{not json", "JSON"),
    ("[1, 2]", "对象"),
    (json.dumps({"type": "unknown"}), "未知事件"),
    (json.dumps({"type": "transcript", "turn_id": str(TURN), "text": "a", "final": True}), "session_id"),
    (json.dumps({"type": "response.text", "session_id": str(SESSION), "text": "a"}), "turn_id"),
    (server("response.text", text="a").replace(str(TURN), "not-a-uuid"), "UUID"),
    (server("transcript", text="a"), "final"),
    (server("transcript", text=1, final=True), "text"),
    (server("response.audio.start", sample_rate=48000), "采样率"),
    (server("response.audio.start", sample_rate=True), "sample_rate"),
    (server("response.audio.start"), "sample_rate"),
    (json.dumps({"type": "error", "code": "x"}), "message"),
])
def test_decode_rejects(text, match):
    with pytest.raises(ProtocolError, match=match):
        decode_event(text)
