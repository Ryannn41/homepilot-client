import asyncio
import json
import os
import struct
import threading
import uuid

import numpy as np
import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from app import main as main_module
from app.audio.vad import SpeechAudio, SpeechEnd, SpeechStart
from app.client import Connected, Disconnected, VoiceClient, next_delay
from app.config import Settings
from app.main import main
from app.protocol import KIND_MIC, KIND_TTS, AudioFrame, ServerEvent, pack_audio
from tools.mock_server import MockServer


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HOMEPILOT_"):
            monkeypatch.delenv(key)


def make_settings(url, **kw):
    return Settings(server_url=url, reconnect_min_s=0.05, reconnect_max_s=0.2, **kw)


def pcm(i: int) -> bytes:
    return np.full(320, i, dtype=np.int16).tobytes()


async def expect(client, cls, timeout=2.0):
    """读取下一个 cls 类型的消息，跳过其他类型。"""
    async def loop():
        while True:
            item = await client.recv()
            if isinstance(item, cls):
                return item
    return await asyncio.wait_for(loop(), timeout)


async def drain(client, timeout=0.3) -> list:
    items = []
    try:
        while True:
            items.append(await asyncio.wait_for(client.recv(), timeout))
    except TimeoutError:
        return items


class RawServer:
    """测试用服务端：记录客户端消息，由测试代码任意发送消息。"""

    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.ws = None
        self.connected = asyncio.Event()

    async def __aenter__(self):
        async def handler(ws):
            self.ws = ws
            self.connected.set()
            try:
                async for msg in ws:
                    await self.inbox.put(msg)
            except ConnectionClosed:
                pass

        self._server = await serve(handler, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self._server.sockets[0].getsockname()[1]}/ws/voice/"
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    async def next_event(self) -> dict:
        while True:
            msg = await asyncio.wait_for(self.inbox.get(), 2)
            if isinstance(msg, str):
                return json.loads(msg)

    async def send_event(self, type_, session, turn, **fields):
        msg = {"type": type_, "session_id": str(session) if session else None, "turn_id": str(turn) if turn else None}
        await self.ws.send(json.dumps({k: v for k, v in {**msg, **fields}.items() if v is not None}))


# ---- 正常收发 ----

def test_roundtrip_with_mock_server():
    async def run():
        async with MockServer() as srv, VoiceClient(make_settings(srv.url, device_id="dev-a")) as client:
            assert await client.wait_connected(2)
            assert isinstance(await client.recv(), Connected)
            session = await client.start_session()
            turn = await client.start_input()
            assert session and turn and turn != session
            for i in range(10):
                assert await client.send_audio(pcm(i))
            assert await client.end_input()

            items = []
            while True:
                items.append(await asyncio.wait_for(client.recv(), 2))
                if isinstance(items[-1], ServerEvent) and items[-1].type == "response.audio.end":
                    break
            events = [i.type for i in items if isinstance(i, ServerEvent)]
            assert events == ["transcript", "response.text", "response.audio.start", "response.audio.end"]
            frames = [i for i in items if isinstance(i, AudioFrame)]
            assert [f.seq for f in frames] == list(range(10))
            assert all(f.kind == KIND_TTS and f.turn_id == turn for f in frames)
            assert [f.pcm for f in frames] == [pcm(i) for i in range(10)]

            assert [e["type"] for e in srv.events] == ["session.start", "input.start", "input.end"]
            assert srv.events[0]["device_id"] == "dev-a" and srv.events[2]["frames"] == 10
            mic = srv.frames[str(turn)]
            assert [f.seq for f in mic] == list(range(10)) and all(f.kind == KIND_MIC for f in mic)
            assert srv.headers[0]["X-Device-Id"] == "dev-a" and "Authorization" not in srv.headers[0]
            assert client.sent_frames == 10

    asyncio.run(run())


def test_seq_restarts_for_each_turn():
    async def run():
        async with MockServer() as srv, VoiceClient(make_settings(srv.url)) as client:
            await client.wait_connected(2)
            await client.start_session()
            turns = []
            for _ in range(2):
                turns.append(await client.start_input())
                for i in range(3):
                    await client.send_audio(pcm(i))
                await client.end_input()
            await asyncio.sleep(0.2)
            assert turns[0] != turns[1]
            for t in turns:
                assert [f.seq for f in srv.frames[str(t)]] == [0, 1, 2]

    asyncio.run(run())


# ---- 认证 ----

def test_bearer_token_sent_and_rejected():
    async def run():
        async with MockServer(token="secret") as srv:
            async with VoiceClient(make_settings(srv.url, auth_token="secret")) as good:
                assert await good.wait_connected(2)
                assert srv.headers[0]["Authorization"] == "Bearer secret"
            async with VoiceClient(make_settings(srv.url, auth_token="wrong")) as bad:
                assert not await bad.wait_connected(0.5)
                assert srv.rejected >= 2  # 被拒后仍按退避持续重试
                assert await bad.start_session() is None

    asyncio.run(run())


# ---- 离线与重连 ----

def test_offline_sends_return_false():
    async def run():
        async with VoiceClient(make_settings("ws://127.0.0.1:1/ws/voice/")) as client:
            assert not await client.wait_connected(0.3)
            assert await client.start_session() is None
            assert await client.start_input() is None
            assert not await client.send_audio(pcm(0))
            assert not await client.end_input()
            assert not await client.cancel_turn("x")
            assert not await client.end_session("x")
            assert client.dropped_offline == 1 and client.sent_frames == 0

    asyncio.run(run())


def test_reconnects_after_server_restart():
    async def run():
        srv = MockServer()
        await srv.start()
        port = srv.port
        async with VoiceClient(make_settings(srv.url)) as client:
            await expect(client, Connected)
            await client.start_session()
            turn = await client.start_input()
            await client.send_audio(pcm(1))
            await srv.stop()
            gone = await expect(client, Disconnected)
            assert gone.aborted_turn == turn
            assert client.session_id is None and client.turn_id is None and not client.connected
            assert not await client.send_audio(pcm(2))  # 断线后上传作废，不缓存

            srv2 = MockServer(port=port)
            await srv2.start()
            try:
                await expect(client, Connected, timeout=3)
                assert await client.start_session() is not None
                for _ in range(50):
                    if srv2.events:
                        break
                    await asyncio.sleep(0.02)
                assert srv2.events[0]["type"] == "session.start"
            finally:
                await srv2.stop()

    asyncio.run(run())


def test_server_close_reason_reported():
    async def run():
        async with RawServer() as srv, VoiceClient(make_settings(srv.url)) as client:
            await expect(client, Connected)
            await srv.ws.close(4001, "kicked")
            gone = await expect(client, Disconnected)
            assert "4001" in gone.reason and "kicked" in gone.reason

    asyncio.run(run())


def test_stop_is_idempotent_and_stops_reconnecting():
    async def run():
        async with MockServer() as srv:
            client = VoiceClient(make_settings(srv.url))
            await client.start()
            await client.start()
            await client.wait_connected(2)
            await client.stop()
            await client.stop()
            assert not client.connected
            await asyncio.sleep(0.3)
            assert len(srv.headers) == 1

    asyncio.run(run())


def test_backoff_doubles_up_to_max():
    s = Settings(reconnect_min_s=1, reconnect_max_s=10)
    delays, d = [], None
    for _ in range(6):
        d = next_delay(d, s)
        delays.append(d)
    assert delays == [1, 2, 4, 8, 10, 10]


def test_reconnect_config_validated():
    with pytest.raises(ValueError):
        Settings(reconnect_min_s=5, reconnect_max_s=1)
    with pytest.raises(ValueError):
        Settings(ping_interval_s=0)


# ---- 下行过滤 ----

def test_filters_stale_duplicate_gap_and_invalid():
    async def run():
        async with RawServer() as srv, VoiceClient(make_settings(srv.url)) as client:
            await expect(client, Connected)
            session = await client.start_session()
            turn = await client.start_input()
            await srv.next_event()
            await srv.next_event()
            other = uuid.uuid4()

            await srv.send_event("transcript", session, other, text="旧轮次", final=True)  # 过期
            await srv.send_event("transcript", uuid.uuid4(), turn, text="旧会话", final=True)  # 过期
            await srv.ws.send("{bad json")  # 非法
            await srv.send_event("weird", session, turn)  # 非法
            await srv.ws.send(struct.pack("<BBH16sI", 9, 2, 0, turn.bytes, 0))  # 非法版本
            await srv.ws.send(pack_audio(KIND_MIC, turn, 0, pcm(0)))  # 下行不应是麦克风音频
            await srv.send_event("transcript", session, turn, text="你好", final=True)
            await srv.send_event("response.audio.start", session, turn, sample_rate=16000)
            for seq in (0, 1, 1, 0, 4, 5):
                await srv.ws.send(pack_audio(KIND_TTS, turn, seq, pcm(seq)))
            await srv.ws.send(pack_audio(KIND_TTS, other, 6, pcm(6)))  # 过期
            await srv.send_event("error", None, None, code="busy", message="稍后再试")
            await srv.send_event("response.audio.end", session, turn, frames=4)

            items = await drain(client)
            got = [(i.type if isinstance(i, ServerEvent) else i.seq) for i in items]
            assert got == ["transcript", "response.audio.start", 0, 1, 4, 5, "error", "response.audio.end"]
            assert client.dropped_stale == 3
            assert client.dropped_duplicate == 2
            assert client.seq_gaps == 2
            assert client.dropped_invalid == 4
            assert client.connected  # 非法消息不断开连接

    asyncio.run(run())


def test_new_turn_and_cancel_drop_late_responses():
    async def run():
        async with RawServer() as srv, VoiceClient(make_settings(srv.url)) as client:
            await expect(client, Connected)
            session = await client.start_session()
            first = await client.start_input()
            await client.end_input()
            second = await client.start_input()
            await srv.ws.send(pack_audio(KIND_TTS, first, 0, pcm(0)))  # 上一轮迟到的音频
            await srv.send_event("response.text", session, first, text="迟到")
            await srv.ws.send(pack_audio(KIND_TTS, second, 0, pcm(1)))
            items = await drain(client)
            assert [(i.turn_id, i.seq) for i in items] == [(second, 0)]

            assert await client.cancel_turn("user")
            await srv.ws.send(pack_audio(KIND_TTS, second, 1, pcm(2)))
            assert await drain(client) == []
            assert client.dropped_stale == 3
            types = []
            while not srv.inbox.empty():
                msg = srv.inbox.get_nowait()
                if isinstance(msg, str):
                    types.append(json.loads(msg)["type"])
            assert types[-1] == "turn.cancel"

    asyncio.run(run())


def test_tts_seq_resets_on_audio_start():
    async def run():
        async with RawServer() as srv, VoiceClient(make_settings(srv.url)) as client:
            await expect(client, Connected)
            session = await client.start_session()
            turn = await client.start_input()
            for _ in range(2):  # 同一轮内两段 TTS，各自从 0 开始
                await srv.send_event("response.audio.start", session, turn, sample_rate=16000)
                for seq in range(3):
                    await srv.ws.send(pack_audio(KIND_TTS, turn, seq, pcm(seq)))
            items = await drain(client)
            assert [i.seq for i in items if isinstance(i, AudioFrame)] == [0, 1, 2, 0, 1, 2]
            assert client.dropped_duplicate == 0

    asyncio.run(run())


def test_incoming_queue_drops_oldest_when_full():
    async def run():
        client = VoiceClient(make_settings("ws://127.0.0.1:1/"), queue_size=3)
        for i in range(5):
            client._put(Disconnected(str(i), None))
        assert client.dropped_overflow == 2
        assert [(await client.recv()).reason for _ in range(3)] == ["2", "3", "4"]

    asyncio.run(run())


# ---- CLI ----

class FakeCapture:
    def __init__(self, settings):
        self.device = type("D", (), {"index": 21, "name": "HyperX", "hostapi": "Windows WASAPI"})()
        self.samplerate, self.channels = 48000, 1
        self.n = 0

    async def __aenter__(self):
        self.queue = asyncio.Queue()
        return self

    async def __aexit__(self, *exc):
        pass

    async def read(self):
        await asyncio.sleep(0)
        self.n += 1
        return pcm(self.n)


class FakeVAD:
    """第 5 帧开始（预缓冲 3 帧），第 16 帧结束。"""

    def __init__(self, settings):
        self.n, self.buf = 0, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def reset(self):
        self.n, self.buf = 0, []

    def process(self, frame):
        self.n += 1
        self.buf = (self.buf + [frame])[-3:]
        if self.n == 5:
            return [SpeechStart(self.n - 3, self.n, tuple(self.buf), 20)]
        if 5 < self.n < 16:
            return [SpeechAudio(self.n, frame)]
        if self.n == 16:
            return [SpeechAudio(self.n, frame), SpeechEnd(2, 16, (), 20, False)]
        return []


class FakePlayer:
    written: list[bytes] = []

    def __init__(self, settings):
        pass

    def start(self):
        pass

    async def write(self, data):
        FakePlayer.written.append(data)

    async def drain(self):
        return True

    def stop(self):
        pass


def test_ws_test_cli_roundtrip(monkeypatch, capsys):
    loop = asyncio.new_event_loop()
    srv = MockServer()
    loop.run_until_complete(srv.start())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("HOMEPILOT_SERVER_URL", srv.url)
        monkeypatch.setattr(main_module, "AudioCapture", FakeCapture)
        monkeypatch.setattr(main_module, "VADDetector", FakeVAD)
        monkeypatch.setattr(main_module, "AudioPlayer", FakePlayer)
        FakePlayer.written = []
        assert main(["--ws-test", "2"]) == 0
    finally:
        asyncio.run_coroutine_threadsafe(srv.stop(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
    out = capsys.readouterr().out
    assert out.count(">>> 语音开始") == 2 and out.count("播放完成: 14 帧") == 2
    assert "完成 2 轮, 上传 28 帧" in out
    # 每轮上传：预缓冲第 3~5 帧 + 第 6~16 帧，服务端原样回传并写入播放器
    assert FakePlayer.written[:14] == [pcm(i) for i in range(3, 17)]
    types = [e["type"] for e in srv.events]
    assert types == ["session.start", "input.start", "input.end", "input.start", "input.end", "session.end"]


def test_ws_test_cli_unreachable(monkeypatch, capsys):
    monkeypatch.setenv("HOMEPILOT_SERVER_URL", "ws://127.0.0.1:1/ws/voice/")
    monkeypatch.setattr(main_module.VoiceClient, "wait_connected", lambda self, timeout: _false())
    assert main(["--ws-test", "1"]) == 1
    assert "未能连接服务端" in capsys.readouterr().err


async def _false():
    return False
