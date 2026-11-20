import asyncio
import os
import uuid
from collections import deque

import pytest

from app import conversation as conv_module
from app.audio.vad import SpeechAudio, SpeechEnd, SpeechStart
from app.audio.wakeword import WakeEvent
from app.client import Connected, Disconnected
from app.config import Settings
from app.conversation import Conversation, State
from app.protocol import KIND_TTS, AudioFrame, ServerEvent

QUIET = b"q" * 640
WAKE = b"w" * 640
VOICE = b"v" * 640


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HOMEPILOT_"):
            monkeypatch.delenv(key)
    FakePlayer.instances = []
    FakePlayer.drain_delay = 0.0
    FakePlayer.fail_start = False


class FakeCapture:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()

    async def read(self):
        return await self.queue.get()

    def push(self, *frames):
        for f in frames:
            self.queue.put_nowait(f)


class FakeKWS:
    def __init__(self):
        self.frames, self.resets = [], 0

    async def process(self, frame):
        self.frames.append(frame)
        return WakeEvent("你好小余", len(self.frames) - 1, 0.0, 0.0) if frame == WAKE else None

    def reset(self):
        self.resets += 1


class FakeVAD:
    """连续 2 帧语音判定开始（预缓冲最多 5 帧），语音中连续 3 帧静音判定结束。"""

    def __init__(self):
        self.frames, self.resets = [], 0
        self._clear()

    def _clear(self):
        self.pre, self.voiced, self.silent, self.seg = deque(maxlen=5), 0, 0, None

    def reset(self):
        self.resets += 1
        self._clear()

    def process(self, frame):
        self.frames.append(frame)
        if self.seg is None:
            self.pre.append(frame)
            self.voiced = self.voiced + 1 if frame == VOICE else 0
            if self.voiced < 2:
                return []
            frames = tuple(self.pre)
            self.pre.clear()
            self.seg = list(frames)
            return [SpeechStart(0, 0, frames, 20)]
        self.seg.append(frame)
        events = [SpeechAudio(0, frame)]
        self.silent = self.silent + 1 if frame != VOICE else 0
        if self.silent >= 3:
            events.append(SpeechEnd(0, 0, tuple(self.seg), 20, False))
            self.seg, self.silent, self.voiced = None, 0, 0
        return events


class FakeClient:
    def __init__(self):
        self.connected = True
        self.session_id = self.turn_id = None
        self.calls: list = []
        self.audio: list[tuple[uuid.UUID, bytes]] = []
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def recv(self):
        return await self.inbox.get()

    async def start_session(self):
        if not self.connected:
            return None
        self.session_id = uuid.uuid4()
        self.calls.append("session.start")
        return self.session_id

    async def end_session(self, reason):
        self.calls.append(("session.end", reason))
        self.session_id = self.turn_id = None
        return True

    async def start_input(self):
        if self.session_id is None:
            return None
        self.turn_id = uuid.uuid4()
        self.calls.append("input.start")
        return self.turn_id

    async def send_audio(self, pcm):
        self.audio.append((self.turn_id, pcm))
        return True

    async def end_input(self, truncated=False):
        self.calls.append("input.end")
        return True

    async def cancel_turn(self, reason):
        self.calls.append(("turn.cancel", reason))
        self.turn_id = None
        return True

    def server(self, type_, **data):
        self.inbox.put_nowait(ServerEvent(type_, self.session_id, self.turn_id, {"type": type_, **data}))

    def tts(self, *chunks):
        for seq, pcm in enumerate(chunks):
            self.inbox.put_nowait(AudioFrame(KIND_TTS, self.turn_id, seq, pcm))

    def notify(self, text, *chunks):
        notify_id = uuid.uuid4()
        self.inbox.put_nowait(ServerEvent("notify.start", None, None, {"notify_id": str(notify_id), "text": text}))
        for seq, pcm in enumerate(chunks):
            self.inbox.put_nowait(AudioFrame(KIND_TTS, notify_id, seq, pcm))
        self.inbox.put_nowait(ServerEvent("notify.end", None, None, {"notify_id": str(notify_id)}))

    def disconnect(self):
        self.connected = False
        self.session_id = self.turn_id = None
        self.inbox.put_nowait(Disconnected("测试断线", None))


class FakePlayer:
    instances: list["FakePlayer"] = []
    drain_delay = 0.0
    fail_start = False

    def __init__(self, settings):
        self.written: list[bytes] = []
        self.started = self.drained = self.stopped = False
        FakePlayer.instances.append(self)

    def start(self):
        if FakePlayer.fail_start:
            raise RuntimeError("设备打不开")
        self.started = True

    async def write(self, data):
        self.written.append(data)

    async def drain(self):
        await asyncio.sleep(FakePlayer.drain_delay)
        self.drained = True
        return True

    def stop(self):
        self.stopped = True


class Harness:
    def __init__(self, **kw):
        params = dict(silence_timeout_s=0.5, response_timeout_s=0.5, playback_tail_ms=50, wake_sound=True)
        params.update(kw)
        self.settings = Settings(**params)
        self.cap, self.kws, self.vad, self.client = FakeCapture(), FakeKWS(), FakeVAD(), FakeClient()
        self.conv = Conversation(self.settings, self.cap, self.kws, self.vad, self.client, player_factory=FakePlayer)

    async def __aenter__(self):
        self.task = asyncio.create_task(self.conv.run())
        return self

    async def __aexit__(self, *exc):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass

    async def until(self, pred, timeout=2.0):
        async def poll():
            while not pred():
                await asyncio.sleep(0.005)
        await asyncio.wait_for(poll(), timeout)

    async def state(self, s, timeout=2.0):
        await self.until(lambda: self.conv.state is s, timeout)

    async def settle(self):
        await asyncio.sleep(0.05)

    async def wake(self):
        self.cap.push(QUIET, WAKE)
        await self.state(State.LISTENING)
        if self.settings.wake_sound:
            await self.until(lambda: self.conv._beep_token is None)

    async def speak(self, voice=5):
        self.cap.push(*([VOICE] * voice + [QUIET] * 3))
        await self.state(State.THINKING)

    def reply_player(self):
        return FakePlayer.instances[-1]


def run(coro):
    asyncio.run(coro)


def sound(player) -> list[bytes]:
    """去掉播放前后垫的静音。"""
    return [c for c in player.written if c != conv_module.PAD_SILENCE]


# ---- IDLE 与唤醒 ----

def test_idle_routes_frames_to_kws_only():
    async def main():
        async with Harness() as h:
            h.cap.push(QUIET, VOICE, VOICE, VOICE, QUIET)
            await h.until(lambda: len(h.kws.frames) == 5)
            assert h.conv.state is State.IDLE
            assert h.vad.frames == [] and h.client.calls == [] and h.client.audio == []
    run(main())


def test_wake_starts_session_and_plays_beep():
    async def main():
        async with Harness() as h:
            await h.wake()
            assert h.client.calls == ["session.start"]
            assert h.kws.resets == 1 and h.vad.resets >= 1
            beep = FakePlayer.instances[0]
            assert beep.started and beep.drained and beep.stopped
            assert len(b"".join(sound(beep))) == 2 * int(0.2 * 16000)
            assert beep.written[0] == beep.written[-1] == conv_module.PAD_SILENCE  # 前后垫静音
    run(main())


def test_mic_ignored_while_beep_plays():
    async def main():
        FakePlayer.drain_delay = 0.3
        async with Harness() as h:
            h.cap.push(WAKE)
            await h.state(State.LISTENING)
            h.cap.push(*[VOICE] * 5)
            await h.settle()
            assert h.vad.frames == [] and h.client.audio == []
            await h.until(lambda: h.conv._beep_token is None)
            h.cap.push(*[VOICE] * 3)
            await h.until(lambda: len(h.vad.frames) == 3)
    run(main())


def test_wake_sound_disabled():
    async def main():
        async with Harness(wake_sound=False) as h:
            await h.wake()
            assert FakePlayer.instances == []
            await h.speak()
    run(main())


def test_wake_ignored_when_offline():
    async def main():
        async with Harness() as h:
            h.client.connected = False
            h.cap.push(WAKE, QUIET)
            await h.until(lambda: len(h.kws.frames) == 2)
            assert h.conv.state is State.IDLE and h.client.calls == []
            assert h.kws.resets == 1
    run(main())


# ---- 完整一轮 ----

def test_full_turn_uploads_preroll_then_plays_reply():
    async def main():
        FakePlayer.drain_delay = 0.2
        async with Harness() as h:
            await h.wake()
            await h.speak(voice=5)
            assert h.client.calls == ["session.start", "input.start", "input.end"]
            turn = h.client.turn_id
            # 预缓冲 2 帧语音 + 后续 3 帧语音 + 3 帧静音
            assert [p for _, p in h.client.audio] == [VOICE] * 5 + [QUIET] * 3
            assert all(t == turn for t, _ in h.client.audio)

            h.client.server("transcript", text="今天天气怎么样", final=True)
            h.client.server("response.text", text="晴天")
            h.client.server("response.audio.start", sample_rate=16000)
            await h.state(State.SPEAKING)
            h.client.tts(b"\x01\x00" * 320, b"\x02\x00" * 320)
            h.client.server("response.audio.end", frames=2)
            h.client.server("response.done")
            await asyncio.sleep(0.1)
            assert h.conv.state is State.SPEAKING  # 仍在排空播放
            await h.state(State.FOLLOW_UP)
            player = h.reply_player()
            assert sound(player) == [b"\x01\x00" * 320, b"\x02\x00" * 320]
            assert player.drained and player.stopped
    run(main())


def test_mic_discarded_while_thinking_and_speaking():
    async def main():
        async with Harness() as h:
            await h.wake()
            await h.speak()
            n_vad, n_kws, n_audio = len(h.vad.frames), len(h.kws.frames), len(h.client.audio)
            h.cap.push(*[VOICE] * 10)
            await h.settle()
            h.client.server("response.audio.start", sample_rate=16000)
            await h.state(State.SPEAKING)
            h.cap.push(*[VOICE, WAKE] * 5)
            await h.settle()
            assert (len(h.vad.frames), len(h.kws.frames), len(h.client.audio)) == (n_vad, n_kws, n_audio)
    run(main())


def test_tail_wait_before_follow_up():
    async def main():
        async with Harness(playback_tail_ms=300) as h:
            await h.wake()
            await h.speak()
            h.client.server("response.audio.start", sample_rate=16000)
            h.client.tts(b"\x00\x00" * 320)
            h.client.server("response.audio.end")
            h.client.server("response.done")
            await h.until(lambda: h.reply_player().stopped)
            await asyncio.sleep(0.15)
            assert h.conv.state is State.SPEAKING
            await h.state(State.FOLLOW_UP, timeout=1)
    run(main())


def test_follow_up_starts_new_turn_without_wake():
    async def main():
        async with Harness() as h:
            await h.wake()
            await h.speak()
            first = h.client.turn_id
            h.client.server("response.done")
            await h.state(State.FOLLOW_UP)
            n_kws = len(h.kws.frames)
            await h.speak()
            assert h.client.turn_id != first
            assert h.client.calls.count("session.start") == 1
            assert h.client.calls.count("input.start") == 2
            assert len(h.kws.frames) == n_kws
    run(main())


def test_multiple_audio_segments_share_one_playback():
    async def main():
        async with Harness() as h:
            await h.wake()
            await h.speak()
            for chunk in (b"\x01\x00", b"\x02\x00"):
                h.client.server("response.audio.start", sample_rate=16000)
                h.client.tts(chunk * 320)
                h.client.server("response.audio.end")
            await h.settle()
            assert h.conv.state is State.SPEAKING
            h.client.server("response.done")
            await h.state(State.FOLLOW_UP)
            assert sound(h.reply_player()) == [b"\x01\x00" * 320, b"\x02\x00" * 320]
            assert len(FakePlayer.instances) == 2  # 提示音 + 一次回复播放
    run(main())


def test_done_without_audio_goes_to_follow_up():
    async def main():
        async with Harness() as h:
            await h.wake()
            await h.speak()
            h.client.server("response.text", text="好的")
            h.client.server("response.done")
            await h.state(State.FOLLOW_UP)
            assert len(FakePlayer.instances) == 1
    run(main())


# ---- 定时提醒 ----

def test_notice_in_idle_starts_session_and_follows_up():
    async def main():
        async with Harness(silence_timeout_s=0.3) as h:
            h.client.notify("提醒你：开会", b"\x01\x00" * 320, b"\x02\x00" * 320)
            await h.state(State.SPEAKING)
            assert h.client.calls == ["session.start"]
            await h.state(State.FOLLOW_UP)
            played = sound(h.reply_player())
            assert played == [h.conv._beep, b"\x01\x00" * 320 + b"\x02\x00" * 320]  # 提示音 + 拼接后的提醒语音
            await h.speak()  # 不用唤醒词直接回应
            assert h.client.calls == ["session.start", "input.start", "input.end"]
    run(main())


def test_notice_without_reply_is_not_false_wake():
    async def main():
        async with Harness(silence_timeout_s=0.2) as h:
            h.client.notify("提醒你：吃药", b"\x01\x00" * 320)
            await h.state(State.FOLLOW_UP)
            await h.state(State.IDLE, timeout=1)
            assert h.client.calls[-1] == ("session.end", "silence_timeout")
            assert h.conv.stats.false_wakes == 0
    run(main())


def test_notice_waits_for_current_reply():
    async def main():
        async with Harness() as h:
            await h.wake()
            await h.speak()
            h.client.notify("提醒你：开会", b"\x09\x00" * 320)
            await h.settle()
            assert h.conv.state is State.THINKING
            h.client.server("response.audio.start", sample_rate=16000)
            h.client.tts(b"\x01\x00" * 320)
            h.client.server("response.audio.end")
            h.client.server("response.done")
            await h.until(lambda: len(FakePlayer.instances) == 3)  # 提示音、回复、提醒
            reply, notice = FakePlayer.instances[1:]
            assert sound(reply) == [b"\x01\x00" * 320]
            assert sound(notice)[-1] == b"\x09\x00" * 320
            assert h.client.calls.count("session.start") == 1
            await h.state(State.FOLLOW_UP)
    run(main())


def test_partial_notice_dropped_on_disconnect():
    async def main():
        async with Harness() as h:
            notify_id = uuid.uuid4()
            h.client.inbox.put_nowait(ServerEvent("notify.start", None, None, {"notify_id": str(notify_id), "text": "x"}))
            h.client.inbox.put_nowait(AudioFrame(KIND_TTS, notify_id, 0, b"\x01\x00" * 320))
            h.client.disconnect()
            h.client.inbox.put_nowait(ServerEvent("notify.end", None, None, {"notify_id": str(notify_id)}))
            await h.until(lambda: h.conv.stats.disconnects == 1)
            await h.settle()
            assert h.conv.state is State.IDLE and FakePlayer.instances == []
    run(main())


# ---- 超时 ----

def test_silence_timeout_in_listening_returns_to_idle():
    async def main():
        async with Harness(silence_timeout_s=0.2) as h:
            await h.wake()
            resets = h.kws.resets
            await h.state(State.IDLE, timeout=1)
            assert h.client.calls[-1] == ("session.end", "silence_timeout")
            assert h.kws.resets == resets + 1
    run(main())


def test_silence_timeout_not_applied_while_uploading():
    async def main():
        async with Harness(silence_timeout_s=0.2) as h:
            await h.wake()
            h.cap.push(VOICE, VOICE, VOICE)
            await h.until(lambda: h.client.calls[-1] == "input.start")
            await asyncio.sleep(0.4)
            assert h.conv.state is State.LISTENING
    run(main())


def test_silence_timeout_in_follow_up_returns_to_idle():
    async def main():
        async with Harness(silence_timeout_s=0.3) as h:
            await h.wake()
            await h.speak()
            h.client.server("response.done")
            await h.state(State.FOLLOW_UP)
            await h.state(State.IDLE, timeout=1)
            assert h.client.calls[-1] == ("session.end", "silence_timeout")
    run(main())


def test_response_timeout_cancels_turn():
    async def main():
        async with Harness(response_timeout_s=0.2) as h:
            await h.wake()
            await h.speak()
            await h.state(State.FOLLOW_UP, timeout=1)
            assert ("turn.cancel", "timeout") in h.client.calls
    run(main())


def test_missing_done_after_audio_end_times_out():
    async def main():
        async with Harness(response_timeout_s=0.2) as h:
            await h.wake()
            await h.speak()
            h.client.server("response.audio.start", sample_rate=16000)
            h.client.tts(b"\x00\x00" * 320)
            h.client.server("response.audio.end")
            await h.state(State.FOLLOW_UP, timeout=1)
            assert h.reply_player().drained
    run(main())


# ---- 错误与断线 ----

def test_server_error_while_thinking():
    async def main():
        async with Harness() as h:
            await h.wake()
            await h.speak()
            h.client.server("error", code="asr_failed", message="识别失败")
            await h.state(State.FOLLOW_UP)
            assert ("turn.cancel", "error") in h.client.calls
    run(main())


def test_server_error_while_speaking_stops_playback():
    async def main():
        async with Harness() as h:
            await h.wake()
            FakePlayer.drain_delay = 5
            await h.speak()
            h.client.server("response.audio.start", sample_rate=16000)
            h.client.tts(b"\x00\x00" * 320)
            h.client.server("response.done")
            await h.until(lambda: len(sound(h.reply_player())) == 1)
            h.client.server("error", code="tts_failed", message="合成失败")
            await h.state(State.FOLLOW_UP)
            assert h.reply_player().stopped and not h.reply_player().drained
    run(main())


@pytest.mark.parametrize("stage", ["listening", "uploading", "thinking", "speaking", "follow_up"])
def test_disconnect_returns_to_idle(stage):
    async def main():
        async with Harness() as h:
            await h.wake()
            if stage == "uploading":
                h.cap.push(VOICE, VOICE)
                await h.until(lambda: h.client.calls[-1] == "input.start")
            if stage in ("thinking", "speaking", "follow_up"):
                await h.speak()
            if stage == "speaking":
                h.client.server("response.audio.start", sample_rate=16000)
                await h.state(State.SPEAKING)
            if stage == "follow_up":
                h.client.server("response.done")
                await h.state(State.FOLLOW_UP)
            h.client.disconnect()
            await h.state(State.IDLE)
            assert not any(isinstance(c, tuple) and c[0] == "session.end" for c in h.client.calls)
            if stage == "speaking":
                assert h.reply_player().stopped
            # 回到 IDLE 后重新只送 KWS
            n_vad = len(h.vad.frames)
            h.cap.push(VOICE, VOICE, VOICE)
            await h.settle()
            assert len(h.vad.frames) == n_vad
    run(main())


def test_reconnected_event_is_harmless():
    async def main():
        async with Harness() as h:
            h.client.inbox.put_nowait(Connected())
            await h.wake()
    run(main())


def test_player_failure_raises_audio_error():
    async def main():
        h = Harness()
        task = asyncio.create_task(h.conv.run())
        await h.wake()
        await h.speak()
        FakePlayer.fail_start = True
        h.client.server("response.audio.start", sample_rate=16000)
        with pytest.raises(conv_module.AudioDeviceError, match="设备打不开"):
            await asyncio.wait_for(task, 2)
        assert h.client.calls[-1] == ("session.end", "error")
    run(main())


def test_beep_failure_raises_audio_error():
    async def main():
        FakePlayer.fail_start = True
        h = Harness()
        task = asyncio.create_task(h.conv.run())
        h.cap.push(WAKE)
        with pytest.raises(conv_module.AudioDeviceError):
            await asyncio.wait_for(task, 2)
    run(main())


def test_digital_silence_raises_audio_error(monkeypatch):
    monkeypatch.setattr(conv_module, "DIGITAL_SILENCE_S", 0.2)  # 10 帧

    async def main():
        h = Harness()
        task = asyncio.create_task(h.conv.run())
        zero = bytes(640)
        h.cap.push(*[zero] * 9, QUIET, *[zero] * 9)  # 中间有底噪则重新计数
        await asyncio.sleep(0.05)
        assert not task.done()
        h.cap.push(zero)
        with pytest.raises(conv_module.AudioDeviceError, match="全零"):
            await asyncio.wait_for(task, 2)
    run(main())


def test_stats_counts_wakes_turns_false_wakes_and_disconnects():
    async def main():
        async with Harness(silence_timeout_s=0.2) as h:
            await h.wake()
            await h.speak()
            h.client.server("response.done")
            await h.state(State.FOLLOW_UP)
            await h.state(State.IDLE, timeout=1)
            await h.wake()  # 唤醒后不说话
            await h.state(State.IDLE, timeout=1)
            h.client.disconnect()
            await h.until(lambda: h.conv.stats.disconnects == 1)
            s = h.conv.stats
            assert (s.wakes, s.turns, s.false_wakes) == (2, 1, 1)
    run(main())


def test_shared_stats_object():
    stats = conv_module.Stats(wakes=5)
    h = Harness()
    conv = Conversation(h.settings, h.cap, h.kws, h.vad, h.client, player_factory=FakePlayer, stats=stats)
    assert conv.stats is stats


# ---- 运行与退出 ----

def test_shutdown_ends_active_session():
    async def main():
        h = Harness()
        async with h:
            await h.wake()
        assert h.client.calls[-1] == ("session.end", "shutdown")
    run(main())


def test_shutdown_in_idle_sends_nothing():
    async def main():
        h = Harness()
        async with h:
            await h.settle()
        assert h.client.calls == []
    run(main())


def test_mic_timeout_stops_run(monkeypatch):
    monkeypatch.setattr(conv_module, "MIC_TIMEOUT_S", 0.1)

    async def main():
        h = Harness()
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(h.conv.run(), 2)
        assert loop.time() - start < 1  # 由采集超时触发，而不是外层 wait_for
    run(main())
