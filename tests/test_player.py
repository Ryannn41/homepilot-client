import asyncio
import os
import threading
import time

import numpy as np
import pytest
import sounddevice as sd

from app.audio import devices, player
from app.audio.player import AudioPlayer, candidate_output_formats, to_pcm16
from app.audio.resample import StreamResampler
from app.config import Settings
from app.main import main, make_tone

HOSTAPIS = [
    {"name": "MME", "default_input_device": -1, "default_output_device": 0},
    {"name": "Windows WASAPI", "default_input_device": -1, "default_output_device": 1},
]
DEVICES = [
    {"name": "头戴式耳机 (HyperX Virtual Surround ", "hostapi": 0, "max_input_channels": 0,
     "max_output_channels": 2, "default_samplerate": 16000.0},
    {"name": "头戴式耳机 (HyperX Virtual Surround Sound)", "hostapi": 1, "max_input_channels": 0,
     "max_output_channels": 2, "default_samplerate": 48000.0},
    {"name": "Mono Speaker", "hostapi": 1, "max_input_channels": 0,
     "max_output_channels": 1, "default_samplerate": 48000.0},
    {"name": "Mic", "hostapi": 1, "max_input_channels": 1,
     "max_output_channels": 0, "default_samplerate": 48000.0},
]


class FakeOutputStream:
    """模拟 PortAudio：start() 后在独立线程中反复调用回调。"""

    instances: list["FakeOutputStream"] = []
    fail_open: set[tuple[int, int]] = set()
    fail_start: set[tuple[int, int]] = set()
    realtime = False

    def __init__(self, **kw):
        fmt = (kw["samplerate"], kw["channels"])
        if fmt in FakeOutputStream.fail_open:
            raise sd.PortAudioError("Invalid device", -9996)
        self.kw = kw
        self.blocks: list[np.ndarray] = []
        self.started = self.stopped = self.aborted = self.closed = False
        self.finished_calls = 0
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        FakeOutputStream.instances.append(self)

    def start(self):
        if (self.kw["samplerate"], self.kw["channels"]) in FakeOutputStream.fail_start:
            raise sd.PortAudioError("Unanticipated host error", -9999)
        self.started = True
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        n, ch = self.kw["blocksize"], self.kw["channels"]
        status = sd.CallbackFlags()
        while not self._halt.is_set():
            out = np.full((n, ch), 12345, dtype=np.int16)  # 未写入的部分会暴露出来
            try:
                self.kw["callback"](out, n, None, status)
            except sd.CallbackStop:
                self.blocks.append(out)
                break
            self.blocks.append(out)
            time.sleep(n / self.kw["samplerate"] if FakeOutputStream.realtime else 0.0005)
        self.finished_calls += 1
        self.kw["finished_callback"]()

    def _join(self):
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def stop(self):
        self.stopped = True
        self._join()

    def abort(self):
        self.aborted = True
        self._halt.set()
        self._join()

    def close(self):
        self.closed = True

    def output(self) -> np.ndarray:
        return np.concatenate(self.blocks) if self.blocks else np.zeros((0, self.kw["channels"]), np.int16)


@pytest.fixture(autouse=True)
def fake_sd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HOMEPILOT_"):
            monkeypatch.delenv(key)
    FakeOutputStream.instances = []
    FakeOutputStream.fail_open = set()
    FakeOutputStream.fail_start = set()
    FakeOutputStream.realtime = False
    monkeypatch.setattr(devices.sd, "query_hostapis", lambda: HOSTAPIS)
    monkeypatch.setattr(devices.sd, "query_devices", lambda: DEVICES)
    monkeypatch.setattr(player.sd, "OutputStream", FakeOutputStream)
    monkeypatch.setattr(player.sd, "check_output_settings", fake_check)


def fake_check(device, samplerate, channels, dtype):
    info = DEVICES[device]
    if channels > info["max_output_channels"]:
        raise sd.PortAudioError("Invalid number of channels", -9998)
    if info["hostapi"] == 1 and samplerate != 48000:
        raise sd.PortAudioError("Invalid sample rate", -9997)


def settings(**env):
    return Settings(**env)


def signal(n=16000):
    # 带直流偏置，便于在输出中定位起止
    t = np.arange(n) / 16000
    return np.rint(6000 + 4000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)


def played(stream: FakeOutputStream) -> np.ndarray:
    """去掉欠载/结束补的静音行后的实际播放内容（测试信号带直流偏置，本身不含 0）。"""
    out = stream.output()
    assert not np.any(out == 12345), "回调没有写满 outdata"
    return out[np.any(out != 0, axis=1)]


def resampled(x, out_rate):
    r = StreamResampler(16000, out_rate)
    y = np.concatenate([r.process(x), r.flush()])
    return y[y != 0]


# ---- 输入与格式 ----

def test_to_pcm16():
    assert to_pcm16(b"\x01\x00\xff\xff").tolist() == [1, -1]
    x = np.zeros(3, dtype=np.int16)
    assert to_pcm16(x) is x
    with pytest.raises(ValueError):
        to_pcm16(b"\x00")
    for bad in (np.zeros(3, np.float32), np.zeros((3, 2), np.int16), [1, 2], "abc"):
        with pytest.raises(TypeError):
            to_pcm16(bad)


def test_candidate_formats_prefer_default_rate_and_stereo():
    wasapi = devices.find_output_device("")
    assert wasapi.index == 1
    assert candidate_output_formats(wasapi) == [(48000, 2), (48000, 1)]
    mme = devices.find_output_device("0")
    assert candidate_output_formats(mme)[0] == (16000, 2)
    assert candidate_output_formats(devices.find_output_device("mono"))== [(48000, 1)]


def test_find_output_device_rejects_input_only():
    with pytest.raises(LookupError):
        devices.find_output_device("3")
    assert devices.find_output_device("hyperx").index == 1


# ---- 播放 ----

def test_passthrough_16k():
    async def run():
        p = AudioPlayer(settings(output_device="0"))
        x = signal()
        assert await p.play(x)
        s = FakeOutputStream.instances[0]
        assert (s.kw["samplerate"], s.kw["channels"], s.kw["blocksize"]) == (16000, 2, 320)
        out = played(s)
        assert np.array_equal(out[:, 0], x) and np.array_equal(out[:, 1], x)
        assert s.closed and not p.running

    asyncio.run(run())


def test_resample_16k_to_48k_stereo():
    async def run():
        p = AudioPlayer(settings())
        x = signal()
        assert await p.play(x)
        s = FakeOutputStream.instances[0]
        assert (s.kw["samplerate"], s.kw["channels"], s.kw["blocksize"]) == (48000, 2, 960)
        ref = resampled(x, 48000)
        out = played(s)
        assert np.array_equal(out[:, 0], ref)
        assert np.array_equal(out[:, 0], out[:, 1])

    asyncio.run(run())


def test_mono_device_gets_single_channel():
    async def run():
        p = AudioPlayer(settings(output_device="Mono"))
        assert await p.play(signal(1600))
        assert FakeOutputStream.instances[0].kw["channels"] == 1
        assert p.channels == 1

    asyncio.run(run())


@pytest.mark.parametrize("chunk", [1, 7, 320, 1000, 16000])
def test_chunk_sizes_give_identical_output(chunk):
    async def run():
        x = signal(8000)
        p = AudioPlayer(settings())
        p.start()
        for i in range(0, len(x), chunk):
            await p.write(x[i : i + chunk])
        assert await p.drain()
        ref = resampled(x, 48000)
        assert np.array_equal(played(FakeOutputStream.instances[0])[:, 0], ref)

    asyncio.run(run())


def test_write_accepts_bytes():
    async def run():
        x = signal(3200)
        p = AudioPlayer(settings(output_device="0"))
        p.start()
        await p.write(x.tobytes())
        assert await p.drain()
        assert np.array_equal(played(FakeOutputStream.instances[0])[:, 0], x)

    asyncio.run(run())


def test_empty_audio_does_not_open_device():
    async def run():
        assert await AudioPlayer(settings()).play(b"")
        assert await AudioPlayer(settings()).play(np.zeros(0, np.int16))
        assert not FakeOutputStream.instances

    asyncio.run(run())


def test_invalid_input_rejected_before_opening():
    async def run():
        p = AudioPlayer(settings())
        with pytest.raises(TypeError):
            await p.play(np.zeros(10, np.float32))
        with pytest.raises(ValueError):
            await p.play(b"\x00\x00\x00")
        assert not FakeOutputStream.instances
        with pytest.raises(RuntimeError):
            await p.write(signal(10))  # 未 start

    asyncio.run(run())


def test_write_after_drain_rejected():
    async def run():
        p = AudioPlayer(settings())
        p.start()
        await p.write(signal(320))
        assert await p.drain()
        with pytest.raises(RuntimeError):
            await p.write(signal(320))

    asyncio.run(run())


# ---- 格式回退 ----

def test_falls_back_when_open_or_start_fails():
    FakeOutputStream.fail_open = {(48000, 2)}

    async def run():
        p = AudioPlayer(settings())
        assert await p.play(signal(1600))
        assert (p.samplerate, p.channels) == (48000, 1)

    asyncio.run(run())

    FakeOutputStream.instances = []
    FakeOutputStream.fail_open = set()
    FakeOutputStream.fail_start = {(48000, 2)}

    async def run2():
        p = AudioPlayer(settings())
        assert await p.play(signal(1600))
        first, second = FakeOutputStream.instances
        assert first.closed and not first.started
        assert second.kw["channels"] == 1 and second.closed

    asyncio.run(run2())


def test_all_formats_fail():
    FakeOutputStream.fail_start = {(48000, 2), (48000, 1)}

    async def run():
        p = AudioPlayer(settings())
        with pytest.raises(RuntimeError):
            await p.play(signal(1600))
        assert all(s.closed for s in FakeOutputStream.instances)
        assert not p.running

    asyncio.run(run())


# ---- 停止、异常、取消 ----

def test_stop_during_play():
    FakeOutputStream.realtime = True

    async def run():
        p = AudioPlayer(settings())
        task = asyncio.create_task(p.play(signal(16000 * 5)))
        await asyncio.sleep(0.1)
        p.stop()
        p.stop()
        assert await asyncio.wait_for(task, 1) is False
        s = FakeOutputStream.instances[0]
        assert s.aborted and s.closed and not p.running

    asyncio.run(run())


def test_cancel_during_play_releases_stream():
    FakeOutputStream.realtime = True

    async def run():
        p = AudioPlayer(settings())
        task = asyncio.create_task(p.play(signal(16000 * 5)))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        s = FakeOutputStream.instances[0]
        assert s.aborted and s.closed and not p.running

    asyncio.run(run())


def test_exception_inside_context_releases_stream():
    async def run():
        with pytest.raises(ValueError):
            async with AudioPlayer(settings()) as p:
                await p.write(signal(320))
                raise ValueError("boom")
        assert FakeOutputStream.instances[0].closed

    asyncio.run(run())


def test_stop_without_start_is_safe():
    p = AudioPlayer(settings())
    p.stop()
    p.stop()


def test_player_reusable_after_play():
    async def run():
        p = AudioPlayer(settings())
        assert await p.play(signal(1600))
        assert await p.play(signal(1600))
        assert len(FakeOutputStream.instances) == 2
        assert all(s.closed for s in FakeOutputStream.instances)

    asyncio.run(run())


def test_event_loop_not_blocked_during_play():
    FakeOutputStream.realtime = True

    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        loop = asyncio.get_running_loop()
        start = loop.time()
        assert await AudioPlayer(settings()).play(signal(16000 // 2))
        elapsed = loop.time() - start
        t.cancel()
        assert elapsed >= 0.4  # 按实时节奏播放了约 0.5 秒
        assert ticks >= elapsed / 0.01 * 0.5

    asyncio.run(run())


# ---- CLI ----

def test_make_tone():
    tone = make_tone(1.0)
    assert tone.dtype == np.int16 and len(tone) == 16000
    assert abs(int(tone[0])) < 50 and abs(int(tone[-1])) < 50
    assert 3200 < np.max(np.abs(tone)) <= 3277


def test_play_test_cli(capsys):
    assert main(["--play-test", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "48000 Hz x 2 ch" in out and "完成" in out
    assert FakeOutputStream.instances[0].closed


def test_play_test_cli_rejects_non_positive():
    assert main(["--play-test", "0"]) == 2
