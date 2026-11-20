import asyncio
import os
import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd

from app.audio import capture, devices
from app.audio.capture import AudioCapture, choose_input_format, downmix
from app.config import Settings
from app.main import main

HOSTAPIS = [
    {"name": "MME", "default_input_device": 0, "default_output_device": -1},
    {"name": "Windows WASAPI", "default_input_device": 1, "default_output_device": -1},
]
DEVICES = [
    {"name": "耳机式麦克风 (HyperX Virtual Surround", "hostapi": 0, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 44100.0},
    {"name": "耳机式麦克风 (HyperX Virtual Surround Sound)", "hostapi": 1, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 48000.0},
    {"name": "Speakers", "hostapi": 1, "max_input_channels": 0,
     "max_output_channels": 2, "default_samplerate": 48000.0},
]
OK = SimpleNamespace(input_overflow=False)


class FakeStream:
    instances: list["FakeStream"] = []
    fail_start = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = self.stopped = self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        if FakeStream.fail_start:
            raise sd.PortAudioError("Device unavailable", -9985)
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def feed(self, block, status=OK):
        self.callback(block, len(block), None, status)


@pytest.fixture(autouse=True)
def fake_sd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HOMEPILOT_"):
            monkeypatch.delenv(key)
    FakeStream.instances = []
    FakeStream.fail_start = False
    monkeypatch.setattr(devices.sd, "query_hostapis", lambda: HOSTAPIS)
    monkeypatch.setattr(devices.sd, "query_devices", lambda: DEVICES)
    monkeypatch.setattr(capture.sd, "InputStream", FakeStream)
    monkeypatch.setattr(capture.sd, "check_input_settings", wasapi_check)


def wasapi_check(device, samplerate, channels, dtype):
    if DEVICES[device]["hostapi"] == 1 and samplerate != 48000:
        raise sd.PortAudioError("Invalid sample rate", -9997)


def stereo_block(n, rate=48000, start=0, freq=440.0):
    t = (np.arange(n) + start) / rate
    left = 12000 * np.sin(2 * np.pi * freq * t)
    right = 4000 * np.sin(2 * np.pi * freq * t)
    return np.rint(np.stack([left, right], axis=1)).astype(np.int16)


async def drain_loop():
    for _ in range(5):
        await asyncio.sleep(0)


# ---- 设备与格式选择 ----

def test_find_input_device_default_prefers_wasapi():
    d = devices.find_input_device("")
    assert d.index == 1 and d.hostapi == "Windows WASAPI"


def test_find_input_device_by_index_and_name():
    assert devices.find_input_device("0").index == 0
    assert devices.find_input_device("hyperx").index == 1
    with pytest.raises(LookupError):
        devices.find_input_device("2")  # 输出设备
    with pytest.raises(LookupError):
        devices.find_input_device("不存在")


def test_choose_input_format_uses_check_settings():
    wasapi = devices.find_input_device("")
    assert choose_input_format(wasapi) == (48000, 1)
    assert choose_input_format(wasapi, channels=2) == (48000, 2)


def test_choose_input_format_falls_back_to_device_channels(monkeypatch):
    def stereo_only(device, samplerate, channels, dtype):
        if channels != 2 or samplerate != 48000:
            raise sd.PortAudioError("Invalid", -9998)

    monkeypatch.setattr(capture.sd, "check_input_settings", stereo_only)
    assert choose_input_format(devices.find_input_device("")) == (48000, 2)


def test_choose_input_format_none_supported(monkeypatch):
    def reject(**kwargs):
        raise sd.PortAudioError("Invalid sample rate", -9997)

    monkeypatch.setattr(capture.sd, "check_input_settings", reject)
    with pytest.raises(RuntimeError):
        choose_input_format(devices.find_input_device(""))


def test_downmix():
    block = np.array([[100, 300], [-32768, -32768], [1, 2]], dtype=np.int16)
    out = downmix(block)
    assert out.dtype == np.int16
    assert out.tolist() == [200, -32768, 2]
    assert downmix(block[:, :1]).tolist() == [100, -32768, 1]


# ---- 采集流程 ----

def test_opens_stream_with_native_format():
    async def run():
        async with AudioCapture(Settings(), channels=2) as cap:
            s = FakeStream.instances[0]
            assert s.started
            assert s.kwargs["device"] == 1
            assert s.kwargs["samplerate"] == 48000
            assert s.kwargs["channels"] == 2
            assert s.kwargs["dtype"] == "int16"
            assert s.kwargs["blocksize"] == 960
            assert cap.running
        assert s.stopped and s.closed and not cap.running

    asyncio.run(run())


def test_48k_stereo_to_16k_mono_frames():
    async def run():
        async with AudioCapture(Settings(), channels=2) as cap:
            s = FakeStream.instances[0]
            for i in range(50):
                s.feed(stereo_block(960, start=i * 960))
            await drain_loop()
            frames = [cap.queue.get_nowait() for _ in range(cap.queue.qsize())]
        assert len(frames) == 49  # 1 秒输入，滤波延迟使最后一帧尚未凑满
        assert all(isinstance(f, bytes) and len(f) == 640 for f in frames)
        pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
        # 下混后幅度约为 (12000 + 4000) / 2
        assert 7500 < np.max(np.abs(pcm[200:])) < 8500
        # 16 kHz 下 440 Hz 正弦的过零次数约 2 * 440 * 时长
        crossings = np.count_nonzero(np.diff(np.signbit(pcm[200:])))
        assert abs(crossings - 2 * 440 * (len(pcm) - 200) / 16000) < 5

    asyncio.run(run())


@pytest.mark.parametrize("block_size", [480, 960, 1000, 37])
def test_frames_are_exactly_320_samples(block_size):
    async def run():
        async with AudioCapture(Settings(), queue_size=1000) as cap:
            s = FakeStream.instances[0]
            for i in range(48000 // block_size):
                s.feed(stereo_block(block_size, start=i * block_size)[:, :1])
            await drain_loop()
            frames = [cap.queue.get_nowait() for _ in range(cap.queue.qsize())]
            assert len(cap._pending) < 320
        assert frames and all(len(f) == 320 * 2 for f in frames)

    asyncio.run(run())


def test_five_seconds_gives_about_250_frames():
    async def run():
        async with AudioCapture(Settings(), queue_size=1000) as cap:
            s = FakeStream.instances[0]
            for i in range(250):
                s.feed(stereo_block(960, start=i * 960)[:, :1])
            await drain_loop()
            n = cap.queue.qsize()
        assert 249 <= n <= 250

    asyncio.run(run())


def test_callback_from_other_thread_is_delivered():
    async def run():
        async with AudioCapture(Settings()) as cap:
            s = FakeStream.instances[0]
            worker_ids = []

            def worker():
                worker_ids.append(threading.get_ident())
                for i in range(10):
                    s.feed(stereo_block(960, start=i * 960)[:, :1])

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            assert cap.queue.qsize() == 0  # 尚未在事件循环中处理
            frame = await asyncio.wait_for(cap.read(), timeout=1)
            assert len(frame) == 640
            assert worker_ids[0] != threading.get_ident()

    asyncio.run(run())


def test_queue_full_drops_oldest():
    async def run():
        async with AudioCapture(Settings(), queue_size=5) as cap:
            s = FakeStream.instances[0]
            for i in range(20):
                s.feed(stereo_block(960, start=i * 960)[:, :1])
            await drain_loop()
            produced = cap.frames_produced
            assert cap.queue.qsize() == 5
            assert cap.dropped_frames == produced - 5
            # 保留的是最新的帧
            kept = [cap.queue.get_nowait() for _ in range(5)]
        last = np.frombuffer(kept[-1], dtype=np.int16)
        assert np.any(last != 0)

    asyncio.run(run())


def test_input_overflow_counted():
    async def run():
        async with AudioCapture(Settings()) as cap:
            s = FakeStream.instances[0]
            s.feed(stereo_block(960)[:, :1], SimpleNamespace(input_overflow=True))
            s.feed(stereo_block(960)[:, :1])
            await drain_loop()
            assert cap.overflow_count == 1

    asyncio.run(run())


def test_start_failure_closes_stream():
    FakeStream.fail_start = True

    async def run():
        cap = AudioCapture(Settings())
        with pytest.raises(sd.PortAudioError):
            cap.start()
        assert FakeStream.instances[0].closed and not cap.running

    asyncio.run(run())


def test_exception_inside_context_releases_stream():
    async def run():
        with pytest.raises(ValueError):
            async with AudioCapture(Settings()):
                raise ValueError("boom")
        assert FakeStream.instances[0].closed

    asyncio.run(run())


def test_cancel_while_reading_releases_stream():
    async def run():
        async def reader():
            async with AudioCapture(Settings()) as cap:
                await cap.read()

        task = asyncio.create_task(reader())
        await drain_loop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert FakeStream.instances[0].closed

    asyncio.run(run())


def test_blocks_after_stop_are_ignored_and_stop_is_idempotent():
    async def run():
        cap = AudioCapture(Settings())
        cap.start()
        s = FakeStream.instances[0]
        s.feed(stereo_block(960 * 2)[:, :1])
        cap.stop()
        await drain_loop()
        assert cap.frames_produced == 0
        cap.stop()

    asyncio.run(run())


def test_callback_after_loop_closed_does_not_raise():
    cap = AudioCapture(Settings())

    async def run():
        cap.start()

    asyncio.run(run())
    FakeStream.instances[0].feed(stereo_block(960)[:, :1])
    cap.stop()


def test_start_requires_running_loop():
    with pytest.raises(RuntimeError):
        AudioCapture(Settings()).start()
    assert not FakeStream.instances


# ---- CLI ----

def test_record_test_cli_writes_wav(monkeypatch, tmp_path):
    class AutoFeedStream(FakeStream):
        def start(self):
            super().start()
            loop = asyncio.get_running_loop()

            def pump(i=0):
                if not self.closed and i < 400:
                    self.feed(stereo_block(960, start=i * 960)[:, :1])
                    loop.call_later(0.001, pump, i + 1)

            loop.call_soon(pump)

    monkeypatch.setattr(capture.sd, "InputStream", AutoFeedStream)
    out = tmp_path / "t.wav"
    assert main(["--record-test", "1", "--output", str(out)]) == 0
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() == 50 * 320
    assert FakeStream.instances[0].closed


def test_record_test_cli_rejects_non_positive():
    assert main(["--record-test", "0"]) == 2
