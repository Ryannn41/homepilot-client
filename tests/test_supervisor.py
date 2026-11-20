import asyncio
import logging
import os

import pytest

from app import conversation as conv_module
from app import main as main_module
from app.config import Settings
from app.main import RunMonitor, retry_delay, setup_file_logging

QUIET = b"q" * 640


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HOMEPILOT_"):
            monkeypatch.delenv(key)
    FakeCapture.script = []
    FakeCapture.opened = 0
    refreshes.clear()
    delays.clear()
    monkeypatch.setattr(main_module, "WakeWordDetector", FakeKWS)
    monkeypatch.setattr(main_module, "VADDetector", FakeVAD)
    monkeypatch.setattr(main_module, "VoiceClient", FakeClient)
    monkeypatch.setattr(main_module, "AudioCapture", FakeCapture)
    monkeypatch.setattr(main_module.devices, "refresh_portaudio", lambda: refreshes.append(1))

    def fake_delay(failures):
        delays.append(failures)
        return 0.01

    monkeypatch.setattr(main_module, "retry_delay", fake_delay)


refreshes: list[int] = []
delays: list[int] = []


class FakeKWS:
    keywords = ["你好小余"]

    def __init__(self, settings):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    async def process(self, frame):
        return None

    def reset(self):
        pass


class FakeVAD(FakeKWS):
    def process(self, frame):
        return []


class FakeClient:
    connected = True
    session_id = None

    def __init__(self, settings):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def recv(self):
        await asyncio.Event().wait()


class FakeCapture:
    """按 script 依次决定每次打开后的行为。"""

    script: list[str] = []
    opened = 0

    def __init__(self, settings):
        self.behavior = FakeCapture.script.pop(0) if FakeCapture.script else "ok"
        self.device = type("D", (), {"index": 21, "name": "HyperX", "hostapi": "Windows WASAPI"})()
        self.samplerate, self.channels, self.dropped_frames = 48000, 1, 3
        self.started = 0.0

    async def __aenter__(self):
        if self.behavior == "fail_open":
            raise LookupError("找不到输入设备: HyperX")
        FakeCapture.opened += 1
        self.started = asyncio.get_running_loop().time()
        return self

    async def __aexit__(self, *exc):
        pass

    async def read(self):
        await asyncio.sleep(0.005)
        elapsed = asyncio.get_running_loop().time() - self.started
        if self.behavior == "timeout" or (self.behavior == "timeout_late" and elapsed > 0.1):
            raise TimeoutError
        if self.behavior == "bug":
            raise ValueError("意外的错误")
        if self.behavior == "zeros":
            return bytes(640)
        return QUIET


async def start(settings=None):
    task = asyncio.create_task(main_module.run_assistant(settings or Settings()))
    return task


async def until(pred, timeout=3.0):
    async def poll():
        while not pred():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(poll(), timeout)


async def stop(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_retries_until_device_available(caplog):
    caplog.set_level(logging.INFO)
    FakeCapture.script = ["fail_open", "fail_open", "ok"]

    async def main():
        task = await start()
        await until(lambda: FakeCapture.opened == 1)
        await asyncio.sleep(0.05)
        assert not task.done()
        await stop(task)
    asyncio.run(main())
    assert len(refreshes) == 2 and delays == [0, 1]
    assert "找不到输入设备" in caplog.text and "第 2 次重启" in caplog.text


def test_capture_timeout_restarts_audio(caplog):
    FakeCapture.script = ["timeout", "ok"]

    async def main():
        task = await start()
        await until(lambda: FakeCapture.opened == 2)
        await stop(task)
    asyncio.run(main())
    assert refreshes == [1]


def test_digital_silence_restarts_audio(monkeypatch, caplog):
    monkeypatch.setattr(conv_module, "DIGITAL_SILENCE_S", 0.1)
    FakeCapture.script = ["zeros", "ok"]

    async def main():
        task = await start()
        await until(lambda: FakeCapture.opened == 2)
        await stop(task)
    asyncio.run(main())
    assert "全零" in caplog.text


def test_unexpected_error_logged_with_traceback_and_recovers(caplog):
    FakeCapture.script = ["bug", "ok"]

    async def main():
        task = await start()
        await until(lambda: FakeCapture.opened == 2)
        await stop(task)
    asyncio.run(main())
    record = next(r for r in caplog.records if r.getMessage() == "运行出错")
    assert record.exc_info and isinstance(record.exc_info[1], ValueError)


def test_backoff_resets_after_healthy_run(monkeypatch):
    monkeypatch.setattr(main_module, "HEALTHY_RUN_S", 0.05)
    FakeCapture.script = ["timeout", "timeout_late", "timeout", "ok"]

    async def main():
        task = await start()
        await until(lambda: FakeCapture.opened == 4)
        await stop(task)
    asyncio.run(main())
    assert delays == [0, 0, 1]


def test_retry_delay_schedule():
    assert [retry_delay(i) for i in range(6)] == [1, 2, 5, 10, 10, 10]


def test_exit_summary_logged_on_cancel(caplog):
    caplog.set_level(logging.INFO)
    FakeCapture.script = ["timeout", "ok"]

    async def main():
        task = await start()
        await until(lambda: FakeCapture.opened == 2)
        await stop(task)
    asyncio.run(main())
    summary = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("退出统计"))
    assert "音频重启 1 次" in summary and "丢帧 6" in summary and "内存" in summary


def test_periodic_report(caplog):
    caplog.set_level(logging.INFO)

    async def main():
        stats = conv_module.Stats(wakes=3, turns=2, false_wakes=1, disconnects=4)
        monitor = RunMonitor(stats, asyncio.get_running_loop().time())
        task = asyncio.create_task(main_module._report_periodically(monitor, 0.02))
        await asyncio.sleep(0.07)
        task.cancel()
    asyncio.run(main())
    reports = [r.getMessage() for r in caplog.records if r.getMessage().startswith("运行统计")]
    assert len(reports) >= 2
    assert "唤醒 3 次, 对话 2 轮, 疑似误唤醒 1 次, 断线 4 次" in reports[0]


# ---- 日志文件 ----

def test_file_logging_writes_utf8(tmp_path):
    handler = setup_file_logging(Settings(log_dir=str(tmp_path / "logs")))
    try:
        logging.getLogger("homepilot").warning("写入日志文件：你好小余")
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    text = (tmp_path / "logs" / "homepilot.log").read_text(encoding="utf-8")
    assert "写入日志文件：你好小余" in text


def test_file_logging_disabled():
    assert setup_file_logging(Settings(log_dir="")) is None


def test_main_writes_log_file_and_removes_handler(monkeypatch, tmp_path):
    async def fake_run(settings):
        logging.getLogger("homepilot").info("助手运行中")
        return 0

    monkeypatch.setattr(main_module, "run_assistant", fake_run)
    before = list(logging.getLogger().handlers)
    assert main_module.main([]) == 0
    assert logging.getLogger().handlers == before
    text = (tmp_path / "logs" / "homepilot.log").read_text(encoding="utf-8")
    assert "HomePilot Client" in text and "助手运行中" in text
