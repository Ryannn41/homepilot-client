"""语音助手常驻运行与启停：供 CLI 与桌面 UI 共用。"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import psutil
import sounddevice as sd

from app.audio import devices
from app.audio.capture import AudioCapture
from app.audio.player import AudioPlayer
from app.audio.vad import VADDetector
from app.audio.wakeword import WakeWordDetector
from app.board import BoardStore
from app.client import VoiceClient
from app.config import Settings
from app.conversation import AudioDeviceError, Conversation, State, Stats

log = logging.getLogger("homepilot")

AUDIO_RETRY_DELAYS = (1.0, 2.0, 5.0, 10.0)
HEALTHY_RUN_S = 60.0
EXPECTED_AUDIO_ERRORS = (TimeoutError, AudioDeviceError, LookupError, sd.PortAudioError)


def retry_delay(failures: int) -> float:
    return AUDIO_RETRY_DELAYS[min(failures, len(AUDIO_RETRY_DELAYS) - 1)]


class RunMonitor:
    def __init__(self, stats: Stats, started: float):
        self.stats = stats
        self.started = started
        self.restarts = 0
        self.dropped_frames = 0
        self._process = psutil.Process()
        self._process.cpu_percent(None)

    def summary(self) -> str:
        s = self.stats
        hours = (asyncio.get_running_loop().time() - self.started) / 3600
        cpu = self._process.cpu_percent(None) / (psutil.cpu_count() or 1)
        rss = self._process.memory_info().rss / 2**20
        return (
            f"运行 {hours:.2f} h, 唤醒 {s.wakes} 次, 对话 {s.turns} 轮, 疑似误唤醒 {s.false_wakes} 次, "
            f"断线 {s.disconnects} 次, 音频重启 {self.restarts} 次, 丢帧 {self.dropped_frames}, "
            f"内存 {rss:.1f} MB, CPU {cpu:.1f}%"
        )


async def _report_periodically(monitor: RunMonitor, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        log.info("运行统计: %s", monitor.summary())


class AssistantSupervisor:
    """管理唤醒、对话循环的生命周期，并向 UI 暴露状态。"""

    def __init__(
        self,
        settings: Settings,
        *,
        board: BoardStore | None = None,
        on_state_change: Callable[[State], None] | None = None,
        on_connection_change: Callable[[bool], None] | None = None,
    ):
        self.settings = settings
        self.board = board or BoardStore()
        self._on_state_change = on_state_change
        self._on_connection_change = on_connection_change
        self.stats = Stats()
        self.conversation_state = State.IDLE
        self._client: VoiceClient | None = None
        self._task: asyncio.Task | None = None
        self._monitor: RunMonitor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def connected(self) -> bool:
        return self._client.connected if self._client is not None else False

    def get_status(self) -> dict:
        monitor = self._monitor
        hours = 0.0
        if monitor is not None and self._loop is not None:
            hours = (self._loop.time() - monitor.started) / 3600
        return {
            "running": self.running,
            "connected": self.connected,
            "conversation_state": self.conversation_state.value,
            "stats": {
                "wakes": self.stats.wakes,
                "turns": self.stats.turns,
                "false_wakes": self.stats.false_wakes,
                "disconnects": self.stats.disconnects,
                "audio_restarts": monitor.restarts if monitor else 0,
                "run_hours": round(hours, 2),
            },
            "board": self.board.to_json(),
        }

    async def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.create_task(self._run(), name="assistant-supervisor")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def run_forever(self) -> int:
        """CLI 入口：运行直到取消。"""
        await self.start()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        return 0

    def _state_hook(self, state: State) -> None:
        self.conversation_state = state
        if self._on_state_change is not None:
            self._on_state_change(state)

    def _connection_hook(self, connected: bool) -> None:
        if self._on_connection_change is not None:
            self._on_connection_change(connected)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        stats = self.stats
        started = self._loop.time()
        with WakeWordDetector(self.settings) as kws, VADDetector(self.settings) as vad:
            log.info("唤醒词: %s（Ctrl+C 退出）", "、".join(kws.keywords))
            async with VoiceClient(self.settings) as client:
                self._client = client
                monitor = RunMonitor(stats, started)
                self._monitor = monitor
                reporter = asyncio.create_task(_report_periodically(monitor, self.settings.stats_interval_s))
                failures = 0
                try:
                    while True:
                        run_started = self._loop.time()
                        cap = None
                        try:
                            async with AudioCapture(self.settings) as cap:
                                d = cap.device
                                log.info(
                                    "麦克风: #%d %s | %s | %d Hz x %d ch",
                                    d.index,
                                    d.name,
                                    d.hostapi,
                                    cap.samplerate,
                                    cap.channels,
                                )
                                conv = Conversation(
                                    self.settings,
                                    cap,
                                    kws,
                                    vad,
                                    client,
                                    stats=stats,
                                    board=self.board,
                                    on_state_change=self._state_hook,
                                    on_connection_change=self._connection_hook,
                                )
                                await conv.run()
                        except EXPECTED_AUDIO_ERRORS as e:
                            log.warning("音频异常: %s", e or type(e).__name__)
                        except Exception:
                            log.exception("运行出错")
                        finally:
                            if cap is not None:
                                monitor.dropped_frames += cap.dropped_frames
                        if self._loop.time() - run_started >= HEALTHY_RUN_S:
                            failures = 0
                        delay = retry_delay(failures)
                        failures += 1
                        monitor.restarts += 1
                        kws.reset()
                        vad.reset()
                        log.warning("%g 秒后重新初始化音频设备（第 %d 次重启）", delay, monitor.restarts)
                        await asyncio.sleep(delay)
                        devices.refresh_portaudio()
                finally:
                    reporter.cancel()
                    log.info("退出统计: %s", monitor.summary())
        self._client = None
        self.conversation_state = State.IDLE


async def run_assistant(settings: Settings, **kwargs) -> int:
    """兼容 CLI：等价于常驻 AssistantSupervisor。"""
    supervisor = AssistantSupervisor(settings, **kwargs)
    return await supervisor.run_forever()