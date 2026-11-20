import asyncio
import logging

import numpy as np
import sounddevice as sd

from app.audio.devices import Device, find_input_device
from app.audio.resample import StreamResampler
from app.config import SAMPLE_RATE, Settings

log = logging.getLogger(__name__)


def choose_input_format(device: Device, channels: int | None = None) -> tuple[int, int]:
    """返回设备实际接受的 (采样率, 声道数)，优先设备默认采样率，避免系统层再做一次重采样。"""
    rates = dict.fromkeys([int(device.default_samplerate), 48000, 44100, SAMPLE_RATE])
    channel_options = [channels] if channels else list(dict.fromkeys([1, device.max_input_channels]))
    for rate in rates:
        for ch in channel_options:
            try:
                sd.check_input_settings(device=device.index, samplerate=rate, channels=ch, dtype="int16")
            except Exception:
                continue
            return rate, ch
    raise RuntimeError(f"输入设备 #{device.index} {device.name} 没有可用的采样率/声道组合")


def downmix(block: np.ndarray) -> np.ndarray:
    """(frames, channels) int16 -> (frames,) int16，多声道取平均。"""
    if block.shape[1] == 1:
        return block[:, 0]
    return np.rint(block.mean(axis=1)).astype(np.int16)


class AudioCapture:
    """采集麦克风音频，输出 16kHz 单声道 PCM16 定长帧（bytes，frame_ms 时长）。

    PortAudio 回调只复制数据并投递到事件循环；下混、重采样、切帧在事件循环线程完成。
    队列满时丢弃最旧的帧并计入 dropped_frames。stop() 时不足一帧的尾部被丢弃。
    """

    def __init__(self, settings: Settings, queue_size: int = 50, channels: int | None = None):
        self._settings = settings
        self._queue_size = queue_size
        self._force_channels = channels
        self.frame_samples = settings.frame_samples

        self.device: Device | None = None
        self.samplerate = 0
        self.channels = 0
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)
        self.dropped_frames = 0
        self.overflow_count = 0
        self.frames_produced = 0

        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._resampler: StreamResampler | None = None
        self._pending = np.zeros(0, dtype=np.int16)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """必须在运行中的事件循环内调用。"""
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self.device = find_input_device(self._settings.input_device, self._settings.audio_hostapi)
        self.samplerate, self.channels = choose_input_format(self.device, self._force_channels)
        self._resampler = StreamResampler(self.samplerate, SAMPLE_RATE)
        self._pending = np.zeros(0, dtype=np.int16)
        self.queue = asyncio.Queue(maxsize=self._queue_size)

        stream = sd.InputStream(
            device=self.device.index,
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.samplerate * self._settings.frame_ms // 1000,
            callback=self._callback,
        )
        try:
            self._running = True
            stream.start()
        except BaseException:
            self._running = False
            stream.close()
            raise
        self._stream = stream
        log.info(
            "开始采集: #%d %s | %s | %d Hz x %d ch",
            self.device.index, self.device.name, self.device.hostapi, self.samplerate, self.channels,
        )

    def stop(self) -> None:
        self._running = False
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()
        log.info("停止采集: 产出 %d 帧, 丢弃 %d 帧, 输入溢出 %d 次",
                 self.frames_produced, self.dropped_frames, self.overflow_count)

    async def read(self) -> bytes:
        return await self.queue.get()

    async def __aenter__(self) -> "AudioCapture":
        self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        self.stop()

    def _callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
        # PortAudio 线程：只做复制和投递
        try:
            self._loop.call_soon_threadsafe(self._on_block, indata.copy(), bool(status.input_overflow))
        except RuntimeError:
            pass  # 事件循环已关闭

    def _on_block(self, block: np.ndarray, overflow: bool) -> None:
        if not self._running:
            return
        if overflow:
            self.overflow_count += 1
        out = self._resampler.process(downmix(block))
        pending = np.concatenate([self._pending, out]) if len(self._pending) else out
        n = self.frame_samples
        full = len(pending) // n
        for i in range(full):
            self._put(pending[i * n : (i + 1) * n].tobytes())
        self._pending = pending[full * n :].copy()

    def _put(self, frame: bytes) -> None:
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped_frames += 1
        self.queue.put_nowait(frame)
        self.frames_produced += 1
