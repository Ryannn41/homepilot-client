import asyncio
import logging
import threading
from collections import deque

import numpy as np
import sounddevice as sd

from app.audio.devices import Device, find_output_device
from app.audio.resample import StreamResampler
from app.config import SAMPLE_RATE, Settings

log = logging.getLogger(__name__)


def to_pcm16(data) -> np.ndarray:
    """把内部格式音频（16kHz 单声道 PCM16 的 bytes，或一维 np.int16）转为 np.int16 数组。"""
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
        if len(raw) % 2:
            raise ValueError("PCM16 字节数必须是偶数")
        return np.frombuffer(raw, dtype=np.int16)
    if isinstance(data, np.ndarray) and data.dtype == np.int16 and data.ndim == 1:
        return data
    raise TypeError("音频必须是 PCM16 bytes 或一维 np.int16 数组")


def candidate_output_formats(device: Device, channels: int | None = None) -> list[tuple[int, int]]:
    """通过 check_output_settings 的 (采样率, 声道数) 候选，按优先级排序。

    优先双声道：由本地把单声道复制到左右声道，不依赖驱动的单声道映射。
    """
    rates = dict.fromkeys([int(device.default_samplerate), 48000, 44100, SAMPLE_RATE])
    if channels:
        channel_options = [channels]
    else:
        channel_options = list(dict.fromkeys([min(2, device.max_output_channels), 1]))
    result = []
    for rate in rates:
        for ch in channel_options:
            try:
                sd.check_output_settings(device=device.index, samplerate=rate, channels=ch, dtype="int16")
            except Exception:
                continue
            result.append((rate, ch))
    return result


class AudioPlayer:
    """把 16kHz 单声道 PCM16 播放到输出设备。

    一次播放：start() -> write() 若干次 -> drain()，或直接 await play(pcm)。
    重采样与声道复制在事件循环线程完成；PortAudio 回调只从缓冲区拷贝数据。
    """

    def __init__(self, settings: Settings, channels: int | None = None, max_buffer_ms: int = 500):
        self._settings = settings
        self._force_channels = channels
        self._max_buffer_ms = max_buffer_ms

        self.device: Device | None = None
        self.samplerate = 0
        self.channels = 0
        self.underruns = 0

        self._stream: sd.OutputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._resampler: StreamResampler | None = None
        self._finished: asyncio.Event | None = None
        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._offset = 0
        self._buffered = 0
        self._eof = False
        self._written = False
        self._aborted = False

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """打开输出流。必须在运行中的事件循环内调用；已启动时忽略。"""
        if self._stream is not None:
            return
        self._loop = asyncio.get_running_loop()
        self.device = find_output_device(self._settings.output_device, self._settings.audio_hostapi)
        self._finished = asyncio.Event()
        self._chunks.clear()
        self._offset = self._buffered = self.underruns = 0
        self._eof = self._written = self._aborted = False

        last_error: Exception | None = None
        for rate, ch in candidate_output_formats(self.device, self._force_channels):
            self.samplerate, self.channels = rate, ch
            self._resampler = StreamResampler(SAMPLE_RATE, rate)
            try:
                stream = sd.OutputStream(
                    device=self.device.index,
                    samplerate=rate,
                    channels=ch,
                    dtype="int16",
                    blocksize=rate * self._settings.frame_ms // 1000,
                    callback=self._callback,
                    finished_callback=self._on_finished,
                )
            except Exception as e:
                last_error = e
                log.warning("输出格式 %d Hz x %d ch 打开失败: %s", rate, ch, e)
                continue
            try:
                stream.start()
            except Exception as e:
                stream.close()
                last_error = e
                log.warning("输出格式 %d Hz x %d ch 启动失败: %s", rate, ch, e)
                continue
            self._stream = stream
            log.info(
                "开始播放: #%d %s | %s | %d Hz x %d ch",
                self.device.index, self.device.name, self.device.hostapi, rate, ch,
            )
            return
        raise RuntimeError(f"输出设备 #{self.device.index} {self.device.name} 没有可用的输出格式") from last_error

    async def write(self, data) -> None:
        """写入一块 16kHz 单声道 PCM16；缓冲超过 max_buffer_ms 时异步等待。"""
        pcm = to_pcm16(data)
        if self._stream is None or self._eof:
            raise RuntimeError("播放器未启动、已停止或已 drain")
        if len(pcm) == 0:
            return
        self._push(self._resampler.process(pcm))
        limit = self.samplerate * self._max_buffer_ms // 1000
        while self._stream is not None and self._buffered > limit:
            await asyncio.sleep(self._settings.frame_ms / 4000)

    async def drain(self) -> bool:
        """输出重采样尾部并等待播放完毕后关闭流。完整播完返回 True，被 stop() 打断返回 False。"""
        if self._stream is None:
            return False
        if not self._eof:
            self._push(self._resampler.flush())
            with self._lock:
                self._eof = True
        timeout = self._buffered / self.samplerate + 2.0
        await asyncio.wait_for(self._finished.wait(), timeout)
        completed = not self._aborted
        self._release(abort=False)
        return completed

    def stop(self) -> None:
        """立即停止并释放输出流；可重复调用。"""
        self._release(abort=True)

    async def play(self, data, chunk_samples: int = 320) -> bool:
        """完整播放一段音频；返回是否播完。异常或取消时也会释放音频流。"""
        pcm = to_pcm16(data)
        if len(pcm) == 0:
            return True
        self.start()
        try:
            for i in range(0, len(pcm), chunk_samples):
                if self._stream is None:
                    return False
                await self.write(pcm[i : i + chunk_samples])
            return await self.drain()
        finally:
            self.stop()

    async def __aenter__(self) -> "AudioPlayer":
        self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        self.stop()

    def _push(self, out: np.ndarray) -> None:
        if len(out) == 0:
            return
        frames = np.repeat(out[:, None], self.channels, axis=1)
        with self._lock:
            self._chunks.append(frames)
            self._buffered += len(frames)
            self._written = True

    def _release(self, abort: bool) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        if abort:
            self._aborted = True
        try:
            if abort:
                stream.abort()
            else:
                stream.stop()
        finally:
            stream.close()
            with self._lock:
                self._chunks.clear()
                self._buffered = 0
            self._finished.set()
            log.info("停止播放: 欠载 %d 次%s", self.underruns, "（被中止）" if abort else "")

    def _callback(self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
        # PortAudio 线程：只做内存拷贝
        with self._lock:
            filled = 0
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                take = min(frames - filled, len(chunk) - self._offset)
                outdata[filled : filled + take] = chunk[self._offset : self._offset + take]
                filled += take
                self._offset += take
                if self._offset == len(chunk):
                    self._chunks.popleft()
                    self._offset = 0
            self._buffered -= filled
            eof = self._eof
            if filled < frames and self._written and not eof:
                self.underruns += 1
        if filled < frames:
            outdata[filled:] = 0
            if eof:
                raise sd.CallbackStop

    def _on_finished(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._finished.set)
        except RuntimeError:
            pass  # 事件循环已关闭
