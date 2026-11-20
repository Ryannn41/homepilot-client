import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import sherpa_onnx

from app.config import SAMPLE_RATE, Settings

log = logging.getLogger(__name__)

WINDOW_SIZE = 512  # Silero VAD 在 16kHz 下的窗口长度


@dataclass(frozen=True)
class SpeechStart:
    """检测到语音开始。frames 为预缓冲中的帧（含当前帧），按时间顺序。"""

    kind: ClassVar[str] = "speech_start"
    start_frame: int  # 片段第一帧（预缓冲最早一帧）的序号
    detected_frame: int  # VAD 判定开始时的帧序号
    frames: tuple[bytes, ...]
    frame_ms: int

    @property
    def duration_s(self) -> float:
        return len(self.frames) * self.frame_ms / 1000


@dataclass(frozen=True)
class SpeechAudio:
    """语音进行中的一帧，紧随 SpeechStart.frames 之后。"""

    kind: ClassVar[str] = "speech_audio"
    frame_index: int
    frame: bytes


@dataclass(frozen=True)
class SpeechEnd:
    """语音结束。frames 为完整片段 = SpeechStart.frames + 全部 SpeechAudio 帧。"""

    kind: ClassVar[str] = "speech_end"
    start_frame: int
    end_frame: int
    frames: tuple[bytes, ...]
    frame_ms: int
    truncated: bool  # 达到最长时长被强制结束

    @property
    def duration_s(self) -> float:
        return len(self.frames) * self.frame_ms / 1000

    @property
    def pcm(self) -> bytes:
        return b"".join(self.frames)


VadEvent = SpeechStart | SpeechAudio | SpeechEnd


class VADDetector:
    """基于 sherpa-onnx Silero VAD 的语音片段检测，带预缓冲。

    process() 每次处理一帧 frame_ms 的 16kHz 单声道 PCM16，返回本帧产生的事件：
    - 空闲时帧进入预缓冲；VAD 判定开始时返回 SpeechStart（携带预缓冲帧），预缓冲随即清空。
    - 语音中每帧返回 SpeechAudio；静音持续 vad_min_silence_ms 后再返回 SpeechEnd。
    片段中的每一帧只通过 SpeechStart.frames 或 SpeechAudio 交付一次。

    单帧推理实测约 0.06 ms，直接在事件循环中同步调用即可。
    """

    def __init__(self, settings: Settings):
        model = Path(settings.vad_model)
        if not model.is_file():
            raise FileNotFoundError(f"VAD 模型不存在: {model}（下载方式见 README）")
        self._frame_ms = settings.frame_ms
        self._frame_bytes = settings.frame_samples * 2
        self._max_frames = int(settings.vad_max_speech_s * 1000) // settings.frame_ms
        if settings.preroll_ms < settings.vad_min_speech_ms + 2 * WINDOW_SIZE * 1000 // SAMPLE_RATE:
            log.warning("preroll_ms=%d 可能不足以覆盖 VAD 开始检测延迟（约 min_speech + 64ms）", settings.preroll_ms)
        try:
            config = sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=str(model),
                    threshold=settings.vad_threshold,
                    min_silence_duration=settings.vad_min_silence_ms / 1000,
                    min_speech_duration=settings.vad_min_speech_ms / 1000,
                    window_size=WINDOW_SIZE,
                    max_speech_duration=settings.vad_max_speech_s,
                ),
                sample_rate=SAMPLE_RATE,
                num_threads=1,
            )
            self._vad = sherpa_onnx.VoiceActivityDetector(
                config, buffer_size_in_seconds=settings.vad_max_speech_s + 10
            )
        except Exception as e:
            raise RuntimeError(f"VAD 模型加载失败: {e}") from e

        self._preroll: deque[bytes] = deque(maxlen=settings.preroll_frames)
        self._pending = np.zeros(0, dtype=np.float32)
        self._segment: list[bytes] | None = None
        self._segment_start = 0
        self._wait_silence = False
        self._frame_index = -1
        self._closed = False
        log.info("VAD 已加载: %s", model)

    @property
    def in_speech(self) -> bool:
        return self._segment is not None

    def process(self, frame: bytes) -> list[VadEvent]:
        if self._closed:
            raise RuntimeError("VADDetector 已关闭")
        if not isinstance(frame, (bytes, bytearray)) or len(frame) != self._frame_bytes:
            raise ValueError(f"音频帧必须是 {self._frame_bytes} 字节的 PCM16 bytes")
        frame = bytes(frame)
        self._frame_index += 1
        speech = self._detect(frame)

        if self._segment is None:
            if self._wait_silence:
                # 强制截断后，等用户停顿再开始新片段，避免把截断片段的后续语音混入预缓冲
                self._wait_silence = speech
                return []
            self._preroll.append(frame)
            if not speech:
                return []
            frames = tuple(self._preroll)
            self._preroll.clear()
            self._segment = list(frames)
            self._segment_start = self._frame_index - len(frames) + 1
            return [SpeechStart(self._segment_start, self._frame_index, frames, self._frame_ms)]

        self._segment.append(frame)
        events: list[VadEvent] = [SpeechAudio(self._frame_index, frame)]
        if not speech:
            events.append(self._end(truncated=False))
        elif len(self._segment) >= self._max_frames:
            events.append(self._end(truncated=True))
            self._wait_silence = True
        return events

    def reset(self) -> None:
        """清空预缓冲、未完成的片段和模型状态；帧序号保持递增。"""
        if self._closed:
            return
        self._vad.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._segment = None
        self._wait_silence = False

    def close(self) -> None:
        self._closed = True
        self._vad = None
        self._preroll.clear()
        self._segment = None

    def __enter__(self) -> "VADDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _detect(self, frame: bytes) -> bool:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, samples])
        try:
            while len(self._pending) >= WINDOW_SIZE:
                self._vad.accept_waveform(self._pending[:WINDOW_SIZE])
                self._pending = self._pending[WINDOW_SIZE:]
            while not self._vad.empty():  # 片段内容由本类自行保存，丢弃 sherpa 的副本
                self._vad.pop()
            return bool(self._vad.is_speech_detected())
        except Exception as e:
            raise RuntimeError(f"VAD 推理失败: {e}") from e

    def _end(self, truncated: bool) -> SpeechEnd:
        frames = tuple(self._segment)
        self._segment = None
        return SpeechEnd(self._segment_start, self._frame_index, frames, self._frame_ms, truncated)
