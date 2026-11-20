import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from app.config import SAMPLE_RATE, Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WakeEvent:
    keyword: str
    frame_index: int  # 触发时所在帧的序号，检测器生命周期内单调递增，从 0 开始
    stream_time_s: float  # 自上次 reset() 以来送入的音频时长
    detected_at: float  # time.monotonic()


def pcm16_to_float32(frame) -> np.ndarray:
    """16kHz 单声道 PCM16（bytes 或一维 np.int16）-> [-1, 1) 的 float32。"""
    if isinstance(frame, (bytes, bytearray, memoryview)):
        raw = bytes(frame)
        if len(raw) % 2:
            raise ValueError("PCM16 字节数必须是偶数")
        pcm = np.frombuffer(raw, dtype=np.int16)
    elif isinstance(frame, np.ndarray) and frame.dtype == np.int16 and frame.ndim == 1:
        pcm = frame
    else:
        raise TypeError("音频帧必须是 PCM16 bytes 或一维 np.int16 数组")
    return pcm.astype(np.float32) / 32768.0


def model_files(settings: Settings) -> dict[str, Path]:
    d = Path(settings.kws_model_dir)
    return {
        "encoder": d / settings.kws_encoder,
        "decoder": d / settings.kws_decoder,
        "joiner": d / settings.kws_joiner,
        "tokens": d / settings.kws_tokens,
        "keywords": Path(settings.kws_keywords_file),
    }


def check_keywords(keywords_file: Path, tokens_file: Path) -> list[str]:
    """校验关键词文件中的 token 都在模型 tokens.txt 中，返回关键词原文列表。

    sherpa-onnx 底层遇到无法编码的关键词可能直接终止进程，因此提前在 Python 层检查。
    """
    vocab = {line.split()[0] for line in tokens_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    keywords = []
    for lineno, line in enumerate(keywords_file.read_text(encoding="utf-8").splitlines(), 1):
        items = line.split()
        if not items:
            continue
        tokens = [t for t in items if t[0] not in ":#@"]
        unknown = [t for t in tokens if t not in vocab]
        if not tokens or unknown:
            raise ValueError(f"{keywords_file} 第 {lineno} 行包含模型不认识的 token: {unknown or line!r}")
        name = next((t[1:] for t in items if t.startswith("@")), " ".join(tokens))
        keywords.append(name)
    if not keywords:
        raise ValueError(f"{keywords_file} 中没有关键词")
    return keywords


class WakeWordDetector:
    """sherpa-onnx 流式唤醒词检测。

    accept() 同步处理一帧；process() 在专用单线程中调用 accept()，不阻塞事件循环
    （实测 sherpa-onnx 解码时释放 GIL）。检测到唤醒词后立即重置解码流，避免同一段语音重复触发。
    """

    def __init__(self, settings: Settings):
        files = model_files(settings)
        missing = [f"{name}: {path}" for name, path in files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("KWS 文件不存在（下载方式见 README）:\n  " + "\n  ".join(missing))
        self.keywords = check_keywords(files["keywords"], files["tokens"])
        try:
            self._kws = sherpa_onnx.KeywordSpotter(
                tokens=str(files["tokens"]),
                encoder=str(files["encoder"]),
                decoder=str(files["decoder"]),
                joiner=str(files["joiner"]),
                keywords_file=str(files["keywords"]),
                num_threads=settings.kws_num_threads,
                keywords_score=settings.kws_keywords_score,
                keywords_threshold=settings.kws_keywords_threshold,
            )
            self._stream = self._kws.create_stream()
        except Exception as e:
            raise RuntimeError(f"KWS 模型加载失败: {e}") from e

        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kws")
        self._frame_index = -1
        self._stream_samples = 0
        log.info("KWS 已加载: %s, 关键词 %s", settings.kws_model_dir, self.keywords)

    def accept(self, frame) -> WakeEvent | None:
        """送入一帧 16kHz 单声道 PCM16，检测到唤醒词时返回 WakeEvent。"""
        samples = pcm16_to_float32(frame)
        with self._lock:
            if self._executor is None:
                raise RuntimeError("WakeWordDetector 已关闭")
            self._frame_index += 1
            self._stream_samples += len(samples)
            if len(samples) == 0:
                return None
            self._stream.accept_waveform(SAMPLE_RATE, samples)
            while self._kws.is_ready(self._stream):
                self._kws.decode_stream(self._stream)
                keyword = self._kws.get_result(self._stream)
                if keyword:
                    event = WakeEvent(
                        keyword=keyword,
                        frame_index=self._frame_index,
                        stream_time_s=self._stream_samples / SAMPLE_RATE,
                        detected_at=time.monotonic(),
                    )
                    self._kws.reset_stream(self._stream)
                    return event
        return None

    async def process(self, frame) -> WakeEvent | None:
        if self._executor is None:
            raise RuntimeError("WakeWordDetector 已关闭")
        return await asyncio.get_running_loop().run_in_executor(self._executor, self.accept, frame)

    def reset(self) -> None:
        """丢弃全部解码状态，重新开始检测；帧序号保持递增。"""
        with self._lock:
            self._stream = self._kws.create_stream()
            self._stream_samples = 0

    def close(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "WakeWordDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
