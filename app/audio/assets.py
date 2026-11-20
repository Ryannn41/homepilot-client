"""内置语音资源（16 kHz 单声道 PCM16）。"""

from functools import lru_cache
from pathlib import Path

from app.config import SAMPLE_RATE, SAMPLE_WIDTH

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_WAKE_PCM = "wake_wozai.pcm"
_SILENCE_FAREWELL_PCM = "silence_farewell.pcm"


def _load_pcm_file(filename: str, label: str, *, max_seconds: float = 3.0) -> bytes:
    path = _ASSETS / filename
    if not path.is_file():
        raise FileNotFoundError(f"未找到{label}资源: {path}")
    pcm = path.read_bytes()
    if len(pcm) % SAMPLE_WIDTH:
        raise ValueError(f"{path}: PCM16 字节数必须是偶数")
    samples = len(pcm) // SAMPLE_WIDTH
    if samples < SAMPLE_RATE // 10 or samples > int(SAMPLE_RATE * max_seconds):
        raise ValueError(f"{path}: 时长异常（{samples / SAMPLE_RATE:.2f}s）")
    return pcm


@lru_cache(maxsize=1)
def load_wake_response() -> bytes:
    """唤醒成功后播放的「我在」语音。"""
    return _load_pcm_file(_WAKE_PCM, "唤醒语音")


@lru_cache(maxsize=1)
def load_silence_farewell() -> bytes:
    """静默超时回到等待唤醒时播放的「有需要再叫我」。"""
    return _load_pcm_file(_SILENCE_FAREWELL_PCM, "静默告别语音", max_seconds=4.0)