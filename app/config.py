from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 内部音频与 WebSocket 音频的统一格式，不可配置
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # PCM16


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HOMEPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    server_url: str = "ws://127.0.0.1:8000/ws/voice/"
    device_id: str = "homepilot-dev"
    auth_token: SecretStr = SecretStr("")

    # 空字符串表示使用系统默认设备；可填设备索引或名称子串
    input_device: str = ""
    output_device: str = ""
    audio_hostapi: str = "wasapi"
    frame_ms: int = 20

    # 唤醒词（sherpa-onnx KWS）：模型文件名相对于 kws_model_dir
    kws_model_dir: str = "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    kws_encoder: str = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    kws_decoder: str = "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    kws_joiner: str = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    kws_tokens: str = "tokens.txt"
    kws_keywords_file: str = "keywords.txt"
    kws_keywords_score: float = Field(default=1.0, gt=0)
    kws_keywords_threshold: float = Field(default=0.25, gt=0, lt=1)
    kws_num_threads: int = Field(default=1, ge=1)

    # 语音活动检测（sherpa-onnx Silero VAD）
    vad_model: str = "models/silero_vad.onnx"
    vad_threshold: float = Field(default=0.5, gt=0, lt=1)
    vad_min_speech_ms: int = Field(default=250, ge=32)  # 连续语音达到该时长才判定开始
    vad_min_silence_ms: int = Field(default=700, ge=100)  # 静音持续该时长才判定结束
    vad_max_speech_s: float = Field(default=30.0, gt=0, le=120)
    preroll_ms: int = Field(default=500, ge=200, le=1000)

    silence_timeout_s: float = Field(default=15.0, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("frame_ms")
    @classmethod
    def _check_frame_ms(cls, v: int) -> int:
        if v not in (20, 40):
            raise ValueError("frame_ms 只能是 20 或 40")
        return v

    @property
    def frame_samples(self) -> int:
        return SAMPLE_RATE * self.frame_ms // 1000

    @property
    def preroll_frames(self) -> int:
        return self.preroll_ms // self.frame_ms
