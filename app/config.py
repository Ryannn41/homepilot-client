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
