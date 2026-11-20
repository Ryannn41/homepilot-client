import logging
import sys

from pydantic import ValidationError

from app import __version__
from app.config import SAMPLE_RATE, Settings

log = logging.getLogger("homepilot")


def main() -> int:
    try:
        settings = Settings()
    except ValidationError as e:
        print(f"配置错误:\n{e}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("HomePilot Client %s 启动", __version__)
    log.info("服务端地址: %s", settings.server_url)
    log.info("设备 ID: %s", settings.device_id)
    log.info("认证令牌: %s", "已配置" if settings.auth_token.get_secret_value() else "未配置")
    log.info(
        "音频: 输入=%s 输出=%s HostAPI=%s",
        settings.input_device or "系统默认",
        settings.output_device or "系统默认",
        settings.audio_hostapi,
    )
    log.info(
        "内部格式: %d Hz 单声道 PCM16, 分片 %d ms (%d samples)",
        SAMPLE_RATE,
        settings.frame_ms,
        settings.frame_samples,
    )
    log.info("静默超时: %.1f s", settings.silence_timeout_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
