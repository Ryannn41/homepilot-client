import argparse
import asyncio
import logging
import sys
import wave

import numpy as np
import sounddevice as sd
from pydantic import ValidationError

from app import __version__
from app.audio import devices
from app.audio.capture import AudioCapture
from app.config import SAMPLE_RATE, SAMPLE_WIDTH, Settings

log = logging.getLogger("homepilot")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.main", description="HomePilot Client")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list-devices", action="store_true", help="列出所有音频设备")
    group.add_argument("--probe", action="store_true", help="探测设备支持的采样率和声道")
    group.add_argument("--record-test", type=float, metavar="SECONDS", help="录音测试：打印电平并保存 WAV")
    parser.add_argument("--output", default="record_test.wav", help="--record-test 的 WAV 输出路径")
    return parser.parse_args(argv)


def _level(pcm: np.ndarray) -> str:
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if len(pcm) else 0.0
    peak = int(np.max(np.abs(pcm.astype(np.int32)))) if len(pcm) else 0
    db = 20 * np.log10(max(rms, 1.0) / 32768)
    bar = "#" * int(max(0.0, db + 60) / 2)
    return f"RMS {db:6.1f} dBFS  峰值 {peak:5d}  |{bar:<30}|"


async def record_test(settings: Settings, seconds: float, output: str) -> int:
    n_frames = int(seconds * 1000 / settings.frame_ms)
    per_second = 1000 // settings.frame_ms
    frames: list[bytes] = []
    async with AudioCapture(settings) as cap:
        d = cap.device
        print(f"设备: #{d.index} {d.name} | {d.hostapi} | {cap.samplerate} Hz x {cap.channels} ch")
        print(f"录音 {seconds:g} 秒，目标 {n_frames} 帧（{settings.frame_ms} ms/帧）...")
        while len(frames) < n_frames:
            frames.append(await asyncio.wait_for(cap.read(), timeout=2.0))
            if len(frames) % per_second == 0:
                pcm = np.frombuffer(b"".join(frames[-per_second:]), dtype=np.int16)
                print(f"  {len(frames) // per_second:3d}s  {_level(pcm)}")
        dropped, overflows = cap.dropped_frames, cap.overflow_count

    with wave.open(output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"".join(frames))
    total = sum(len(f) for f in frames) // SAMPLE_WIDTH
    print(f"完成: {len(frames)} 帧, {total} 样本 ({total / SAMPLE_RATE:.2f} s), "
          f"丢帧 {dropped}, 输入溢出 {overflows}")
    print(f"已保存: {output}")
    return 0


def log_startup(settings: Settings) -> None:
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as e:
        print(f"配置错误:\n{e}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_devices:
        print(devices.format_device_list(devices.list_devices(settings.audio_hostapi)))
        return 0
    if args.probe:
        print(devices.format_probe_report(devices.list_devices(settings.audio_hostapi)))
        return 0
    if args.record_test is not None:
        if args.record_test <= 0:
            print("--record-test 必须大于 0", file=sys.stderr)
            return 2
        try:
            return asyncio.run(record_test(settings, args.record_test, args.output))
        except KeyboardInterrupt:
            print("已中断", file=sys.stderr)
            return 130
        except TimeoutError:
            print("采集失败: 2 秒内没有收到音频帧", file=sys.stderr)
            return 1
        except (LookupError, RuntimeError, sd.PortAudioError) as e:
            print(f"采集失败: {e}", file=sys.stderr)
            return 1

    log_startup(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
