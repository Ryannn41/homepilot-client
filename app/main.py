import argparse
import asyncio
import logging
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from pydantic import ValidationError

from app import __version__
from app.audio import devices
from app.audio.capture import AudioCapture
from app.audio.player import AudioPlayer
from app.audio.vad import SpeechAudio, SpeechStart, VADDetector
from app.audio.wakeword import WakeWordDetector
from app.config import SAMPLE_RATE, SAMPLE_WIDTH, Settings

log = logging.getLogger("homepilot")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.main", description="HomePilot Client")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list-devices", action="store_true", help="列出所有音频设备")
    group.add_argument("--probe", action="store_true", help="探测设备支持的采样率和声道")
    group.add_argument("--record-test", type=float, metavar="SECONDS", help="录音测试：打印电平并保存 WAV")
    group.add_argument("--play-test", type=float, metavar="SECONDS", help="播放 440 Hz 测试音")
    group.add_argument(
        "--kws-test", type=float, nargs="?", const=0.0, metavar="SECONDS",
        help="从麦克风检测唤醒词；可选时长，省略则一直运行到 Ctrl+C",
    )
    group.add_argument(
        "--vad-test", type=float, nargs="?", const=0.0, metavar="SECONDS",
        help="从麦克风检测语音片段；可选时长，省略则一直运行到 Ctrl+C",
    )
    parser.add_argument("--output", default="record_test.wav", help="--record-test 的 WAV 输出路径")
    parser.add_argument("--vad-save", metavar="DIR", help="--vad-test 时把每个语音片段保存为 WAV 到该目录")
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

    _write_wav(output, b"".join(frames))
    total = sum(len(f) for f in frames) // SAMPLE_WIDTH
    print(f"完成: {len(frames)} 帧, {total} 样本 ({total / SAMPLE_RATE:.2f} s), "
          f"丢帧 {dropped}, 输入溢出 {overflows}")
    print(f"已保存: {output}")
    return 0


def make_tone(seconds: float, freq: float = 440.0, amplitude: float = 0.1) -> np.ndarray:
    """16kHz 正弦测试音，首尾 20ms 淡入淡出避免爆音。"""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    tone = amplitude * 32767 * np.sin(2 * np.pi * freq * t)
    fade = min(n // 2, SAMPLE_RATE // 50)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[n - fade :] *= ramp[::-1]
    return np.rint(tone).astype(np.int16)


async def play_test(settings: Settings, seconds: float) -> int:
    tone = make_tone(seconds)
    loop = asyncio.get_running_loop()
    async with AudioPlayer(settings) as player:
        d = player.device
        print(f"设备: #{d.index} {d.name} | {d.hostapi} | {player.samplerate} Hz x {player.channels} ch")
        print(f"播放 {seconds:g} 秒 440 Hz 测试音...")
        started = loop.time()
        completed = await player.play(tone)
        elapsed = loop.time() - started
    print(f"{'完成' if completed else '已停止'}: 耗时 {elapsed:.2f} s, 欠载 {player.underruns} 次")
    return 0 if completed else 1


async def kws_test(settings: Settings, seconds: float) -> int:
    level_every = 3000 // settings.frame_ms
    max_frames = int(seconds * 1000 / settings.frame_ms) if seconds > 0 else None
    detections = 0
    with WakeWordDetector(settings) as detector:
        async with AudioCapture(settings) as cap:
            d = cap.device
            print(f"设备: #{d.index} {d.name} | {d.hostapi} | {cap.samplerate} Hz x {cap.channels} ch")
            print(f"关键词: {', '.join(detector.keywords)}；请说出唤醒词（Ctrl+C 退出）")
            recent: list[bytes] = []
            frames = 0
            while max_frames is None or frames < max_frames:
                frame = await asyncio.wait_for(cap.read(), timeout=2.0)
                frames += 1
                event = await detector.process(frame)
                if event:
                    detections += 1
                    print(f">>> 唤醒: {event.keyword}  帧 #{event.frame_index}  "
                          f"t={event.frame_index * settings.frame_ms / 1000:.2f}s")
                    detector.reset()
                recent.append(frame)
                if len(recent) == level_every:
                    print(f"    {_level(np.frombuffer(b''.join(recent), dtype=np.int16))}")
                    recent.clear()
    print(f"结束: 共 {frames} 帧, 唤醒 {detections} 次, 丢帧 {cap.dropped_frames}")
    return 0


async def vad_test(settings: Settings, seconds: float, save_dir: str | None) -> int:
    ms = settings.frame_ms
    max_frames = int(seconds * 1000 / ms) if seconds > 0 else None
    out = Path(save_dir) if save_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    segments, speech_s, streamed = 0, 0.0, 0
    costs: list[float] = []
    with VADDetector(settings) as vad:
        async with AudioCapture(settings) as cap:
            d = cap.device
            print(f"设备: #{d.index} {d.name} | {d.hostapi} | {cap.samplerate} Hz x {cap.channels} ch")
            print(f"预缓冲 {settings.preroll_frames} 帧 ({settings.preroll_frames * ms} ms)，"
                  f"结束静音 {settings.vad_min_silence_ms} ms；请说话（Ctrl+C 退出）")
            frames = 0
            while max_frames is None or frames < max_frames:
                frame = await asyncio.wait_for(cap.read(), timeout=2.0)
                frames += 1
                t0 = time.perf_counter()
                events = vad.process(frame)
                costs.append(time.perf_counter() - t0)
                for ev in events:
                    if isinstance(ev, SpeechStart):
                        streamed = len(ev.frames)
                        print(f">>> 语音开始  t={ev.detected_frame * ms / 1000:.2f}s  "
                              f"片段起点 t={ev.start_frame * ms / 1000:.2f}s  预缓冲 {len(ev.frames)} 帧")
                    elif isinstance(ev, SpeechAudio):
                        streamed += 1
                    else:
                        segments += 1
                        speech_s += ev.duration_s
                        note = "（达到最长时长，已截断）" if ev.truncated else ""
                        print(f"<<< 语音结束  t={ev.end_frame * ms / 1000:.2f}s  时长 {ev.duration_s:.2f}s  "
                              f"{len(ev.frames)} 帧  流式交付 {streamed} 帧{note}")
                        print(f"    {_level(np.frombuffer(ev.pcm, dtype=np.int16))}")
                        if out:
                            path = out / f"segment_{segments:03d}.wav"
                            _write_wav(path, ev.pcm)
                            print(f"    已保存: {path}")
    c = np.array(costs) * 1000 if costs else np.zeros(1)
    print(f"结束: 共 {frames} 帧 ({frames * ms / 1000:.1f}s), 语音片段 {segments} 个, 语音总时长 {speech_s:.2f}s, "
          f"丢帧 {cap.dropped_frames}")
    print(f"VAD 耗时/帧: 平均 {c.mean():.3f} ms, 最大 {c.max():.3f} ms")
    return 0


def _write_wav(path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


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
        return _run_audio(record_test(settings, args.record_test, args.output), "采集", "2 秒内没有收到音频帧")
    if args.play_test is not None:
        if args.play_test <= 0:
            print("--play-test 必须大于 0", file=sys.stderr)
            return 2
        return _run_audio(play_test(settings, args.play_test), "播放", "播放未在预期时间内结束")
    if args.kws_test is not None:
        if args.kws_test < 0:
            print("--kws-test 时长不能为负数", file=sys.stderr)
            return 2
        try:
            return _run_audio(kws_test(settings, args.kws_test), "唤醒检测", "2 秒内没有收到音频帧")
        except (FileNotFoundError, ValueError) as e:
            print(f"唤醒检测失败: {e}", file=sys.stderr)
            return 1
    if args.vad_test is not None:
        if args.vad_test < 0:
            print("--vad-test 时长不能为负数", file=sys.stderr)
            return 2
        try:
            return _run_audio(vad_test(settings, args.vad_test, args.vad_save), "语音检测", "2 秒内没有收到音频帧")
        except (FileNotFoundError, ValueError) as e:
            print(f"语音检测失败: {e}", file=sys.stderr)
            return 1

    log_startup(settings)
    return 0


def _run_audio(coro, action: str, timeout_message: str) -> int:
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except TimeoutError:
        print(f"{action}失败: {timeout_message}", file=sys.stderr)
        return 1
    except (LookupError, RuntimeError, sd.PortAudioError) as e:
        print(f"{action}失败: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
