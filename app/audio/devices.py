from dataclasses import dataclass
from enum import Enum
from typing import Literal

import sounddevice as sd

PROBE_SAMPLERATES = (16000, 44100, 48000)
PROBE_CHANNELS = (1, 2)
PROBE_DTYPE = "int16"

# PortAudio 错误码：paInvalidChannelCount / paInvalidSampleRate / paSampleFormatNotSupported
# 这些表示"格式明确不被接受"，其余错误视为探测失败
_UNSUPPORTED_CODES = {-9998, -9997, -9994}

_HOSTAPI_ORDER = ("wasapi", "mme", "directsound", "wdm-ks")

Kind = Literal["input", "output"]


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool
    is_default_output: bool


class Status(str, Enum):
    SUPPORTED = "支持"
    UNSUPPORTED = "不支持"
    FAILED = "探测失败"


@dataclass(frozen=True)
class ProbeResult:
    samplerate: int
    channels: int
    status: Status
    detail: str = ""


def hostapi_rank(hostapi: str, preferred: str = "wasapi") -> int:
    name = hostapi.lower()
    order = (preferred.lower(), *_HOSTAPI_ORDER) if preferred else _HOSTAPI_ORDER
    for rank, key in enumerate(order):
        if key in name:
            return rank
    return len(order)


def list_devices(preferred_hostapi: str = "wasapi") -> list[Device]:
    hostapis = sd.query_hostapis()
    devices = []
    for index, info in enumerate(sd.query_devices()):
        api = hostapis[info["hostapi"]]
        devices.append(
            Device(
                index=index,
                name=info["name"],
                hostapi=api["name"],
                max_input_channels=info["max_input_channels"],
                max_output_channels=info["max_output_channels"],
                default_samplerate=info["default_samplerate"],
                is_default_input=api["default_input_device"] == index,
                is_default_output=api["default_output_device"] == index,
            )
        )
    devices.sort(key=lambda d: (hostapi_rank(d.hostapi, preferred_hostapi), d.index))
    return devices


def refresh_portaudio() -> None:
    """重新初始化 PortAudio，使设备插拔、默认设备变化可见。调用前必须关闭所有音频流。"""
    sd._terminate()
    sd._initialize()


def find_input_device(spec: str = "", preferred_hostapi: str = "wasapi") -> Device:
    """按配置解析输入设备：空为首选 Host API 的默认输入，纯数字为索引，否则为名称子串。"""
    return _find_device(spec, preferred_hostapi, "input")


def find_output_device(spec: str = "", preferred_hostapi: str = "wasapi") -> Device:
    """按配置解析输出设备，规则同 find_input_device。"""
    return _find_device(spec, preferred_hostapi, "output")


def _find_device(spec: str, preferred_hostapi: str, kind: Kind) -> Device:
    label = "输入" if kind == "input" else "输出"

    def usable(d: Device) -> bool:
        return (d.max_input_channels if kind == "input" else d.max_output_channels) > 0

    def is_default(d: Device) -> bool:
        return d.is_default_input if kind == "input" else d.is_default_output

    candidates = [d for d in list_devices(preferred_hostapi) if usable(d)]
    if not candidates:
        raise LookupError(f"没有可用的{label}设备")
    spec = spec.strip()
    if not spec:
        return next((d for d in candidates if is_default(d)), candidates[0])
    if spec.isdigit():
        for d in candidates:
            if d.index == int(spec):
                return d
        raise LookupError(f"{label}设备 #{spec} 不存在或没有{label}通道")
    for d in candidates:
        if spec.lower() in d.name.lower():
            return d
    raise LookupError(f"找不到名称包含 {spec!r} 的{label}设备")


def probe(device: Device, kind: Kind) -> list[ProbeResult]:
    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    results = []
    for rate in PROBE_SAMPLERATES:
        for channels in PROBE_CHANNELS:
            try:
                check(device=device.index, channels=channels, samplerate=rate, dtype=PROBE_DTYPE)
            except sd.PortAudioError as e:
                code = e.args[1] if len(e.args) > 1 else None
                status = Status.UNSUPPORTED if code in _UNSUPPORTED_CODES else Status.FAILED
                results.append(ProbeResult(rate, channels, status, str(e.args[0]) if e.args else ""))
            except Exception as e:
                results.append(ProbeResult(rate, channels, Status.FAILED, f"{type(e).__name__}: {e}"))
            else:
                results.append(ProbeResult(rate, channels, Status.SUPPORTED))
    return results


def _display_name(d: Device) -> str:
    # 部分蓝牙驱动的设备名包含换行
    return " ".join(d.name.splitlines())


def _flags(d: Device) -> str:
    flags = []
    if d.is_default_input:
        flags.append("默认输入")
    if d.is_default_output:
        flags.append("默认输出")
    return ",".join(flags)


def format_device_list(devices: list[Device]) -> str:
    lines = []
    current = None
    for d in devices:
        if d.hostapi != current:
            current = d.hostapi
            lines.append(f"[{current}]")
        lines.append(
            f"  #{d.index:<3} in={d.max_input_channels:<2} out={d.max_output_channels:<2} "
            f"{d.default_samplerate:>7.0f} Hz  {_flags(d):<10} {_display_name(d)}"
        )
    return "\n".join(lines)


def format_probe_report(devices: list[Device]) -> str:
    lines = []
    counts = {s: 0 for s in Status}
    for d in devices:
        for kind, max_ch, label in (
            ("input", d.max_input_channels, "输入"),
            ("output", d.max_output_channels, "输出"),
        ):
            if max_ch == 0:
                continue
            lines.append(f"== #{d.index} {_display_name(d)} | {d.hostapi} | {label} | 默认 {d.default_samplerate:.0f} Hz")
            for r in probe(d, kind):
                counts[r.status] += 1
                ch = "单声道" if r.channels == 1 else "双声道"
                detail = f"  ({r.detail})" if r.detail else ""
                lines.append(f"   {r.samplerate:>5} Hz {ch}: {r.status.value}{detail}")
    lines.append("")
    lines.append("汇总: " + "  ".join(f"{s.value} {n}" for s, n in counts.items()))
    return "\n".join(lines)
