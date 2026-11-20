import os

import pytest
import sounddevice as sd

from app.audio import devices
from app.audio.devices import Status
from app.main import main

HOSTAPIS = [
    {"name": "MME", "default_input_device": 1, "default_output_device": 2},
    {"name": "Windows DirectSound", "default_input_device": 5, "default_output_device": -1},
    {"name": "Windows WASAPI", "default_input_device": 3, "default_output_device": 4},
]


def _dev(name, hostapi, inp, out, rate):
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": inp,
        "max_output_channels": out,
        "default_samplerate": rate,
    }


DEVICES = [
    _dev("Microsoft 声音映射器 - Input", 0, 2, 0, 44100.0),
    _dev("麦克风 (Realtek)", 0, 2, 0, 44100.0),
    _dev("扬声器 (Realtek)", 0, 0, 2, 44100.0),
    _dev("麦克风 (Realtek)", 2, 2, 0, 48000.0),
    _dev("扬声器 (Realtek)", 2, 0, 2, 48000.0),
    _dev("主声音捕获驱动程序", 1, 2, 0, 44100.0),
]


def fake_check(kind):
    def check(device, channels, samplerate, dtype):
        assert dtype == "int16"
        info = DEVICES[device]
        if channels > info[f"max_{kind}_channels"]:
            raise sd.PortAudioError("Invalid number of channels", -9998)
        if device == 5:
            raise RuntimeError("driver crashed")
        if info["hostapi"] == 2:  # WASAPI 共享模式：只接受混音格式采样率
            if samplerate == 44100:
                raise sd.PortAudioError("Unanticipated host error", -9999, (0, -1, "x"))
            if samplerate != 48000:
                raise sd.PortAudioError("Invalid sample rate", -9997)

    return check


@pytest.fixture(autouse=True)
def fake_sd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HOMEPILOT_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(devices.sd, "query_hostapis", lambda: HOSTAPIS)
    monkeypatch.setattr(devices.sd, "query_devices", lambda: DEVICES)
    monkeypatch.setattr(devices.sd, "check_input_settings", fake_check("input"))
    monkeypatch.setattr(devices.sd, "check_output_settings", fake_check("output"))


def by_index(index):
    return next(d for d in devices.list_devices() if d.index == index)


def test_list_devices_wasapi_first_and_keeps_others():
    result = devices.list_devices()
    assert [d.index for d in result] == [3, 4, 0, 1, 2, 5]
    assert [d.hostapi for d in result[:2]] == ["Windows WASAPI"] * 2


def test_list_devices_respects_preferred_hostapi():
    result = devices.list_devices("mme")
    assert [d.index for d in result] == [0, 1, 2, 3, 4, 5]


def test_list_devices_fields_and_defaults():
    mic = by_index(3)
    assert mic.name == "麦克风 (Realtek)"
    assert mic.max_input_channels == 2 and mic.max_output_channels == 0
    assert mic.default_samplerate == 48000.0
    assert mic.is_default_input and not mic.is_default_output
    assert by_index(4).is_default_output
    assert not by_index(0).is_default_input


def test_hostapi_rank_unknown_goes_last():
    assert devices.hostapi_rank("Windows WASAPI") == 0
    assert devices.hostapi_rank("ASIO") > devices.hostapi_rank("Windows WDM-KS")


def test_probe_covers_all_combinations():
    results = devices.probe(by_index(1), "input")
    assert [(r.samplerate, r.channels) for r in results] == [
        (16000, 1), (16000, 2), (44100, 1), (44100, 2), (48000, 1), (48000, 2)
    ]
    assert all(r.status is Status.SUPPORTED for r in results)


def test_probe_classifies_unsupported_and_failed():
    results = {(r.samplerate, r.channels): r for r in devices.probe(by_index(3), "input")}
    assert results[(16000, 1)].status is Status.UNSUPPORTED
    assert "Invalid sample rate" in results[(16000, 1)].detail
    assert results[(44100, 1)].status is Status.FAILED
    assert results[(48000, 1)].status is Status.SUPPORTED
    assert results[(48000, 2)].status is Status.SUPPORTED


def test_probe_uses_output_check_for_output():
    results = devices.probe(by_index(2), "output")
    assert all(r.status is Status.SUPPORTED for r in results)


def test_probe_channel_count_unsupported(monkeypatch):
    mono_mic = devices.Device(1, "mono", "MME", 1, 0, 44100.0, False, False)
    monkeypatch.setitem(DEVICES[1], "max_input_channels", 1)
    statuses = {r.channels: r.status for r in devices.probe(mono_mic, "input")}
    assert statuses == {1: Status.SUPPORTED, 2: Status.UNSUPPORTED}


def test_probe_unexpected_exception_is_failed():
    results = devices.probe(by_index(5), "input")
    mono = [r for r in results if r.channels == 1]
    assert all(r.status is Status.FAILED for r in mono)
    assert "RuntimeError" in mono[0].detail


def test_probe_report_skips_missing_direction():
    report = devices.format_probe_report(devices.list_devices())
    assert "#3 麦克风 (Realtek) | Windows WASAPI | 输入" in report
    assert "#3 麦克风 (Realtek) | Windows WASAPI | 输出" not in report
    assert "#4 扬声器 (Realtek) | Windows WASAPI | 输出" in report
    # 6 个设备各 1 个方向 x 6 种组合 = 36
    assert "汇总: 支持 " in report
    total = sum(int(x) for x in report.rsplit("汇总:", 1)[1].split()[1::2])
    assert total == 36


def test_device_name_newline_flattened(monkeypatch):
    monkeypatch.setitem(DEVICES[5], "name", "耳机 (Hands-Free%0\n;(HECATE))")
    out = devices.format_device_list(devices.list_devices())
    assert "耳机 (Hands-Free%0 ;(HECATE))" in out
    assert by_index(5).name == "耳机 (Hands-Free%0\n;(HECATE))"


def test_main_list_devices(capsys):
    assert main(["--list-devices"]) == 0
    out = capsys.readouterr().out
    assert out.index("[Windows WASAPI]") < out.index("[MME]")
    assert "默认输入" in out


def test_main_probe(capsys):
    assert main(["--probe"]) == 0
    out = capsys.readouterr().out
    assert "不支持" in out and "探测失败" in out and "支持" in out
