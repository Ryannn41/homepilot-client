import math
import os
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from app import main as main_module
from app.audio import vad as vad_module
from app.audio.vad import WINDOW_SIZE, SpeechAudio, SpeechEnd, SpeechStart, VADDetector
from app.config import Settings
from app.main import main

FRAME = 320


def frame(tag: int, loud: bool) -> bytes:
    """每帧样本值唯一标识其序号：静音帧 = 序号，语音帧 = 10000 + 序号。"""
    return np.full(FRAME, 10000 + tag if loud else tag, dtype=np.int16).tobytes()


def tag_of(f: bytes) -> int:
    v = int(np.frombuffer(f, dtype=np.int16)[0])
    return v - 10000 if v >= 10000 else v


class FakeVad:
    """模拟 sherpa-onnx VoiceActivityDetector：连续语音窗达到最短语音时长判定开始，连续静音窗达到最短静音时长判定结束。"""

    instances: list["FakeVad"] = []
    fail_load = False
    fail_accept = False

    def __init__(self, config, buffer_size_in_seconds):
        if FakeVad.fail_load:
            raise RuntimeError("onnx load error")
        self.config = config
        self.buffer_size = buffer_size_in_seconds
        sil = config.silero_vad
        self.start_n = math.ceil(sil.min_speech_duration * 16000 / sil.window_size)
        self.end_n = math.ceil(sil.min_silence_duration * 16000 / sil.window_size)
        self.windows: list[np.ndarray] = []
        self.segments: list[object] = []
        self.resets = 0
        self._clear()
        FakeVad.instances.append(self)

    def _clear(self):
        self.speech = False
        self.voiced_run = self.silent_run = 0

    def accept_waveform(self, samples):
        if FakeVad.fail_accept:
            raise RuntimeError("onnx run error")
        assert samples.dtype == np.float32 and len(samples) == WINDOW_SIZE
        self.windows.append(samples.copy())
        voiced = float(np.max(np.abs(samples))) > 0.1
        self.voiced_run = self.voiced_run + 1 if voiced else 0
        self.silent_run = 0 if voiced else self.silent_run + 1
        if not self.speech and self.voiced_run >= self.start_n:
            self.speech = True
        elif self.speech and self.silent_run >= self.end_n:
            self.speech = False
            self.segments.append(object())

    def is_speech_detected(self):
        return self.speech

    def empty(self):
        return not self.segments

    def pop(self):
        self.segments.pop(0)

    def reset(self):
        self.resets += 1
        self._clear()


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HOMEPILOT_"):
            monkeypatch.delenv(key)
    path = tmp_path / "vad.onnx"
    path.write_bytes(b"fake")
    FakeVad.instances = []
    FakeVad.fail_load = FakeVad.fail_accept = False
    monkeypatch.setattr(vad_module.sherpa_onnx, "SileroVadModelConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(vad_module.sherpa_onnx, "VadModelConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(vad_module.sherpa_onnx, "VoiceActivityDetector", FakeVad)
    return path


def make(model, **kw):
    # 最短语音 100ms = 4 个窗(约 6.4 帧)，结束静音 200ms = 7 个窗(约 11.2 帧)，预缓冲 10 帧
    params = dict(vad_model=str(model), vad_min_speech_ms=100, vad_min_silence_ms=200, preroll_ms=200)
    params.update(kw)
    return VADDetector(Settings(**params))


def feed(det, pattern: str, start: int = 0) -> list:
    """pattern 中 'L' 为语音帧、'Q' 为静音帧；返回全部事件。"""
    events = []
    for i, c in enumerate(pattern):
        events += det.process(frame(start + i, c == "L"))
    return events


def of(events, cls):
    return [e for e in events if isinstance(e, cls)]


# ---- 加载与配置 ----

def test_config_passed_to_sherpa(model):
    det = make(model, vad_threshold=0.6, vad_max_speech_s=12)
    fake = FakeVad.instances[0]
    sil = fake.config.silero_vad
    assert sil.model == str(model) and sil.window_size == WINDOW_SIZE
    assert (sil.threshold, sil.min_speech_duration, sil.min_silence_duration, sil.max_speech_duration) == (0.6, 0.1, 0.2, 12)
    assert fake.config.sample_rate == 16000
    assert fake.buffer_size > 12
    det.close()


def test_default_preroll_is_25_frames():
    assert Settings().preroll_frames == 25
    assert Settings(frame_ms=40).preroll_frames == 12


@pytest.mark.parametrize("kw", [{"preroll_ms": 100}, {"preroll_ms": 1500}, {"vad_threshold": 1.0}, {"vad_min_silence_ms": 50}])
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        Settings(**kw)


# ---- 开始与结束 ----

def test_speech_start_includes_preroll(model):
    with make(model) as det:
        events = feed(det, "Q" * 30 + "L" * 20)
        [start] = of(events, SpeechStart)
        assert start.kind == "speech_start"
        tags = [tag_of(f) for f in start.frames]
        assert tags == list(range(start.start_frame, start.detected_frame + 1))
        assert len(tags) == 10 and start.duration_s == pytest.approx(0.2)
        assert 30 in tags and tags[0] < 30  # 语音第一帧及其前的静音都在预缓冲中
        assert start.detected_frame > 30  # VAD 存在检测延迟
        assert det.in_speech


def test_speech_end_outputs_complete_segment(model):
    with make(model) as det:
        events = feed(det, "Q" * 30 + "L" * 20 + "Q" * 30)
        [start], [end] = of(events, SpeechStart), of(events, SpeechEnd)
        assert end.kind == "speech_end" and not end.truncated
        tags = [tag_of(f) for f in end.frames]
        assert tags == list(range(start.start_frame, end.end_frame + 1))
        assert tags[0] < 30 and 49 in tags  # 首尾语音帧都完整
        assert end.duration_s == pytest.approx(len(end.frames) * 0.02)
        assert end.pcm == b"".join(end.frames)
        assert not det.in_speech


def test_short_pause_does_not_end_speech(model):
    with make(model) as det:
        events = feed(det, "Q" * 20 + "L" * 20 + "Q" * 8 + "L" * 20 + "Q" * 30)
        [end] = of(events, SpeechEnd)
        tags = [tag_of(f) for f in end.frames]
        assert 39 in tags and 48 in tags and 67 in tags


def test_ends_only_after_silence_threshold(model):
    with make(model) as det:
        events = feed(det, "Q" * 20 + "L" * 20)
        last_loud = 39
        n = 40
        while not of(events, SpeechEnd):
            events = det.process(frame(n, False))
            n += 1
            assert n < 100
        [end] = of(events, SpeechEnd)
        assert (end.end_frame - last_loud) * 20 >= 200  # 静音持续 200ms 以上才结束


# ---- 预缓冲 ----

def test_preroll_is_bounded(model):
    with make(model) as det:
        events = feed(det, "Q" * 500 + "L" * 15)
        [start] = of(events, SpeechStart)
        assert len(start.frames) == 10
        assert len(det._preroll) == 0


def test_frames_delivered_exactly_once(model):
    with make(model) as det:
        events = feed(det, "Q" * 30 + "L" * 40 + "Q" * 30)
        [start], [end] = of(events, SpeechStart), of(events, SpeechEnd)
        streamed = list(start.frames) + [e.frame for e in of(events, SpeechAudio)]
        assert streamed == list(end.frames)
        tags = [tag_of(f) for f in streamed]
        assert len(tags) == len(set(tags))
        audio_idx = [e.frame_index for e in of(events, SpeechAudio)]
        assert audio_idx[0] == start.detected_frame + 1


def test_consecutive_segments_do_not_mix(model):
    with make(model) as det:
        events = feed(det, "Q" * 30 + "L" * 20 + "Q" * 30 + "L" * 20 + "Q" * 30)
        starts, ends = of(events, SpeechStart), of(events, SpeechEnd)
        assert len(starts) == len(ends) == 2
        assert min(tag_of(f) for f in starts[1].frames) > ends[0].end_frame
        assert min(tag_of(f) for f in ends[1].frames) > ends[0].end_frame


def test_window_splicing_and_segment_queue_drained(model):
    with make(model) as det:
        feed(det, "Q" * 10 + "L" * 30 + "Q" * 30)
        fake = FakeVad.instances[0]
        got = np.concatenate(fake.windows)
        expected = np.concatenate([np.frombuffer(frame(i, 10 <= i < 40), dtype=np.int16) for i in range(70)])
        n = len(got)
        assert n == (70 * FRAME // WINDOW_SIZE) * WINDOW_SIZE
        assert np.array_equal(got, expected[:n].astype(np.float32) / 32768.0)
        assert fake.empty()


# ---- 状态管理 ----

def test_reset_clears_state(model):
    with make(model) as det:
        feed(det, "Q" * 30 + "L" * 20)
        assert det.in_speech
        det.reset()
        assert not det.in_speech and FakeVad.instances[0].resets == 1
        assert len(det._preroll) == 0 and len(det._pending) == 0
        events = feed(det, "Q" * 5 + "L" * 20, start=50)
        [start] = of(events, SpeechStart)
        assert min(tag_of(f) for f in start.frames) >= 50
        assert start.detected_frame >= 50  # 帧序号继续递增


def test_close_blocks_further_use(model):
    det = make(model)
    det.close()
    det.close()
    det.reset()
    with pytest.raises(RuntimeError, match="已关闭"):
        det.process(frame(0, False))
    with make(model) as det2:
        pass
    with pytest.raises(RuntimeError):
        det2.process(frame(0, False))


@pytest.mark.parametrize("bad", [b"", b"\x00" * 639, b"\x00" * 1280, "x" * 640, np.zeros(320, np.int16), None])
def test_invalid_frame_rejected_without_side_effects(model, bad):
    with make(model) as det:
        det.process(frame(0, False))
        with pytest.raises(ValueError):
            det.process(bad)
        assert len(det._preroll) == 1 and det._frame_index == 0
        assert sum(len(w) for w in FakeVad.instances[0].windows) == 0  # 320 样本尚不足一个窗


def test_frame_size_follows_frame_ms(model):
    with make(model, frame_ms=40) as det:
        with pytest.raises(ValueError):
            det.process(frame(0, False))
        assert det.process(np.zeros(640, np.int16).tobytes()) == []


# ---- 时长限制 ----

def test_short_burst_below_min_speech_ignored(model):
    with make(model) as det:
        events = feed(det, "Q" * 20 + "L" * 3 + "Q" * 30 + "L" * 4 + "Q" * 30)
        assert events == []


def test_max_speech_truncates_and_waits_for_silence(model):
    with make(model, vad_max_speech_s=0.5) as det:
        events = feed(det, "Q" * 20 + "L" * 80)
        [start], [end] = of(events, SpeechStart), of(events, SpeechEnd)
        assert end.truncated and len(end.frames) == 25
        assert [tag_of(f) for f in end.frames] == list(range(start.start_frame, end.end_frame + 1))
        assert not det.in_speech
        # 截断后用户仍在说话：不开启新片段，也不把这些帧放进预缓冲
        events = feed(det, "Q" * 30 + "L" * 20, start=100)
        [start2] = of(events, SpeechStart)
        assert min(tag_of(f) for f in start2.frames) > 100


# ---- 模型异常 ----

def test_missing_model(model):
    model.unlink()
    with pytest.raises(FileNotFoundError, match="VAD 模型不存在"):
        make(model)
    assert not FakeVad.instances


def test_model_load_failure(model):
    FakeVad.fail_load = True
    with pytest.raises(RuntimeError, match="VAD 模型加载失败"):
        make(model)


def test_inference_failure(model):
    with make(model) as det:
        det.process(frame(0, False))
        FakeVad.fail_accept = True
        with pytest.raises(RuntimeError, match="VAD 推理失败"):
            det.process(frame(1, False))


# ---- CLI ----

class FakeCapture:
    def __init__(self, settings):
        self.device = SimpleNamespace(index=21, name="HyperX", hostapi="Windows WASAPI")
        self.samplerate, self.channels, self.dropped_frames = 48000, 1, 0
        pattern = "Q" * 30 + "L" * 30 + "Q" * 40 + "L" * 20 + "Q" * 30
        self._frames = [frame(i, c == "L") for i, c in enumerate(pattern)]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def read(self):
        return self._frames.pop(0)


def test_vad_test_cli_prints_and_saves_segments(model, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(main_module, "AudioCapture", FakeCapture)
    for key, value in {"HOMEPILOT_VAD_MODEL": str(model), "HOMEPILOT_VAD_MIN_SPEECH_MS": "100",
                       "HOMEPILOT_VAD_MIN_SILENCE_MS": "200", "HOMEPILOT_PREROLL_MS": "200"}.items():
        monkeypatch.setenv(key, value)
    assert main(["--vad-test", "3", "--vad-save", "segs"]) == 0
    out = capsys.readouterr().out
    assert out.count(">>> 语音开始") == 2 and out.count("<<< 语音结束") == 2
    assert "语音片段 2 个" in out and "共 150 帧" in out
    files = sorted((tmp_path / "segs").glob("*.wav"))
    assert [f.name for f in files] == ["segment_001.wav", "segment_002.wav"]
    with wave.open(str(files[0])) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        first = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert first[0] < 10000 and 10000 + 30 in first  # 片段以静音预缓冲开头，包含第一帧语音


def test_vad_test_cli_missing_model(model, capsys):
    model.unlink()
    assert main(["--vad-test", "1"]) == 1
    assert "VAD 模型不存在" in capsys.readouterr().err
