import asyncio
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app import main as main_module
from app.audio import wakeword
from app.audio.wakeword import WakeWordDetector, check_keywords, pcm16_to_float32
from app.config import Settings
from app.main import main

CHUNK = 1600  # 假模型每 100ms 解码一次
LOUD = np.full(320, 32000, dtype=np.int16)
QUIET = np.zeros(320, dtype=np.int16)


class FakeStream:
    def __init__(self):
        self.buf = np.zeros(0, dtype=np.float32)
        self.received: list[np.ndarray] = []
        self.last = np.zeros(0, dtype=np.float32)

    def accept_waveform(self, sample_rate, samples):
        assert sample_rate == 16000
        assert samples.dtype == np.float32
        self.received.append(samples.copy())
        self.buf = np.concatenate([self.buf, samples])


class FakeSpotter:
    instances: list["FakeSpotter"] = []
    fail = False
    decode_delay = 0.0

    def __init__(self, **kw):
        if FakeSpotter.fail:
            raise RuntimeError("onnx load error")
        self.kw = kw
        self.streams: list[FakeStream] = []
        self.resets = 0
        self.decode_threads: set[int] = set()
        FakeSpotter.instances.append(self)

    def create_stream(self):
        s = FakeStream()
        self.streams.append(s)
        return s

    def is_ready(self, s):
        return len(s.buf) >= CHUNK

    def decode_stream(self, s):
        self.decode_threads.add(threading.get_ident())
        if FakeSpotter.decode_delay:
            time.sleep(FakeSpotter.decode_delay)
        s.last, s.buf = s.buf[:CHUNK], s.buf[CHUNK:]

    def get_result(self, s):
        return "你好小余" if len(s.last) and np.max(np.abs(s.last)) > 0.9 else ""

    def reset_stream(self, s):
        self.resets += 1
        s.buf = np.zeros(0, dtype=np.float32)
        s.last = np.zeros(0, dtype=np.float32)


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HOMEPILOT_"):
            monkeypatch.delenv(key)
    d = tmp_path / "kws"
    d.mkdir()
    for name in ("enc.onnx", "dec.onnx", "join.onnx"):
        (d / name).write_bytes(b"fake")
    (d / "tokens.txt").write_text(
        "\n".join(f"{t} {i}" for i, t in enumerate(["<blk>", "n", "ǐ", "h", "ǎo", "x", "iǎo", "y", "ú"])),
        encoding="utf-8",
    )
    (tmp_path / "keywords.txt").write_text("n ǐ h ǎo x iǎo y ú @你好小余\n", encoding="utf-8")
    FakeSpotter.instances = []
    FakeSpotter.fail = False
    FakeSpotter.decode_delay = 0.0
    monkeypatch.setattr(wakeword.sherpa_onnx, "KeywordSpotter", FakeSpotter)
    return d


def make_settings(model_dir, **kw):
    return Settings(
        kws_model_dir=str(model_dir),
        kws_encoder="enc.onnx",
        kws_decoder="dec.onnx",
        kws_joiner="join.onnx",
        kws_keywords_file="keywords.txt",
        **kw,
    )


# ---- 格式转换与关键词校验 ----

def test_pcm16_to_float32():
    x = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    for data in (x, x.tobytes()):
        y = pcm16_to_float32(data)
        assert y.dtype == np.float32
        assert y.tolist() == pytest.approx([0.0, 0.5, -0.5, 32767 / 32768, -1.0])
    assert len(pcm16_to_float32(b"")) == 0


@pytest.mark.parametrize("bad,exc", [
    (b"\x00", ValueError),
    (np.zeros(320, np.float32), TypeError),
    (np.zeros((320, 2), np.int16), TypeError),
    ([0, 1], TypeError),
])
def test_pcm16_to_float32_rejects(bad, exc):
    with pytest.raises(exc):
        pcm16_to_float32(bad)


def test_check_keywords(model_dir, tmp_path):
    tokens = model_dir / "tokens.txt"
    kw = tmp_path / "k.txt"
    kw.write_text("n ǐ h ǎo x iǎo y ú :1.5 #0.3 @你好小余\n\nx iǎo y ú\n", encoding="utf-8")
    assert check_keywords(kw, tokens) == ["你好小余", "x iǎo y ú"]
    kw.write_text("n ǐ h ǎo x iǎo y ǔ @你好小雨\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ǔ"):
        check_keywords(kw, tokens)
    kw.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        check_keywords(kw, tokens)


def test_repo_keywords_file_is_valid():
    repo = Path(__file__).resolve().parent.parent
    model = repo / Settings.model_fields["kws_model_dir"].default
    if not (model / "tokens.txt").is_file():
        pytest.skip("本机未下载 KWS 模型")
    assert check_keywords(repo / "keywords.txt", model / "tokens.txt") == ["你好小余"]


# ---- 加载 ----

def test_loads_with_config(model_dir):
    det = WakeWordDetector(make_settings(model_dir, kws_keywords_threshold=0.4, kws_keywords_score=2.0, kws_num_threads=2))
    kw = FakeSpotter.instances[0].kw
    assert kw["encoder"].endswith("enc.onnx") and kw["tokens"].endswith("tokens.txt")
    assert kw["keywords_file"].endswith("keywords.txt")
    assert (kw["keywords_threshold"], kw["keywords_score"], kw["num_threads"]) == (0.4, 2.0, 2)
    assert det.keywords == ["你好小余"]
    det.close()


def test_missing_files_reported(model_dir):
    (model_dir / "dec.onnx").unlink()
    with pytest.raises(FileNotFoundError, match="decoder"):
        WakeWordDetector(make_settings(model_dir))
    assert not FakeSpotter.instances


def test_model_load_failure(model_dir):
    FakeSpotter.fail = True
    with pytest.raises(RuntimeError, match="KWS 模型加载失败"):
        WakeWordDetector(make_settings(model_dir))


def test_invalid_keywords_rejected_before_loading(model_dir, tmp_path):
    (tmp_path / "keywords.txt").write_text("q ü @未知\n", encoding="utf-8")
    with pytest.raises(ValueError):
        WakeWordDetector(make_settings(model_dir))
    assert not FakeSpotter.instances


# ---- 连续检测 ----

def test_continuous_frames_keep_stream_state(model_dir):
    with WakeWordDetector(make_settings(model_dir)) as det:
        rng = np.random.default_rng(0)
        frames = [rng.integers(-1000, 1000, 320, dtype=np.int16) for _ in range(12)]
        for f in frames:
            assert det.accept(f) is None
        stream = FakeSpotter.instances[0].streams[0]
        assert len(FakeSpotter.instances[0].streams) == 1  # 同一个流
        got = np.concatenate(stream.received)
        assert np.array_equal(got, np.concatenate(frames).astype(np.float32) / 32768.0)
        assert len(stream.buf) == 12 * 320 - 2 * CHUNK  # 已解码 2 块，余下等待


def test_detects_keyword_with_frame_index(model_dir):
    with WakeWordDetector(make_settings(model_dir)) as det:
        for _ in range(10):
            assert det.accept(QUIET) is None
        events = [det.accept(LOUD.tobytes()) for _ in range(5)]
        hits = [e for e in events if e]
        assert len(hits) == 1
        e = hits[0]
        assert e.keyword == "你好小余"
        assert e.frame_index == 14  # 第 15 帧凑满第 3 个解码块
        assert e.stream_time_s == pytest.approx(15 * 0.02)
        assert e.detected_at <= time.monotonic()
        assert FakeSpotter.instances[0].resets == 1


def test_no_duplicate_trigger_after_detection(model_dir):
    with WakeWordDetector(make_settings(model_dir)) as det:
        results = [det.accept(LOUD) for _ in range(5)] + [det.accept(QUIET) for _ in range(20)]
        assert sum(1 for r in results if r) == 1


def test_reset_allows_detecting_again(model_dir):
    with WakeWordDetector(make_settings(model_dir)) as det:
        det.accept(QUIET)
        det.accept(QUIET)
        first = next(e for e in (det.accept(LOUD) for _ in range(5)) if e)
        det.reset()
        assert len(FakeSpotter.instances[0].streams) == 2
        second = next(e for e in (det.accept(LOUD) for _ in range(5)) if e)
        assert second.frame_index > first.frame_index
        assert second.stream_time_s == pytest.approx(5 * 0.02)


def test_invalid_frame_does_not_change_state(model_dir):
    with WakeWordDetector(make_settings(model_dir)) as det:
        det.accept(QUIET)
        with pytest.raises(ValueError):
            det.accept(b"\x00\x00\x00")
        with pytest.raises(TypeError):
            det.accept(np.zeros(320, np.float32))
        assert sum(len(r) for r in FakeSpotter.instances[0].streams[0].received) == 320
        hit = [det.accept(LOUD) for _ in range(5)]
        assert next(e for e in hit if e).frame_index == 4  # 非法帧未计入：320 + 4*320 = 1600


def test_empty_frame_is_ignored(model_dir):
    with WakeWordDetector(make_settings(model_dir)) as det:
        assert det.accept(b"") is None
        assert not FakeSpotter.instances[0].streams[0].received


# ---- 异步与资源 ----

def test_process_runs_off_event_loop(model_dir):
    FakeSpotter.decode_delay = 0.05

    async def run():
        with WakeWordDetector(make_settings(model_dir)) as det:
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.005)
                    ticks += 1

            t = asyncio.create_task(ticker())
            events = [await det.process(LOUD) for _ in range(10)]  # 2 次解码，各 50ms
            t.cancel()
            assert sum(1 for e in events if e) == 2  # 每次触发后重置，下一块仍是唤醒音频
            assert threading.get_ident() not in FakeSpotter.instances[0].decode_threads
            assert ticks >= 10  # 解码期间事件循环仍在运行

    asyncio.run(run())


def test_process_keeps_frame_order(model_dir):
    async def run():
        with WakeWordDetector(make_settings(model_dir)) as det:
            frames = [np.full(320, i, dtype=np.int16) for i in range(20)]
            await asyncio.gather(*(det.process(f) for f in frames))
            got = np.concatenate(FakeSpotter.instances[0].streams[0].received)
            assert np.array_equal(got, np.concatenate(frames).astype(np.float32) / 32768.0)

    asyncio.run(run())


def test_close_is_idempotent_and_blocks_further_use(model_dir):
    det = WakeWordDetector(make_settings(model_dir))
    det.close()
    det.close()
    with pytest.raises(RuntimeError):
        det.accept(QUIET)

    async def run():
        with pytest.raises(RuntimeError):
            await det.process(QUIET)

    asyncio.run(run())


# ---- CLI ----

class FakeCapture:
    def __init__(self, settings):
        self.device = type("D", (), {"index": 21, "name": "HyperX", "hostapi": "Windows WASAPI"})()
        self.samplerate, self.channels, self.dropped_frames = 48000, 1, 0
        self._frames = [QUIET] * 20 + [LOUD] * 5 + [QUIET] * 125

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def read(self):
        return self._frames.pop(0).tobytes()


def test_kws_test_cli_prints_detection(model_dir, monkeypatch, capsys):
    monkeypatch.setattr(main_module, "AudioCapture", FakeCapture)
    for key, value in {
        "HOMEPILOT_KWS_MODEL_DIR": str(model_dir), "HOMEPILOT_KWS_ENCODER": "enc.onnx",
        "HOMEPILOT_KWS_DECODER": "dec.onnx", "HOMEPILOT_KWS_JOINER": "join.onnx",
    }.items():
        monkeypatch.setenv(key, value)
    assert main(["--kws-test", "3"]) == 0
    out = capsys.readouterr().out
    assert ">>> 唤醒: 你好小余" in out
    assert "唤醒 1 次" in out and "共 150 帧" in out


def test_kws_test_cli_missing_model(model_dir, capsys):
    assert main(["--kws-test", "1"]) == 1
    assert "KWS 文件不存在" in capsys.readouterr().err
