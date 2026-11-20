import numpy as np
import pytest
from scipy.signal import resample_poly

from app.audio.resample import StreamResampler


def sine(rate, seconds=1.0, freq=440.0, amp=10000.0):
    t = np.arange(int(rate * seconds)) / rate
    return np.rint(amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def reference(x, in_rate, out_rate):
    r = StreamResampler(in_rate, out_rate)
    y = resample_poly(x.astype(np.float64), r.up, r.down)
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def run_chunks(r, x, sizes):
    out, pos, i = [], 0, 0
    while pos < len(x):
        size = sizes[i % len(sizes)]
        out.append(r.process(x[pos : pos + size]))
        pos += size
        i += 1
    out.append(r.flush())
    return np.concatenate(out)


def assert_close(y, ref):
    assert len(y) == len(ref)
    assert np.max(np.abs(y.astype(np.int32) - ref.astype(np.int32)), initial=0) <= 1


@pytest.mark.parametrize("sizes", [[1], [7], [160], [960], [1023], [48000], [960, 1, 333, 4096, 17]])
def test_48k_to_16k_matches_whole_signal(sizes):
    x = sine(48000)
    y = run_chunks(StreamResampler(48000, 16000), x, sizes)
    assert_close(y, reference(x, 48000, 16000))


def test_random_chunk_sizes():
    rng = np.random.default_rng(0)
    x = rng.integers(-20000, 20000, 48000 * 2, dtype=np.int16)
    sizes = list(rng.integers(1, 3000, 200))
    y = run_chunks(StreamResampler(48000, 16000), x, sizes)
    assert_close(y, reference(x, 48000, 16000))


@pytest.mark.parametrize("in_rate,out_rate", [(44100, 16000), (32000, 16000), (22050, 16000), (8000, 16000), (16000, 48000), (16000, 44100)])
def test_other_rates(in_rate, out_rate):
    x = sine(in_rate, 0.5)
    y = run_chunks(StreamResampler(in_rate, out_rate), x, [int(in_rate * 0.02), 101])
    assert_close(y, reference(x, in_rate, out_rate))


@pytest.mark.parametrize("n_in", [1, 2, 3, 5, 100, 48001])
def test_output_length_matches_ratio(n_in):
    x = sine(48000, 2.0)[:n_in]
    y = run_chunks(StreamResampler(48000, 16000), x, [960])
    assert len(y) == -(-n_in // 3)
    assert_close(y, reference(x, 48000, 16000))


def test_short_audio_needs_flush():
    r = StreamResampler(48000, 16000)
    x = sine(48000)[100:110]
    assert len(r.process(x)) == 0  # 不足以覆盖滤波器延迟
    tail = r.flush()
    assert_close(tail, reference(x, 48000, 16000))


def test_flush_returns_tail():
    r = StreamResampler(48000, 16000)
    x = sine(48000, 0.1)
    head = r.process(x)
    tail = r.flush()
    assert 0 < len(tail) < 20
    assert len(head) + len(tail) == len(x) // 3


def test_low_latency_per_frame():
    r = StreamResampler(48000, 16000)
    assert len(r.process(sine(48000)[:960])) >= 300


def test_empty_input_and_empty_stream():
    r = StreamResampler(48000, 16000)
    out = r.process(np.zeros(0, dtype=np.int16))
    assert out.dtype == np.int16 and len(out) == 0
    assert len(r.flush()) == 0


def test_repeated_flush_and_process_after_flush():
    r = StreamResampler(48000, 16000)
    r.process(sine(48000, 0.1))
    assert len(r.flush()) > 0
    assert len(r.flush()) == 0
    with pytest.raises(RuntimeError):
        r.process(sine(48000, 0.1))


@pytest.mark.parametrize("bad", [0, -16000, 48000.5, True, "48000"])
def test_invalid_rates(bad):
    with pytest.raises(ValueError):
        StreamResampler(bad, 16000)
    with pytest.raises(ValueError):
        StreamResampler(48000, bad)


def test_too_complex_ratio():
    with pytest.raises(ValueError):
        StreamResampler(48001, 16000)


@pytest.mark.parametrize("bad", [np.zeros(10, dtype=np.float32), np.zeros((10, 2), dtype=np.int16), [1, 2, 3], b"\x00\x00"])
def test_invalid_input(bad):
    with pytest.raises(TypeError):
        StreamResampler(48000, 16000).process(bad)


def test_passthrough():
    r = StreamResampler(16000, 16000)
    x = sine(16000, 0.1)
    out = r.process(x)
    assert np.array_equal(out, x) and out is not x
    assert len(r.flush()) == 0


def test_no_boundary_jumps():
    x = sine(48000, 1.0, freq=440.0, amp=10000.0)
    rng = np.random.default_rng(1)
    y = run_chunks(StreamResampler(48000, 16000), x, list(rng.integers(1, 2000, 100)))
    steady = y[50:-50].astype(np.int32)
    # 440 Hz 正弦在 16 kHz 下相邻样本的最大差值约 A * 2*pi*f/fs
    max_step = 10000 * 2 * np.pi * 440 / 16000
    assert np.max(np.abs(np.diff(steady))) <= max_step * 1.05 + 2


def test_output_clipped_to_int16():
    x = np.tile(np.r_[np.full(50, 32767), np.full(50, -32768)].astype(np.int16), 200)
    y = run_chunks(StreamResampler(48000, 16000), x, [960])
    assert y.dtype == np.int16
    assert_close(y, reference(x, 48000, 16000))
    assert y.max() == 32767 and y.min() == -32768
