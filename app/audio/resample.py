from math import gcd

import numpy as np
from scipy.signal import firwin

# up/down 化简后的上限；超过时滤波器过长（如 48001 Hz 这类非标准采样率）
MAX_RATIO_TERM = 1000


class StreamResampler:
    """流式重采样单声道 PCM16。

    输入输出均为一维 np.int16。对任意分块方式，process() 输出拼接 + flush()
    的结果等同于对整段音频调用 scipy.signal.resample_poly(x, up, down)
    （相同的 Kaiser 窗 FIR、两端零填充），差异只来自 int16 取整。
    流结束后输出总点数为 ceil(输入总点数 * out_rate / in_rate)。
    """

    def __init__(self, in_rate: int, out_rate: int = 16000):
        for name, rate in (("in_rate", in_rate), ("out_rate", out_rate)):
            if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
                raise ValueError(f"{name} 必须是正整数，当前为 {rate!r}")
        g = gcd(in_rate, out_rate)
        up, down = out_rate // g, in_rate // g
        if max(up, down) > MAX_RATIO_TERM:
            raise ValueError(f"不支持的采样率转换 {in_rate} -> {out_rate}（比例 {up}/{down} 过于复杂）")

        self.in_rate = in_rate
        self.out_rate = out_rate
        self.up = up
        self.down = down
        self._passthrough = up == down
        self._n_in = 0
        self._n_out = 0
        self._flushed = False
        if self._passthrough:
            return

        # 与 resample_poly 相同的滤波器；delay 为上采样域中的群延迟
        max_rate = max(up, down)
        self._delay = 10 * max_rate
        h = firwin(2 * self._delay + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * up
        self._taps = -(-len(h) // up)
        h = np.concatenate([h, np.zeros(self._taps * up - len(h))])
        # _phases[p, j] = h[p + j * up]
        self._phases = h.reshape(self._taps, up).T.copy()
        self._tap_offsets = np.arange(self._taps)

        # _buf[i - _buf_start] 为第 i 个输入样本；负序号视为 0
        self._buf = np.zeros(self._taps - 1)
        self._buf_start = -(self._taps - 1)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """送入一块连续音频，返回当前已能确定的输出（可能为空）。"""
        if self._flushed:
            raise RuntimeError("重采样器已 flush，不能继续输入")
        x = _check_input(chunk)
        if self._passthrough:
            self._n_in += len(x)
            self._n_out += len(x)
            return x.copy()
        if len(x) == 0:
            return np.zeros(0, dtype=np.int16)

        self._buf = np.concatenate([self._buf, x.astype(np.float64)])
        self._n_in += len(x)
        # 输出 n 需要的最大输入序号为 (n*down + delay) // up，必须已到达
        limit = self._n_in * self.up - 1 - self._delay
        n_ready = limit // self.down + 1 if limit >= 0 else 0
        return self._emit(n_ready)

    def flush(self) -> np.ndarray:
        """流结束：把未来输入视为 0，输出剩余尾部。重复调用返回空数组。"""
        if self._flushed:
            return np.zeros(0, dtype=np.int16)
        self._flushed = True
        if self._passthrough:
            return np.zeros(0, dtype=np.int16)

        n_total = -(-self._n_in * self.up // self.down)
        if n_total > self._n_out:
            last_needed = ((n_total - 1) * self.down + self._delay) // self.up
            pad = last_needed - (self._buf_start + len(self._buf) - 1)
            if pad > 0:
                self._buf = np.concatenate([self._buf, np.zeros(pad)])
        return self._emit(n_total)

    def _emit(self, n_end: int) -> np.ndarray:
        if n_end <= self._n_out:
            return np.zeros(0, dtype=np.int16)

        n = np.arange(self._n_out, n_end)
        pos = n * self.down + self._delay
        newest = pos // self.up
        phase = pos - newest * self.up
        idx = newest[:, None] - self._tap_offsets[None, :] - self._buf_start
        y = np.einsum("nk,nk->n", self._phases[phase], self._buf[idx])
        self._n_out = n_end

        # 丢弃下一个输出不再需要的历史样本
        keep_from = (self._n_out * self.down + self._delay) // self.up - (self._taps - 1)
        drop = keep_from - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start = keep_from
        return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def _check_input(chunk: np.ndarray) -> np.ndarray:
    if not isinstance(chunk, np.ndarray) or chunk.dtype != np.int16 or chunk.ndim != 1:
        raise TypeError("输入必须是一维 np.int16 数组（单声道 PCM16）")
    return chunk
