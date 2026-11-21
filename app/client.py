"""与云端 homepilot-agent 的 WebSocket 通信：长连接、自动重连、会话与轮次标识、收包过滤。"""

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from app import __version__, protocol
from app.config import Settings
from app.protocol import KIND_MIC, KIND_TTS, AudioFrame, ProtocolError, ServerEvent

log = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 1 << 20


@dataclass(frozen=True)
class Connected:
    pass


@dataclass(frozen=True)
class Disconnected:
    reason: str
    aborted_turn: uuid.UUID | None  # 断线时未完成的轮次，已作废


Incoming = ServerEvent | AudioFrame | Connected | Disconnected


def next_delay(previous: float | None, settings: Settings) -> float:
    """不含抖动的下一次重连等待时间：从 reconnect_min_s 开始翻倍，不超过 reconnect_max_s。"""
    if previous is None:
        return settings.reconnect_min_s
    return min(previous * 2, settings.reconnect_max_s)


class VoiceClient:
    """保持到服务端的长连接并在断线后自动重连。

    - 会话：start_session() 生成 session_id；断线后会话作废，需要重新 start_session()。
    - 上行：start_input() 生成 turn_id，send_audio() 按顺序分配 seq，end_input() 结束本轮输入。
    - 下行：recv() 依次返回服务端事件、TTS 音频帧和连接状态变化。只放行当前会话、当前轮次的
      消息；TTS 帧 seq 不大于上一帧的视为重复丢弃，跳号记为缺口但照常放行。
      服务端主动推送的提醒（notify.*）、日程（schedule.*）及其音频帧不属于任何会话，始终放行。
    - 离线、会话或输入已因断线作废时，各发送方法返回 False/None，音频直接丢弃，不缓存。
    """

    def __init__(self, settings: Settings, queue_size: int = 3000):
        self._settings = settings
        self._queue: asyncio.Queue[Incoming] = asyncio.Queue(maxsize=queue_size)
        self._ws: ClientConnection | None = None
        self._task: asyncio.Task | None = None
        self._connected = asyncio.Event()
        self.session_id: uuid.UUID | None = None
        self.turn_id: uuid.UUID | None = None
        self._input_open = False
        self._seq = 0
        self._last_tts_seq = -1
        self._notify_ids: set[uuid.UUID] = set()  # 正在接收音频的提醒
        self.sent_frames = 0
        self.dropped_offline = 0
        self.dropped_stale = 0
        self.dropped_duplicate = 0
        self.dropped_invalid = 0
        self.dropped_overflow = 0
        self.seq_gaps = 0

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="voice-client")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def __aenter__(self) -> "VoiceClient":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def wait_connected(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
            return True
        except TimeoutError:
            return False

    async def recv(self) -> Incoming:
        return await self._queue.get()

    # ---- 上行 ----

    async def start_session(self) -> uuid.UUID | None:
        session_id = uuid.uuid4()
        if not await self._send(protocol.session_start(session_id, self._settings.device_id, __version__)):
            return None
        self.session_id = session_id
        self.turn_id = None
        self._input_open = False
        return session_id

    async def end_session(self, reason: str) -> bool:
        session_id, self.session_id = self.session_id, None
        self.turn_id = None
        self._input_open = False
        if session_id is None:
            return False
        return await self._send(protocol.session_end(session_id, reason))

    async def start_input(self) -> uuid.UUID | None:
        if self.session_id is None:
            return None
        turn_id = uuid.uuid4()
        # 先切换轮次，旧轮次迟到的响应从此刻起被丢弃
        self.turn_id = turn_id
        self._seq = 0
        self._last_tts_seq = -1
        self._input_open = False
        if not await self._send(protocol.input_start(self.session_id, turn_id)):
            return None
        self._input_open = True
        return turn_id

    async def send_audio(self, pcm: bytes) -> bool:
        if not self._input_open:
            self.dropped_offline += 1
            return False
        if not await self._send(protocol.pack_audio(KIND_MIC, self.turn_id, self._seq, pcm)):
            self.dropped_offline += 1
            return False
        self._seq += 1
        self.sent_frames += 1
        return True

    async def end_input(self, truncated: bool = False) -> bool:
        if not self._input_open:
            return False
        self._input_open = False
        return await self._send(protocol.input_end(self.session_id, self.turn_id, self._seq, truncated))

    async def cancel_turn(self, reason: str) -> bool:
        turn_id, self.turn_id = self.turn_id, None
        self._input_open = False
        if turn_id is None or self.session_id is None:
            return False
        return await self._send(protocol.turn_cancel(self.session_id, turn_id, reason))

    async def _send(self, message: str | bytes) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(message)
            return True
        except ConnectionClosed:
            return False

    # ---- 连接与下行 ----

    async def _run(self) -> None:
        s = self._settings
        headers = {"X-Device-Id": s.device_id}
        if token := s.auth_token.get_secret_value():
            headers["Authorization"] = f"Bearer {token}"
        delay = None
        while True:
            reason = ""
            try:
                async with connect(
                    s.server_url,
                    additional_headers=headers,
                    user_agent_header=f"homepilot-client/{__version__}",
                    compression=None,  # PCM 几乎不可压缩，压缩只增加延迟
                    proxy=None,
                    open_timeout=10,
                    ping_interval=s.ping_interval_s,
                    ping_timeout=s.ping_interval_s,
                    max_size=MAX_MESSAGE_BYTES,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    delay = None
                    log.info("已连接服务端: %s", s.server_url)
                    self._put(Connected())
                    try:
                        async for message in ws:
                            self._handle(message)
                    except ConnectionClosed:
                        pass
                    reason = _close_reason(ws)
            except InvalidStatus as e:
                reason = f"服务端拒绝连接: HTTP {e.response.status_code}"
            except (OSError, TimeoutError, WebSocketException) as e:
                reason = f"连接失败: {e or type(e).__name__}"
            finally:
                if self._ws is not None:
                    self._ws = None
                    self._connected.clear()
                    aborted = self.turn_id
                    self.session_id = self.turn_id = None
                    self._input_open = False
                    self._notify_ids.clear()
                    self._put(Disconnected(reason or "连接已关闭", aborted))
            delay = next_delay(delay, s)
            wait = delay * random.uniform(0.8, 1.2)
            log.warning("%s；%.1f 秒后重连", reason or "连接已断开", wait)
            await asyncio.sleep(wait)

    def _handle(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            self._handle_audio(message)
            return
        try:
            event = protocol.decode_event(message)
        except ProtocolError as e:
            self.dropped_invalid += 1
            log.warning("丢弃非法事件: %s", e)
            return
        if event.type.startswith("schedule."):
            self._put(event)
            return
        if event.type.startswith("notify."):  # 主动推送的提醒不属于任何会话，始终放行
            notify_id = uuid.UUID(event.data["notify_id"])
            if event.type == "notify.start":
                self._notify_ids.add(notify_id)
            else:
                self._notify_ids.discard(notify_id)
            self._put(event)
            return
        if event.session_id is not None and event.session_id != self.session_id:
            self._drop_stale(event.type, event.session_id, event.turn_id)
            return
        if event.turn_id is not None and event.turn_id != self.turn_id:
            self._drop_stale(event.type, event.session_id, event.turn_id)
            return
        if event.type == "response.audio.start":
            self._last_tts_seq = -1
        self._put(event)

    def _handle_audio(self, data: bytes) -> None:
        try:
            frame = protocol.unpack_audio(data)
        except ProtocolError as e:
            self.dropped_invalid += 1
            log.warning("丢弃非法音频帧: %s", e)
            return
        if frame.kind != KIND_TTS:
            self.dropped_invalid += 1
            log.warning("丢弃非 TTS 类型的下行音频帧: kind=%d", frame.kind)
            return
        if frame.turn_id in self._notify_ids:
            self._put(frame)
            return
        if frame.turn_id != self.turn_id:
            self._drop_stale("audio", None, frame.turn_id)
            return
        if frame.seq <= self._last_tts_seq:
            self.dropped_duplicate += 1
            log.debug("丢弃重复或乱序音频帧: seq=%d，上一帧 %d", frame.seq, self._last_tts_seq)
            return
        if frame.seq > self._last_tts_seq + 1:
            missing = frame.seq - self._last_tts_seq - 1
            self.seq_gaps += missing
            log.warning("TTS 音频缺少 %d 帧（seq %d 之前）", missing, frame.seq)
        self._last_tts_seq = frame.seq
        self._put(frame)

    def _drop_stale(self, kind: str, session_id, turn_id) -> None:
        self.dropped_stale += 1
        log.debug("丢弃过期消息 %s: session=%s turn=%s", kind, session_id, turn_id)

    def _put(self, item: Incoming) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped_overflow += 1
        self._queue.put_nowait(item)


def _close_reason(ws: ClientConnection) -> str:
    if ws.close_code is None or ws.close_code == 1006:
        return "连接异常断开"
    return f"连接关闭: {ws.close_code} {ws.close_reason}".strip()
