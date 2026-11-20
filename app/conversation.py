"""对话状态机：按状态路由麦克风帧，串联唤醒词、VAD、云端通信和播放。"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from enum import Enum

from app.audio.player import AudioPlayer, make_tone
from app.audio.vad import SpeechAudio, SpeechStart
from app.client import Connected, Disconnected
from app.config import SAMPLE_RATE, Settings
from app.protocol import AudioFrame, ServerEvent

log = logging.getLogger(__name__)

MIC_TIMEOUT_S = 2.0
DIGITAL_SILENCE_S = 10.0  # 麦克风连续输出全零超过该时长视为设备失效（真实麦克风总有底噪）
PAD_SILENCE = bytes(2 * SAMPLE_RATE * 150 // 1000)  # 每次播放前后各垫 150ms 静音


class AudioDeviceError(RuntimeError):
    """音频设备失效，需要重新初始化音频子系统。"""


@dataclass
class Stats:
    """跨音频重启累计的运行统计。"""

    wakes: int = 0
    turns: int = 0
    false_wakes: int = 0  # 唤醒后一句话没说就因静默超时结束的会话
    disconnects: int = 0


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    FOLLOW_UP = "FOLLOW_UP"


class Speaker:
    """在独立任务中播放一段音频流，不阻塞状态机。

    start() 返回本次播放的标识；feed() 追加音频，finish() 表示没有更多音频。
    播放完成后调用 on_done(标识, None)，播放器出错时调用 on_done(标识, 异常)；cancel() 立即停止且不回调。
    """

    def __init__(self, settings: Settings, player_factory, on_done):
        self._settings = settings
        self._factory = player_factory
        self._on_done = on_done
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> object:
        self.cancel()
        token = object()
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run(token, self._queue), name="speaker")
        return token

    def feed(self, pcm: bytes) -> None:
        if self._queue is not None:
            self._queue.put_nowait(pcm)

    def finish(self) -> None:
        if self._queue is not None:
            self._queue.put_nowait(None)
            self._queue = None

    def cancel(self) -> None:
        self._queue = None
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self, token: object, queue: asyncio.Queue) -> None:
        player = self._factory(self._settings)
        try:
            player.start()
            await player.write(PAD_SILENCE)  # USB 等设备刚打开时会吞掉开头一小段声音
            while (pcm := await queue.get()) is not None:
                await player.write(pcm)
            await player.write(PAD_SILENCE)  # 停止输出流时设备缓冲中的结尾可能来不及播出
            await player.drain()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error = e
        else:
            error = None
        finally:
            player.stop()
        self._on_done(token, error)


class Conversation:
    """IDLE -> LISTENING -> THINKING -> SPEAKING -> FOLLOW_UP 的半双工对话控制。

    麦克风路由：
    - IDLE：只送唤醒词检测。
    - LISTENING / FOLLOW_UP：送 VAD（含预缓冲）；检测到开始说话后开始上传。
    - THINKING / SPEAKING：丢弃。
    所有状态变化都在 run() 的单个循环中按事件顺序发生。
    """

    def __init__(self, settings: Settings, capture, kws, vad, client, player_factory=AudioPlayer,
                 stats: Stats | None = None):
        self._settings = settings
        self._capture = capture
        self._kws = kws
        self._vad = vad
        self._client = client
        self._events: asyncio.Queue = asyncio.Queue()
        self._speaker = Speaker(settings, player_factory, lambda *done: self._events.put_nowait(("played", done)))
        self.stats = stats or Stats()
        self._beep = make_tone(0.2, freq=880.0, amplitude=0.25).tobytes()
        self.state = State.IDLE
        self._deadline: float | None = None
        self._uploading = False
        self._beep_token: object | None = None  # 提示音播放中，忽略麦克风
        self._reply_token: object | None = None
        self._tail = False  # 播放已结束，正在等待余音消散
        self._session_turns = 0
        self._notice_session = False  # 会话由提醒开启，不是唤醒，没人回应不算误唤醒
        self._receiving: dict[uuid.UUID, tuple[str, list[bytes]]] = {}  # notify_id -> (文字, 已收到的音频)
        self._notices: list[tuple[str, bytes]] = []  # 已收完、等待播放的提醒
        self._zero_frames = 0
        self._zero_limit = int(DIGITAL_SILENCE_S * 1000) // settings.frame_ms

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        pumps = [
            asyncio.create_task(self._pump("mic", lambda: asyncio.wait_for(self._capture.read(), MIC_TIMEOUT_S))),
            asyncio.create_task(self._pump("net", self._client.recv)),
        ]
        log.info("等待唤醒词")
        end_reason = "shutdown"
        try:
            while True:
                await self._play_notice_if_free()
                if self._deadline is not None and loop.time() >= self._deadline:
                    self._deadline = None
                    await self._on_timeout()
                    continue
                timeout = None if self._deadline is None else self._deadline - loop.time()
                try:
                    kind, item = await asyncio.wait_for(self._events.get(), timeout)
                except TimeoutError:
                    continue
                if kind == "mic":
                    await self._on_mic(item)
                elif kind == "net":
                    await self._on_net(item)
                elif kind == "played":
                    self._on_played(*item)
                else:
                    raise item
        except Exception:
            end_reason = "error"
            raise
        finally:
            for task in pumps:
                task.cancel()
            self._speaker.cancel()
            if self._client.session_id is not None and self._client.connected:
                await self._client.end_session(end_reason)

    async def _pump(self, kind: str, source) -> None:
        try:
            while True:
                self._events.put_nowait((kind, await source()))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._events.put_nowait(("error", e))

    # ---- 麦克风 ----

    async def _on_mic(self, frame: bytes) -> None:
        if frame.count(0) == len(frame):
            self._zero_frames += 1
            if self._zero_frames >= self._zero_limit:
                raise AudioDeviceError(f"麦克风连续 {DIGITAL_SILENCE_S:g} 秒输出全零（设备失效或被静音）")
        else:
            self._zero_frames = 0
        if self.state is State.IDLE:
            event = await self._kws.process(frame)
            if event:
                await self._on_wake(event)
        elif self.state in (State.LISTENING, State.FOLLOW_UP) and self._beep_token is None:
            for event in self._vad.process(frame):
                await self._on_vad(event)

    async def _on_wake(self, event) -> None:
        self._kws.reset()
        log.info("唤醒: %s（第 %d 帧）", event.keyword, event.frame_index)
        if not self._client.connected:
            log.warning("未连接服务端，忽略本次唤醒")
            return
        if await self._client.start_session() is None:
            log.warning("会话开始失败，忽略本次唤醒")
            return
        self.stats.wakes += 1
        self._session_turns = 0
        self._notice_session = False
        self._listen(State.LISTENING, f"唤醒，session={self._client.session_id}")
        if self._settings.wake_sound:
            self._beep_token = self._speaker.start()
            self._speaker.feed(self._beep)
            self._speaker.finish()

    async def _on_vad(self, event) -> None:
        if isinstance(event, SpeechStart):
            turn = await self._client.start_input()
            if turn is None:
                await self._go_idle("无法开始上传", end_reason=None)
                return
            self._uploading = True
            self._deadline = None
            self._enter(State.LISTENING, f"开始说话，turn={turn}，预缓冲 {len(event.frames)} 帧")
            for frame in event.frames:
                await self._client.send_audio(frame)
        elif isinstance(event, SpeechAudio):
            if self._uploading:
                await self._client.send_audio(event.frame)
        elif self._uploading:
            self._uploading = False
            await self._client.end_input(event.truncated)
            self.stats.turns += 1
            self._session_turns += 1
            self._reply_token = None
            self._deadline = self._loop.time() + self._settings.response_timeout_s
            self._enter(State.THINKING, f"说话结束，{event.duration_s:.2f}s{'（已截断）' if event.truncated else ''}")

    # ---- 服务端 ----

    async def _on_net(self, item) -> None:
        if isinstance(item, Connected):
            log.info("服务端已连接")
        elif isinstance(item, Disconnected):
            self.stats.disconnects += 1
            self._receiving.clear()  # 没收完的提醒作废，已收完的照常播放
            if self.state is not State.IDLE:
                await self._go_idle(f"连接断开: {item.reason}", end_reason=None)
        elif isinstance(item, AudioFrame):
            if item.turn_id in self._receiving:
                self._receiving[item.turn_id][1].append(item.pcm)
            elif self.state is State.SPEAKING:
                self._speaker.feed(item.pcm)
        elif isinstance(item, ServerEvent):
            await self._on_server_event(item)

    async def _on_server_event(self, event: ServerEvent) -> None:
        data, state = event.data, self.state
        if event.type == "transcript":
            log.info("识别%s: %s", "" if data["final"] else "（中间结果）", data["text"])
        elif event.type == "response.text":
            log.info("回复: %s", data["text"])
        elif event.type == "response.audio.start":
            if state is State.THINKING:
                self._reply_token = self._speaker.start()
                self._deadline = None
                self._enter(State.SPEAKING, "开始播放回复")
        elif event.type == "response.audio.end":
            if state is State.SPEAKING and self._reply_token is not None:
                self._deadline = self._loop.time() + self._settings.response_timeout_s  # 等待 response.done
        elif event.type == "response.done":
            if state is State.THINKING:
                self._listen(State.FOLLOW_UP, "回复结束（无语音）")
            elif state is State.SPEAKING:
                self._speaker.finish()
                self._deadline = None
        elif event.type == "notify.start":
            log.info("收到提醒: %s", data["text"])
            self._receiving[uuid.UUID(data["notify_id"])] = (data["text"], [])
        elif event.type == "notify.end":
            if received := self._receiving.pop(uuid.UUID(data["notify_id"]), None):
                text, chunks = received
                self._notices.append((text, b"".join(chunks)))
        elif event.type == "error":
            log.warning("服务端错误 %s: %s", data["code"], data["message"])
            if state in (State.THINKING, State.SPEAKING):
                self._speaker.cancel()
                await self._client.cancel_turn("error")
                self._listen(State.FOLLOW_UP, "服务端错误")

    async def _play_notice_if_free(self) -> None:
        """有待播提醒且当前不在一轮对话中（没在说话、等回复或播放回复）时，播放一条提醒。

        提醒按回复的方式播放（提示音 + 语音），播完进入 FOLLOW_UP，用户可以直接回应。
        """
        if not self._notices or self._uploading or self._beep_token is not None:
            return
        if self.state not in (State.IDLE, State.LISTENING, State.FOLLOW_UP):
            return
        text, pcm = self._notices.pop(0)
        if self._client.session_id is None:
            if await self._client.start_session() is None:
                log.warning("会话开始失败，提醒播完后无法直接回应")
            self._session_turns = 0
            self._notice_session = True
            self._kws.reset()
        self._reply_token = self._speaker.start()
        self._speaker.feed(self._beep)
        self._speaker.feed(pcm)
        self._speaker.finish()
        self._deadline = None
        self._tail = False
        self._enter(State.SPEAKING, f"播放提醒：{text}")

    # ---- 播放与超时 ----

    def _on_played(self, token: object, error: Exception | None) -> None:
        if error is not None and token in (self._beep_token, self._reply_token):
            raise AudioDeviceError(f"播放失败: {error}")
        if token is self._beep_token:
            self._beep_token = None
        elif token is self._reply_token and self.state is State.SPEAKING:
            self._reply_token = None
            self._tail = True
            self._deadline = self._loop.time() + self._settings.playback_tail_ms / 1000

    async def _on_timeout(self) -> None:
        state = self.state
        if state in (State.LISTENING, State.FOLLOW_UP) and not self._uploading:
            if self._session_turns == 0 and not self._notice_session:
                self.stats.false_wakes += 1
                log.info("唤醒后没有说话（疑似误唤醒）")
            await self._go_idle(f"{self._settings.silence_timeout_s:g} 秒没有说话", end_reason="silence_timeout")
        elif state is State.THINKING:
            await self._client.cancel_turn("timeout")
            self._listen(State.FOLLOW_UP, f"{self._settings.response_timeout_s:g} 秒内没有回复")
        elif state is State.SPEAKING:
            if self._tail:
                self._listen(State.FOLLOW_UP, "播放结束")
            else:
                log.warning("语音结束后未收到 response.done，按回复结束处理")
                self._speaker.finish()

    # ---- 状态切换 ----

    def _enter(self, state: State, reason: str) -> None:
        if state is self.state:
            log.info("%s：%s", state.value, reason)
        else:
            log.info("%s -> %s：%s", self.state.value, state.value, reason)
        self.state = state

    def _listen(self, state: State, reason: str) -> None:
        """进入 LISTENING（等待说话）或 FOLLOW_UP：清空 VAD 与预缓冲，开始静默计时。"""
        self._vad.reset()
        self._uploading = False
        self._tail = False
        self._reply_token = None
        self._deadline = self._loop.time() + self._settings.silence_timeout_s
        self._enter(state, reason)
        if state is State.FOLLOW_UP:
            log.info("可以直接继续说话；%g 秒内没开口将回到等待唤醒", self._settings.silence_timeout_s)

    async def _go_idle(self, reason: str, end_reason: str | None) -> None:
        self._speaker.cancel()
        self._beep_token = self._reply_token = None
        self._uploading = self._tail = False
        self._deadline = None
        if end_reason is not None:
            await self._client.end_session(end_reason)
        self._kws.reset()
        self._vad.reset()
        self._enter(State.IDLE, reason)
        log.info("等待唤醒词")
