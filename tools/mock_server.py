"""本地模拟服务端：按 docs/protocol.md 接收语音，回一段识别文本和回复文本，并把收到的音频原样作为 TTS 发回。
指定 --notify-every 时，还会定时向每个连接推送一条模拟提醒（notify.*），语音为一段提示音。

运行：python -m tools.mock_server [--port 8000] [--token xxx] [--realtime] [--notify-every 秒]
"""

import argparse
import asyncio
import contextlib
import json
import logging
import uuid
from http import HTTPStatus

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed

from app.audio.player import make_tone
from app.config import SAMPLE_RATE
from app.protocol import KIND_MIC, KIND_TTS, AudioFrame, ProtocolError, pack_audio, unpack_audio

log = logging.getLogger("mock_server")

NOTIFY_CHUNK = SAMPLE_RATE * 2 // 10  # 提醒音频每帧 100ms


class MockServer:
    def __init__(
        self, host: str = "127.0.0.1", port: int = 0, token: str = "", realtime: bool = False, notify_every: float = 0
    ):
        self.host, self.port, self.token, self.realtime = host, port, token, realtime
        self.notify_every = notify_every  # 大于 0 时每隔这么多秒推送一条提醒
        self.notified = 0
        self.events: list[dict] = []  # 收到的 JSON 事件
        self.frames: dict[str, list[AudioFrame]] = {}  # turn_id -> 收到的麦克风帧
        self.headers: list[Headers] = []  # 每个连接的请求头（键名不区分大小写）
        self.rejected = 0
        self._server: Server | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws/voice/"

    async def start(self) -> None:
        self._server = await serve(
            self._handler, self.host, self.port, process_request=self._check_auth, compression=None
        )
        self.port = self._server.sockets[0].getsockname()[1]
        log.info("模拟服务端已启动: %s", self.url)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> "MockServer":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    def _check_auth(self, connection: ServerConnection, request):
        if self.token and request.headers.get("Authorization") != f"Bearer {self.token}":
            self.rejected += 1
            log.warning("拒绝连接：认证失败")
            return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return None

    async def _handler(self, ws: ServerConnection) -> None:
        self.headers.append(ws.request.headers)
        log.info("客户端已连接: device=%s", ws.request.headers.get("X-Device-Id"))
        session = None
        cancelled: set[str] = set()
        tasks: set[asyncio.Task] = set()
        if self.notify_every > 0:
            tasks.add(asyncio.create_task(self._notify_periodically(ws)))
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    self._on_audio(message)
                    continue
                event = json.loads(message)
                self.events.append(event)
                type_, turn = event.get("type"), event.get("turn_id")
                log.info("收到事件 %s %s", type_, {k: v for k, v in event.items() if k not in ("type", "session_id")})
                if type_ == "session.start":
                    session = event["session_id"]
                elif type_ == "input.start":
                    self.frames[turn] = []
                elif type_ == "input.end":
                    task = asyncio.create_task(self._respond(ws, session, turn, self.frames.get(turn, []), cancelled))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                elif type_ == "turn.cancel":
                    cancelled.add(turn)
        except ConnectionClosed:
            pass
        finally:
            for task in tasks:
                task.cancel()
            log.info("客户端已断开")

    def _on_audio(self, data: bytes) -> None:
        try:
            frame = unpack_audio(data)
        except ProtocolError as e:
            log.warning("非法音频帧: %s", e)
            return
        frames = self.frames.setdefault(str(frame.turn_id), [])
        expected = frames[-1].seq + 1 if frames else 0
        if frame.kind != KIND_MIC or frame.seq != expected:
            log.warning("音频帧异常: kind=%d seq=%d，期望 seq=%d", frame.kind, frame.seq, expected)
        frames.append(frame)

    async def _respond(self, ws, session, turn, frames, cancelled) -> None:
        seconds = sum(len(f.pcm) for f in frames) / 2 / SAMPLE_RATE

        async def send(type_, **fields):
            await ws.send(json.dumps({"type": type_, "session_id": session, "turn_id": turn, **fields}, ensure_ascii=False))

        await send("transcript", text=f"（模拟识别）收到 {len(frames)} 帧，{seconds:.2f} 秒", final=True)
        await send("response.text", text="（模拟回复）下面原样播放你刚才说的话")
        await send("response.audio.start", sample_rate=SAMPLE_RATE)
        sent = 0
        for seq, frame in enumerate(frames):
            if turn in cancelled:
                break
            await ws.send(pack_audio(KIND_TTS, uuid.UUID(turn), seq, frame.pcm))
            sent += 1
            if self.realtime:
                await asyncio.sleep(len(frame.pcm) / 2 / SAMPLE_RATE)
        await send("response.audio.end", frames=sent)
        await send("response.done")
        log.info("已回复 turn %s：%d 帧", turn, sent)

    async def _notify_periodically(self, ws) -> None:
        with contextlib.suppress(ConnectionClosed):
            while True:
                await asyncio.sleep(self.notify_every)
                self.notified += 1
                await self.notify(ws, f"（模拟提醒）第 {self.notified} 条")

    async def notify(self, ws, text: str, pcm: bytes | None = None) -> None:
        """推送一条提醒；pcm 缺省为三声提示音。"""
        if pcm is None:
            beep, gap = make_tone(0.25, 660, 0.2).tobytes(), bytes(SAMPLE_RATE * 2 // 5)
            pcm = (beep + gap) * 3
        notify_id = uuid.uuid4()
        chunks = [pcm[i : i + NOTIFY_CHUNK] for i in range(0, len(pcm), NOTIFY_CHUNK)]
        await ws.send(json.dumps({"type": "notify.start", "notify_id": str(notify_id), "text": text,
                                  "sample_rate": SAMPLE_RATE}, ensure_ascii=False))
        for seq, chunk in enumerate(chunks):
            await ws.send(pack_audio(KIND_TTS, notify_id, seq, chunk))
        await ws.send(json.dumps({"type": "notify.end", "notify_id": str(notify_id), "frames": len(chunks)}))
        log.info("已推送提醒 %s：%s，%d 帧", notify_id, text, len(chunks))


async def _serve_forever(args) -> None:
    async with MockServer(args.host, args.port, args.token, args.realtime, args.notify_every):
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tools.mock_server", description="HomePilot 模拟服务端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default="", help="要求客户端携带的 Bearer 令牌，留空不校验")
    parser.add_argument("--realtime", action="store_true", help="按实时速度发送 TTS 音频（默认尽快发送）")
    parser.add_argument("--notify-every", type=float, default=0, metavar="S", help="客户端连上后每隔 S 秒推送一条模拟提醒，默认不推送")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_serve_forever(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
