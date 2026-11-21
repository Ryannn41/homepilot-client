"""pywebview JS API：启停助手、状态与设置。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path

from pydantic import ValidationError

from app.assistant import AssistantSupervisor
from app.audio import devices
from app.config import Settings

log = logging.getLogger("homepilot.desktop")

UI_DIR = Path(__file__).resolve().parent / "ui"
DOTENV_PATH = Path.cwd() / ".env"
SETTINGS_KEYS = ("server_url", "device_id", "auth_token", "input_device", "output_device")


class Bridge:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._supervisor: AssistantSupervisor | None = None
        self._settings: Settings | None = None

    # ---- JS API ----

    def get_state(self) -> dict:
        if self._supervisor is None:
            return {
                "running": False,
                "connected": False,
                "conversation_state": "IDLE",
                "stats": {},
                "board": {"version": 0, "columns": {"upcoming": [], "todo": [], "done": []}},
            }
        return self._supervisor.get_status()

    def start_assistant(self) -> dict:
        self._ensure_thread()
        if self._supervisor is None or not self._supervisor.running:
            fut = asyncio.run_coroutine_threadsafe(self._supervisor.start(), self._loop)
            fut.result(timeout=30)
        return {"ok": True, **self.get_state()}

    def stop_assistant(self) -> dict:
        if self._supervisor is None or self._loop is None:
            return {"ok": True, **self.get_state()}
        fut = asyncio.run_coroutine_threadsafe(self._supervisor.stop(), self._loop)
        try:
            fut.result(timeout=30)
        except Exception as e:
            log.warning("停止助手: %s", e)
        return {"ok": True, **self.get_state()}

    def get_settings(self) -> dict:
        try:
            s = Settings()
        except ValidationError as e:
            return {"error": str(e)}
        token = s.auth_token.get_secret_value()
        return {
            "server_url": s.server_url,
            "device_id": s.device_id,
            "auth_token": token,
            "input_device": s.input_device,
            "output_device": s.output_device,
        }

    def save_settings(self, payload: dict) -> dict:
        lines: list[str] = []
        if DOTENV_PATH.is_file():
            lines = DOTENV_PATH.read_text(encoding="utf-8").splitlines()
        mapping = {
            "server_url": "HOMEPILOT_SERVER_URL",
            "device_id": "HOMEPILOT_DEVICE_ID",
            "auth_token": "HOMEPILOT_AUTH_TOKEN",
            "input_device": "HOMEPILOT_INPUT_DEVICE",
            "output_device": "HOMEPILOT_OUTPUT_DEVICE",
        }
        updates = {mapping[k]: str(payload.get(k, "")) for k in SETTINGS_KEYS if k in payload}
        out: list[str] = []
        seen = set()
        for line in lines:
            key = line.split("=", 1)[0].strip() if "=" in line else ""
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
            else:
                out.append(line)
        for key, value in updates.items():
            if key not in seen:
                out.append(f"{key}={value}")
        DOTENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
        for env_key, value in updates.items():
            os.environ[env_key] = value
        if self._supervisor is not None and self._supervisor.running:
            return {"ok": True, "note": "部分项需停止并重新启动助手后生效"}
        try:
            self._settings = Settings()
        except ValidationError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True}

    def list_audio_devices(self) -> str:
        try:
            s = Settings()
        except ValidationError:
            return "[]"
        items = [
            {
                "index": d.index,
                "name": d.name,
                "hostapi": d.hostapi,
                "inputs": d.max_input_channels,
                "outputs": d.max_output_channels,
            }
            for d in devices.list_devices(s.audio_hostapi)
        ]
        return json.dumps(items, ensure_ascii=False)

    def shutdown(self) -> None:
        self.stop_assistant()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._thread_main, name="assistant-loop", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=15):
            raise RuntimeError("助手线程启动超时")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            self._settings = Settings()
        except ValidationError as e:
            log.error("配置错误: %s", e)
            self._ready.set()
            return
        self._supervisor = AssistantSupervisor(self._settings)
        self._ready.set()
        loop.run_forever()
        loop.close()