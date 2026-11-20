import os

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.main import main


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    # 隔离开发者本机的 .env 和环境变量
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HOMEPILOT_"):
            monkeypatch.delenv(key)


def test_defaults():
    s = Settings()
    assert s.server_url == "ws://127.0.0.1:8000/ws/voice/"
    assert s.device_id == "homepilot-dev"
    assert s.auth_token.get_secret_value() == ""
    assert s.input_device == ""
    assert s.audio_hostapi == "wasapi"
    assert s.frame_ms == 20
    assert s.frame_samples == 320
    assert s.silence_timeout_s == 15.0
    assert s.log_level == "INFO"


def test_env_override(monkeypatch):
    monkeypatch.setenv("HOMEPILOT_FRAME_MS", "40")
    monkeypatch.setenv("HOMEPILOT_SILENCE_TIMEOUT_S", "8")
    s = Settings()
    assert s.frame_ms == 40
    assert s.frame_samples == 640
    assert s.silence_timeout_s == 8.0


def test_reads_dotenv(tmp_path):
    (tmp_path / ".env").write_text(
        "HOMEPILOT_DEVICE_ID=living-room\nHOMEPILOT_AUTH_TOKEN=secret\n", encoding="utf-8"
    )
    s = Settings()
    assert s.device_id == "living-room"
    assert s.auth_token.get_secret_value() == "secret"
    assert "secret" not in repr(s)


@pytest.mark.parametrize(
    "key,value",
    [
        ("HOMEPILOT_FRAME_MS", "30"),
        ("HOMEPILOT_SILENCE_TIMEOUT_S", "0"),
        ("HOMEPILOT_LOG_LEVEL", "VERBOSE"),
    ],
)
def test_invalid_values(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        Settings()


def test_main_starts(caplog):
    caplog.set_level("INFO")
    assert main([]) == 0
    assert "HomePilot Client" in caplog.text


def test_main_rejects_bad_config(monkeypatch, capsys):
    monkeypatch.setenv("HOMEPILOT_FRAME_MS", "30")
    assert main([]) == 2
    assert "配置错误" in capsys.readouterr().err
