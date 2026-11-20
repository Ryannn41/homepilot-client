from app.audio.assets import load_silence_farewell, load_wake_response


def test_wake_response_pcm():
    pcm = load_wake_response()
    assert len(pcm) % 2 == 0
    assert 1600 <= len(pcm) // 2 <= 32000  # 约 0.1–2 s @ 16 kHz


def test_silence_farewell_pcm():
    pcm = load_silence_farewell()
    assert len(pcm) % 2 == 0
    assert 1600 <= len(pcm) // 2 <= 64000  # 约 0.1–4 s @ 16 kHz