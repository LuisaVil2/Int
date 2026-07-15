"""FishSpeechProvider reutiliza una sola requests.Session entre llamadas (nunca
recrea el cliente HTTP), y su benchmark_stats reporta una ventana móvil de latencia."""
import io
import wave

import numpy as np
import pytest

from src.tts.fish_provider import FishSpeechProvider


def _wav_bytes(seconds: float = 0.1, sr: int = 44100) -> bytes:
    n = int(seconds * sr)
    pcm = (np.zeros(n, dtype=np.int16)).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "test-key")
    monkeypatch.setenv("FISH_VOICE_ID", "test-voice")
    return FishSpeechProvider({})


def test_synthesize_reuses_single_session(provider, monkeypatch):
    calls = []

    def fake_post(self, url, json=None, headers=None, timeout=None):
        calls.append(self)
        return _FakeResponse(_wav_bytes())

    monkeypatch.setattr("requests.Session.post", fake_post, raising=True)

    provider.synthesize("hola", "es")
    provider.synthesize("adiós", "es")

    assert len(calls) == 2
    assert calls[0] is calls[1]  # mismo objeto Session en ambas llamadas
    assert calls[0] is provider._session


def test_benchmark_stats_rolling_window(provider, monkeypatch):
    monkeypatch.setattr("requests.Session.post",
                        lambda self, *a, **k: _FakeResponse(_wav_bytes()), raising=True)

    for _ in range(5):
        provider.synthesize("hola", "es")

    stats = provider.benchmark_stats()
    assert stats["sample_count"] == 5
    assert "mean_ms" in stats and "p50_ms" in stats and "p95_ms" in stats


def test_model_header_defaults_to_free_model(provider, monkeypatch):
    seen_headers = {}

    def fake_post(self, url, json=None, headers=None, timeout=None):
        seen_headers.update(headers)
        return _FakeResponse(_wav_bytes())

    monkeypatch.setattr("requests.Session.post", fake_post, raising=True)
    provider.synthesize("hola", "es")
    assert seen_headers["model"] == "s2.1-pro-free"
