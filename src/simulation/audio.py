"""Optional local WAV tracks for review. The mock makes audible test tones."""
from __future__ import annotations

import math
import struct
import wave

from .schemas import Session
from .store import ReviewStore


class MockTTS:
    def synthesize(self, text: str, lang: str):
        rate = 8000
        frequency = 440 if lang == "en" else 550
        return rate, b"".join(struct.pack("<h", round(2200 * math.sin(2 * math.pi * frequency * i / rate)))
                              for i in range(rate // 4))


class EdgeTTS:
    def __init__(self, voice_en=None, voice_es=None):
        from ..tts.common import EdgeTTSProvider

        self.provider = EdgeTTSProvider({"voice_en": voice_en, "voice_es": voice_es})

    def synthesize(self, text: str, lang: str):
        import numpy as np

        samples, rate = self.provider.synthesize(text, lang)
        return rate, (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def render_tracks(session: Session, store: ReviewStore, *, allow_network=False,
                  voice_en=None, voice_es=None) -> tuple[int, int]:
    if session.audio == "none":
        return 0, 0
    if session.audio == "edge" and not allow_network:
        raise ValueError("External audio requires --allow-network")
    renderer = MockTTS() if session.audio == "mock" else EdgeTTS(voice_en, voice_es)
    saved = failed = 0
    for turn in session.turns:
        tracks = [("source", turn.source, turn.source_language)]
        if turn.disposition == "accepted" and turn.interpretations and not turn.flags:
            candidate = " ".join(item.text for item in turn.interpretations)
            languages = {item.target_language for item in turn.interpretations}
            if languages == {turn.target_language}:
                tracks.append(("output", candidate, turn.target_language))
        for kind, text, lang in tracks:
            try:
                rate, data = renderer.synthesize(text, lang)
                if not data:
                    raise ValueError("Empty audio")
                path = store.audio_path(session.id, turn.id, kind)
                path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                    wav.writeframes(data)
                saved += 1
            except Exception:
                failed += 1
    return saved, failed
