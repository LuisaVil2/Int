from __future__ import annotations

import asyncio
import io
import os
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class TTSConfigError(RuntimeError):
    """Raised when the TTS configuration is invalid."""


class BaseTTSProvider:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError


class EdgeTTSProvider(BaseTTSProvider):
    VOICES = {
        "es": os.getenv("TTS_VOICE_ES", "es-MX-DaliaNeural"),
        "en": os.getenv("TTS_VOICE_EN", "en-US-AriaNeural"),
    }

    def _ffmpeg(self) -> str:
        return os.getenv("FFMPEG_PATH", "ffmpeg")

    async def _mp3_bytes(self, text: str, voice: str) -> bytes:
        import edge_tts

        buf = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        voice = self.VOICES.get(lang, self.VOICES["en"])
        mp3 = asyncio.run(self._mp3_bytes(text, voice))
        if not mp3:
            return np.zeros(0, dtype=np.float32), 16000
        proc = subprocess.run(
            [
                self._ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=mp3,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not proc.stdout:
            return np.zeros(0, dtype=np.float32), 16000
        with wave.open(io.BytesIO(proc.stdout), "rb") as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr


def load_tts_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = Path(config_path or os.getenv("TTS_CONFIG_PATH") or root / "config.yaml")
    if not path.is_absolute():
        path = (root / path).resolve()

    default_cfg = {
        "provider": "fish",
        "reference_voice": "voices/luisa.wav",
        "language_auto": True,
    }
    if not path.exists():
        return default_cfg

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TTSConfigError("TTS config must be a mapping")

    tts_cfg = data.get("tts", {}) if isinstance(data.get("tts", {}), dict) else {}
    resolved = dict(default_cfg)
    resolved.update(tts_cfg)
    return resolved
