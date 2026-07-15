from __future__ import annotations

import io
import os
import time
import wave
from collections import deque
from typing import Any

import numpy as np
import requests

from .common import BaseTTSProvider

_LATENCY_WINDOW = 50


class FishSpeechProvider(BaseTTSProvider):
    """TTS provider backed by the Fish Audio (Fish Speech) cloud API."""

    API_URL = "https://api.fish.audio/v1/tts"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._api_key = os.getenv("FISH_API_KEY", "")
        self._reference_id = config.get("fish_reference_id") or os.getenv("FISH_VOICE_ID", "")
        self._model = os.getenv("FISH_MODEL", "s2.1-pro-free")
        self._timeout = float(os.getenv("FISH_TIMEOUT", "30"))
        self._synthesis_latency_ms = 0.0
        self._latencies_ms: deque[float] = deque(maxlen=_LATENCY_WINDOW)
        # Sesión persistente: reutiliza la conexión TCP/TLS entre llamadas en vez de
        # abrir una nueva por cada síntesis (nunca recrear clientes de red).
        self._session = requests.Session()

        self._init_error: Exception | None = None
        if not self._api_key:
            self._init_error = RuntimeError("FISH_API_KEY is not set")
        elif not self._reference_id:
            self._init_error = RuntimeError("FISH_VOICE_ID is not set")
        self._available = self._init_error is None

    def benchmark_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "provider": "fish",
            "initialized": self._available,
            "average_synthesis_latency_ms": self._synthesis_latency_ms,
            "reference_id": self._reference_id,
            "sample_count": len(self._latencies_ms),
        }
        if self._latencies_ms:
            ordered = sorted(self._latencies_ms)
            stats["mean_ms"] = round(sum(ordered) / len(ordered), 2)
            stats["p50_ms"] = round(ordered[len(ordered) // 2], 2)
            stats["p95_ms"] = round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2)
        return stats

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        if not self._available:
            raise RuntimeError(f"Fish Speech not configured: {self._init_error}")

        start = time.perf_counter()
        payload = {
            "text": text,
            "reference_id": self._reference_id,
            "format": "wav",
            "normalize": True,
            "latency": "normal",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "model": self._model,
        }

        try:
            resp = self._session.post(
                self.API_URL, json=payload, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
            audio_bytes = resp.content
        except Exception as exc:
            raise RuntimeError(f"Fish Speech API request failed: {exc}") from exc

        if not audio_bytes:
            raise RuntimeError("Fish Speech API returned empty audio")

        samples, sr = self._decode_wav(audio_bytes)
        self._synthesis_latency_ms = round((time.perf_counter() - start) * 1000, 2)
        self._latencies_ms.append(self._synthesis_latency_ms)
        return samples, sr

    @staticmethod
    def _decode_wav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            sr = w.getframerate()
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            frames = w.readframes(w.getnframes())

        if sampwidth == 2:
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        return samples, sr
