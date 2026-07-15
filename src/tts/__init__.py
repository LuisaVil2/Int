"""TTS wrapper with Fish Speech primary provider and edge-tts fallback."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .common import EdgeTTSProvider, load_tts_config as _load_tts_config
from .fish_provider import FishSpeechProvider

_CONFIG_PATH = os.getenv("TTS_CONFIG_PATH", "config.yaml")
_PROVIDER: Any | None = None


def _get_provider() -> Any:
    global _PROVIDER

    config = load_tts_config(_CONFIG_PATH)
    provider_name = config.get("provider", "fish")
    if provider_name == "fish":
        expected_cls = FishSpeechProvider
    else:
        expected_cls = EdgeTTSProvider

    if _PROVIDER is None or type(_PROVIDER) is not expected_cls:
        try:
            _PROVIDER = expected_cls(config)
        except TypeError:
            _PROVIDER = expected_cls()
    return _PROVIDER


def initialize_tts() -> Any:
    return _get_provider()


def synthesize(text: str, lang: str) -> tuple[np.ndarray, int]:
    """text+lang -> (samples float32 mono, samplerate). Vacío si falla."""
    try:
        provider = _get_provider()
        return provider.synthesize(text, lang)
    except Exception:
        fallback_config = load_tts_config(_CONFIG_PATH)
        try:
            fallback_provider = EdgeTTSProvider(fallback_config)
        except TypeError:
            fallback_provider = EdgeTTSProvider()
        try:
            return fallback_provider.synthesize(text, lang)
        except Exception:
            return np.zeros(0, dtype=np.float32), 16000


def load_tts_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    return _load_tts_config(config_path)


def benchmark_stats() -> dict[str, Any]:
    try:
        provider = _get_provider()
        if hasattr(provider, "benchmark_stats"):
            return provider.benchmark_stats()
    except Exception:
        pass
    return {"provider": "edge", "initialized": False}


initialize_tts()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    s, sr = synthesize("Buenos días, ¿en qué le puedo ayudar?", "es")
    print(f"sintetizado {len(s)} muestras @ {sr}Hz")
