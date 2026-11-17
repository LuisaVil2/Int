"""TTS wrapper with Fish Speech primary provider and edge-tts fallback."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .common import EdgeTTSProvider, load_tts_config as _load_tts_config
from .fish_provider import FishSpeechProvider

_CONFIG_PATH = None
_PROVIDER: Any | None = None


def _get_provider() -> Any:
    global _PROVIDER

    config = load_tts_config()
    provider_name = config.get("provider", "fish")
    if provider_name == "fish":
        expected_cls = FishSpeechProvider
    else:
        expected_cls = EdgeTTSProvider

    if _PROVIDER is None or type(_PROVIDER) is not expected_cls:
        _PROVIDER = expected_cls(config)
    return _PROVIDER


def initialize_tts() -> Any:
    return _get_provider()


def synthesize(text: str, lang: str) -> tuple[np.ndarray, int]:
    """text+lang -> (samples float32 mono, samplerate). Vacío si falla."""
    try:
        provider = _get_provider()
        samples, sr = provider.synthesize(text, lang)
        if samples.size or isinstance(provider, EdgeTTSProvider):
            return samples, sr
    except Exception:
        pass
    try:
        return EdgeTTSProvider(load_tts_config()).synthesize(text, lang)
    except Exception:
        return np.zeros(0, dtype=np.float32), 16000


def load_tts_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    return _load_tts_config(config_path or _CONFIG_PATH)


def benchmark_stats() -> dict[str, Any]:
    try:
        provider = _get_provider()
        if hasattr(provider, "benchmark_stats"):
            return provider.benchmark_stats()
    except Exception:
        pass
    return {"provider": "edge", "initialized": False}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    s, sr = synthesize("Buenos días, ¿en qué le puedo ayudar?", "es")
    print(f"sintetizado {len(s)} muestras @ {sr}Hz")
