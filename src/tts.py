"""TTS barato sin key — edge-tts (voces neuronales femeninas EN/ES, gratis).

Devuelve PCM float32 mono para reproducir por sounddevice en el dispositivo elegido.
Decodifica el mp3 de edge-tts con ffmpeg (ya instalado). Swappeable luego por Piper
(HF, local) o MeloTTS (chino) sin tocar el resto.
"""
from __future__ import annotations

import asyncio
import io
import os
import subprocess
import wave

import numpy as np

# voces femeninas por idioma
VOICES = {
    "es": os.getenv("TTS_VOICE_ES", "es-MX-DaliaNeural"),
    "en": os.getenv("TTS_VOICE_EN", "en-US-AriaNeural"),
}


def _ffmpeg() -> str:
    return os.getenv("FFMPEG_PATH", "ffmpeg")


async def _mp3_bytes(text: str, voice: str) -> bytes:
    import edge_tts

    buf = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)


def synthesize(text: str, lang: str) -> tuple[np.ndarray, int]:
    """text+lang -> (samples float32 mono, samplerate). Vacío si falla."""
    voice = VOICES.get(lang, VOICES["en"])
    mp3 = asyncio.run(_mp3_bytes(text, voice))
    if not mp3:
        return np.zeros(0, dtype=np.float32), 16000
    # mp3 -> wav PCM s16le 16k mono via ffmpeg (stdin->stdout)
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
        input=mp3, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32), 16000
    with wave.open(io.BytesIO(proc.stdout), "rb") as w:
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    s, sr = synthesize("Buenos días, ¿en qué le puedo ayudar?", "es")
    print(f"sintetizado {len(s)} muestras @ {sr}Hz")
