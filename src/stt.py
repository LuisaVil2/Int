"""STT con 3 backends: Deepgram (key), faster-whisper (local), o subtítulos VTT.

Devuelve segmentos: list[Segment(start, lang, text)].
Para el test offline, la dirección de traducción se decide por el idioma detectado del segmento.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float          # segundos
    text: str
    lang: str | None      # 'en' | 'es' | None
    speaker: int | None = None
    asr_confidence: float = 0.85


# ---------- Whisper local (faster-whisper) ----------
def transcribe_whisper(audio_path: str, model_size: str | None = None) -> list[Segment]:
    from faster_whisper import WhisperModel  # type: ignore

    model_size = model_size or os.getenv("WHISPER_MODEL", "small")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # multilingual=True => detección de idioma por-segmento (clave en escenas EN/ES mezcladas).
    kwargs = dict(vad_filter=True, beam_size=5)
    try:
        segments, info = model.transcribe(audio_path, multilingual=True, **kwargs)
    except TypeError:  # versión vieja sin multilingual: detección global
        segments, info = model.transcribe(audio_path, **kwargs)
    out: list[Segment] = []
    for s in segments:
        lang = getattr(s, "language", None) or getattr(info, "language", None)
        out.append(Segment(start=round(s.start, 2), text=s.text.strip(), lang=lang))
    return out


# ---------- Deepgram vía REST (SDK v7 cambió la API; REST es estable) ----------
# nova-3 + language=multi = multilingüe EN/ES con code-switching.
DG_URL = "https://api.deepgram.com/v1/listen"


def _dg_post(wav_bytes: bytes, api_key: str, model: str, params: dict) -> dict:
    import httpx

    q = {"model": model, "punctuate": "true", "smart_format": "true", **params}
    r = httpx.post(DG_URL, params=q, content=wav_bytes,
                   headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
                   timeout=60)
    r.raise_for_status()
    return r.json()


def transcribe_deepgram(audio_path: str, api_key: str, model: str = "nova-3") -> list[Segment]:
    with open(audio_path, "rb") as f:
        data = f.read()
    resp = _dg_post(data, api_key, model, {"language": "multi", "utterances": "true",
                                           "diarize": "true"})
    out: list[Segment] = []
    for u in resp.get("results", {}).get("utterances", []) or []:
        words = u.get("words") or []
        out.append(Segment(start=round(u.get("start", 0.0), 2),
                           text=(u.get("transcript") or "").strip(),
                           lang=u.get("language") or (words[0].get("language") if words else None),
                           speaker=words[0].get("speaker") if words else None,
                           asr_confidence=float(u.get("confidence", 0.85) or 0.85)))
    return out


# ---------- Deepgram por-utterance desde buffer numpy (para modo en vivo) ----------
def np_to_wav_bytes(samples, sr: int = 16000) -> bytes:
    import io
    import wave
    import numpy as np

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe_np_deepgram(samples, sr, api_key: str,
                           model: str = "nova-3") -> tuple[str, str | None, float]:
    """Una utterance -> (texto, idioma, confianza). Para el bot en vivo."""
    resp = _dg_post(np_to_wav_bytes(samples, sr), api_key, model, {"language": "multi"})
    chans = resp.get("results", {}).get("channels", [])
    if not chans or not chans[0].get("alternatives"):
        return "", None, 0.0
    ch = chans[0]
    alt = ch["alternatives"][0]
    words = alt.get("words") or []
    lang = ch.get("detected_language") or (words[0].get("language") if words else None)
    return (alt.get("transcript") or "").strip(), lang, float(alt.get("confidence", 0.85) or 0.85)


# ---------- Subtítulos VTT (gratis, vía yt-dlp) ----------
_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")


def load_vtt(path: str | Path) -> list[Segment]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[Segment] = []
    start = 0.0
    buf: list[str] = []
    for ln in lines:
        if "-->" in ln:
            m = _TS.search(ln)
            if m:
                h, mn, s, ms = map(int, m.groups())
                start = h * 3600 + mn * 60 + s + ms / 1000
            buf = []
        elif ln.strip() == "":
            if buf:
                txt = _clean(" ".join(buf))
                if txt:
                    out.append(Segment(start=round(start, 2), text=txt, lang=None))
            buf = []
        elif "WEBVTT" in ln or ln.strip().isdigit() or "Kind:" in ln or "Language:" in ln:
            continue
        else:
            buf.append(ln)
    if buf:
        txt = _clean(" ".join(buf))
        if txt:
            out.append(Segment(start=round(start, 2), text=txt, lang=None))
    return _dedup(out)


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)          # tags <c> de YT
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedup(segs: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        if out and s.text == out[-1].text:
            continue
        out.append(s)
    return out


def get_segments(audio_path: str | None, vtt_path: str | None) -> list[Segment]:
    """Dispatcher: Deepgram > Whisper > VTT, según lo disponible."""
    dg_key = os.getenv("DEEPGRAM_API_KEY")
    if audio_path and dg_key:
        try:
            return transcribe_deepgram(audio_path, dg_key,
                                       os.getenv("DEEPGRAM_MODEL", "nova-3"))
        except Exception as e:  # noqa
            print(f"[stt] Deepgram falló ({e}); intento Whisper")
    if audio_path:
        try:
            return transcribe_whisper(audio_path)
        except Exception as e:  # noqa
            print(f"[stt] Whisper no disponible ({e}); uso subtítulos VTT")
    if vtt_path and Path(vtt_path).exists():
        return load_vtt(vtt_path)
    raise RuntimeError("Sin backend STT: no hay key Deepgram, ni faster-whisper, ni VTT.")
