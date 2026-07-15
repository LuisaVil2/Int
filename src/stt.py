"""STT con 3 backends: Deepgram (key), faster-whisper (local), o subtítulos VTT.

Devuelve segmentos: list[Segment(start, lang, text)].
Para el test offline, la dirección de traducción se decide por el idioma detectado del segmento.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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
    import time

    import httpx

    q = {"model": model, "punctuate": "true", "smart_format": "true", **params}
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = httpx.post(DG_URL, params=q, content=wav_bytes,
                           headers={"Authorization": f"Token {api_key}",
                                    "Content-Type": "audio/wav"},
                           timeout=60)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 and e.response.status_code != 429:
                raise          # error de cliente (key mala, payload): reintentar no ayuda
            last = e
        except httpx.TransportError as e:   # red caída, timeout, reset
            last = e
        time.sleep(0.5 * (attempt + 1))
    raise last


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
                           model: str = "nova-3") -> tuple[str, str | None, float, int | None]:
    """Una utterance -> (texto, idioma, confianza, speaker_id). Para el bot en vivo.

    speaker_id es diarización POR-UTTERANCE (cada llamada es un request independiente):
    útil para distinguir hablantes superpuestos dentro de un mismo turno, pero NO es una
    identidad estable entre turnos (el hablante "0" de esta utterance no necesariamente
    es la misma persona física que el "0" de la próxima).
    """
    resp = _dg_post(np_to_wav_bytes(samples, sr), api_key, model,
                    {"language": "multi", "diarize": "true"})
    chans = resp.get("results", {}).get("channels", [])
    if not chans or not chans[0].get("alternatives"):
        return "", None, 0.0, None
    ch = chans[0]
    alt = ch["alternatives"][0]
    words = alt.get("words") or []
    lang = ch.get("detected_language") or (words[0].get("language") if words else None)
    speaker = words[0].get("speaker") if words else None
    return ((alt.get("transcript") or "").strip(), lang,
            float(alt.get("confidence", 0.85) or 0.85), speaker)


# ---------- Deepgram live (WebSocket streaming, para el bot en vivo) ----------
DG_WS_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramLive:
    """STT streaming: manda PCM continuo por WebSocket; Deepgram hace el endpointing
    en el servidor y esta clase entrega utterances completas por callback.

    on_utterance(text, lang, confidence) se llama desde el hilo receptor cuando llega
    un resultado con speech_final=True. nova-3 + language=multi soporta code-switching
    EN/ES; el idioma se toma por mayoría de las words del utterance.

    Si la conexión se cae (red, timeout del server), reconecta sola con backoff;
    on_error solo se llama cuando los reintentos se agotan.
    """

    RECONNECT_DELAYS = (0.5, 1.0, 2.0)

    def __init__(self, api_key: str, model: str = "nova-3", sr: int = 16000,
                 on_utterance: Callable[[str, str | None, float], None] | None = None,
                 on_error: Callable[[Exception], None] | None = None,
                 endpointing_ms: int = 400):
        self.api_key = api_key
        self.model = model
        self.sr = sr
        self.on_utterance = on_utterance
        self.on_error = on_error
        self.endpointing_ms = endpointing_ms
        self._ws = None
        self._rx: threading.Thread | None = None
        self._closed = False
        self._gen = 0                      # generación de conexión (para reconexión)
        self._lock = threading.Lock()

    def start(self):
        self._connect()
        self._rx = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx.start()

    def _connect(self):
        from urllib.parse import urlencode

        from websockets.sync.client import connect

        params = {"model": self.model, "language": "multi", "encoding": "linear16",
                  "sample_rate": str(self.sr), "channels": "1",
                  "punctuate": "true", "smart_format": "true",
                  "interim_results": "false", "endpointing": str(self.endpointing_ms)}
        self._ws = connect(f"{DG_WS_URL}?{urlencode(params)}",
                           additional_headers={"Authorization": f"Token {self.api_key}"})

    def _reconnect(self, gen: int, cause: Exception) -> bool:
        """Reintenta la conexión con backoff. True si hay conexión viva (esta llamada
        u otro hilo ya reconectó); False si se agotaron los intentos (queda cerrada)."""
        import time

        with self._lock:
            if self._closed:
                return False
            if gen != self._gen:           # otro hilo ya reconectó
                return True
            for delay in self.RECONNECT_DELAYS:
                time.sleep(delay)
                try:
                    self._connect()
                    self._gen += 1
                    return True
                except Exception:  # noqa
                    continue
            self._closed = True
            if self.on_error:
                self.on_error(cause)
            return False

    def send_np(self, samples) -> None:
        """Manda un bloque float32 [-1,1] como PCM16. Silencio también cuenta: mantiene
        viva la conexión (Deepgram corta tras ~10s sin audio)."""
        import numpy as np

        if self._closed or self._ws is None:
            return
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        gen = self._gen
        try:
            self._ws.send(pcm)
        except Exception as e:  # noqa — el chunk se pierde; la conexión se recupera
            self._reconnect(gen, e)

    def _recv_loop(self):
        while not self._closed:
            gen = self._gen
            # estado del utterance en curso; se descarta si la conexión se cae
            parts: list[str] = []
            langs: list[str] = []
            confs: list[float] = []
            try:
                for raw in self._ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "Results":
                        continue
                    alts = (msg.get("channel") or {}).get("alternatives") or [{}]
                    alt = alts[0]
                    text = (alt.get("transcript") or "").strip()
                    if text:
                        parts.append(text)
                        confs.append(float(alt.get("confidence", 0.85) or 0.85))
                        langs += [w["language"] for w in alt.get("words") or []
                                  if w.get("language")]
                    if msg.get("speech_final") and parts:
                        utter = " ".join(parts)
                        lang = max(set(langs), key=langs.count) if langs else None
                        conf = sum(confs) / len(confs)
                        parts, langs, confs = [], [], []
                        if self.on_utterance:
                            self.on_utterance(utter, lang, conf)
            except Exception as e:  # noqa — conexión caída: intenta revivirla
                if self._closed or not self._reconnect(gen, e):
                    break
            else:                       # el server cerró limpio (p. ej. timeout de audio)
                if self._closed or not self._reconnect(gen, ConnectionError("closed by server")):
                    break

    def close(self):
        self._closed = True
        try:
            self._ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:  # noqa
            pass
        try:
            self._ws.close()
        except Exception:  # noqa
            pass


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
