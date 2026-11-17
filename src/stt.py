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
                           asr_confidence=float(u.get("confidence") or 0.0)))
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
            float(alt.get("confidence") or 0.0), speaker)


# ---------- Deepgram live (WebSocket streaming, para el bot en vivo) ----------
DG_WS_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramLive:
    """STT streaming: manda PCM continuo por WebSocket; Deepgram hace el endpointing
    en el servidor y esta clase entrega utterances completas por callback.

    Solo se acumulan resultados finales. Una discontinuidad descarta el turno
    pendiente y se comunica al motor mediante on_gap(reason).
    """

    RECONNECT_DELAYS = (0.5, 1.0, 2.0)

    def __init__(self, api_key: str, model: str = "nova-3", sr: int = 16000,
                 on_utterance: Callable[[str, str | None, float], None] | None = None,
                 on_error: Callable[[Exception], None] | None = None,
                 endpointing_ms: int = 400,
                 on_gap: Callable[[str], None] | None = None):
        self.api_key = api_key
        self.model = model
        self.sr = sr
        self.on_utterance = on_utterance
        self.on_error = on_error
        self.endpointing_ms = endpointing_ms
        self.on_gap = on_gap
        self._cancel = threading.Event()
        self._muted = False
        self._mute_epoch = 0
        self._ws = None
        self._rx: threading.Thread | None = None
        self._closed = False
        self._gen = 0                      # generación de conexión (para reconexión)
        self._lock = threading.Lock()

    def start(self):
        ws = self._connect()
        with self._lock:
            if self._cancel.is_set():
                ws.close()
                return
            self._ws = ws
        self._rx = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx.start()

    def _connect(self):
        from urllib.parse import urlencode

        from websockets.sync.client import connect

        params = {"model": self.model, "language": "multi", "encoding": "linear16",
                  "sample_rate": str(self.sr), "channels": "1",
                  "punctuate": "true", "smart_format": "true",
                  "interim_results": "true", "utterance_end_ms": "1000", "diarize": "true",
                  "endpointing": str(self.endpointing_ms)}
        return connect(f"{DG_WS_URL}?{urlencode(params)}",
                       additional_headers={"Authorization": f"Token {self.api_key}"},
                       open_timeout=5, close_timeout=1)

    def _reconnect(self, gen: int, cause: Exception) -> bool:
        """Solo el receptor reconecta. El capturador nunca espera el backoff."""
        if self._cancel.is_set():
            return False
        self._report_gap("stt_connection_lost")
        for delay in self.RECONNECT_DELAYS:
            if self._cancel.wait(delay):
                return False
            try:
                ws = self._connect()
            except Exception:
                continue
            with self._lock:
                if not self._cancel.is_set():
                    self._ws = ws
                    self._gen = gen + 1
                    return True
            ws.close()
            return False
        self._closed = True
        if self.on_error:
            self.on_error(cause)
        return False

    def _report_gap(self, reason: str) -> None:
        if self.on_gap and not self._cancel.is_set():
            self.on_gap(reason)

    def set_capture_muted(self, muted: bool) -> None:
        if muted and not self._muted:
            self._mute_epoch += 1
        self._muted = muted

    @property
    def closed(self) -> bool:
        return self._closed

    def send_np(self, samples) -> None:
        """Envía PCM16. Durante reconexión se descartan frames sin reproducirlos."""
        import numpy as np

        ws = self._ws
        if self._closed or ws is None:
            return
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        try:
            ws.send(pcm)
        except Exception:
            # Despierta el receptor; la recuperación y el cambio de socket son suyos.
            self._close_socket(ws)

    @staticmethod
    def _close_socket(ws) -> None:
        try:
            ws.close()
        except Exception:
            pass

    def _recv_loop(self):
        while not self._closed:
            gen, ws = self._gen, self._ws
            pending = _FinalResults(discontinuous=gen > 0)
            mute_epoch = self._mute_epoch
            cause = ConnectionError("STT connection closed")
            try:
                for raw in ws:
                    if self._cancel.is_set():
                        return
                    msg = json.loads(raw)
                    if self._muted or mute_epoch != self._mute_epoch:
                        if pending.parts or pending.interim_word_end >= 0 or self._muted:
                            pending.incomplete = True
                        mute_epoch = self._mute_epoch
                    turn, issue = pending.accept(msg)
                    if issue:
                        self._report_gap(issue)
                    if turn and self.on_utterance:
                        self.on_utterance(*turn)
            except Exception as exc:
                cause = exc
            finally:
                self._close_socket(ws)
                with self._lock:
                    if self._ws is ws:
                        self._ws = None
            if self._closed or not self._reconnect(gen, cause):
                break

    def close(self):
        self._cancel.set()
        self._closed = True
        with self._lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            self._close_socket(ws)
        if self._rx is not None and self._rx is not threading.current_thread():
            self._rx.join(timeout=1)


_SHARED_MEDICAL_TERMS = {"metformin", "metformina", "insulin", "insulina", "diabetes", "gerd"}


class _FinalResults:
    """Acumula rangos en orden dentro de un solo WebSocket mono."""

    def __init__(self, discontinuous: bool = False):
        self.parts: list[str] = []
        self.languages: set[str] = set()
        self.speakers: set[int] = set()
        self.confidences: list[float] = []
        self.last_range: tuple[float, float] | None = None
        self.final_word_end = -1.0
        self.interim_word_end = -1.0
        self.incomplete = discontinuous

    def accept(self, msg: dict):
        kind = msg.get("type")
        if kind == "UtteranceEnd":
            boundary = msg.get("last_word_end", -1)
            if self.parts and boundary >= self.final_word_end - 0.001:
                return self._finish()
            return None, None
        if kind != "Results":
            return None, None

        alt = ((msg.get("channel") or {}).get("alternatives") or [{}])[0]
        text = (alt.get("transcript") or "").strip()
        words = alt.get("words") or []
        start = float(msg.get("start", 0))
        end = start + float(msg.get("duration", 0))
        word_end = float(words[-1].get("end", end)) if words else end
        if not msg.get("is_final"):
            if text:
                self.interim_word_end = max(self.interim_word_end, word_end)
            return None, None

        if text:
            if self.last_range and (start < self.last_range[1] - 0.001
                                    or (start, end) == self.last_range):
                if (start, end) == self.last_range:
                    if msg.get("speech_final") and self.parts:
                        return self._finish()
                    return None, None
                if end <= self.last_range[1] + 0.001:
                    return None, None
                self.incomplete = True
            self.last_range = (start, end)
            if len(self.parts) < 64 and not self.incomplete:
                self.parts.append(text)
            elif len(self.parts) >= 64:
                self.incomplete = True
            if not self.parts:
                # Mantener un marcador hasta el siguiente fin de turno.
                self.parts.append("")
            self.final_word_end = word_end
            tags = {w["language"] for w in words if w.get("language")
                    and w.get("word", "").casefold() not in _SHARED_MEDICAL_TERMS}
            if not any(w.get("language") for w in words):
                tags = set(alt.get("languages") or [])
            self.languages.update(tag.lower().split("-")[0] for tag in tags)
            self.speakers.update(w["speaker"] for w in words if w.get("speaker") is not None)
            conf = alt.get("confidence")
            if not self.confidences or not self.incomplete:
                self.confidences.append(float(conf) if conf is not None else 0.0)
        if msg.get("speech_final") and self.parts:
            return self._finish()
        return None, None

    def _finish(self):
        incomplete = self.incomplete or self.interim_word_end > self.final_word_end + 0.05
        multiple_speakers = len(self.speakers) > 1
        lang = next(iter(self.languages)) if len(self.languages) == 1 else None
        if len(self.languages) > 1:
            lang = "mixed"
        turn = (" ".join(self.parts), lang, min(self.confidences))
        self.parts.clear()
        self.languages.clear()
        self.speakers.clear()
        self.confidences.clear()
        self.incomplete = False
        self.interim_word_end = -1.0
        if incomplete:
            return None, "stt_incomplete_turn"
        if multiple_speakers:
            return None, "stt_multiple_speakers"
        return turn, None


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
        except Exception:
            print("[stt] Deepgram falló; intento Whisper")
    if audio_path:
        try:
            return transcribe_whisper(audio_path)
        except Exception:
            print("[stt] Whisper no disponible; uso subtítulos VTT")
    if vtt_path and Path(vtt_path).exists():
        return load_vtt(vtt_path)
    raise RuntimeError("Sin backend STT: no hay key Deepgram, ni faster-whisper, ni VTT.")
