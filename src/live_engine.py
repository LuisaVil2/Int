"""Motor del intérprete en vivo. Usado por la GUI y por el CLI.

Captura audio (loopback del sistema o micrófono) -> segmenta por energía ->
STT (Deepgram si hay key, si no Whisper; Deepgram->Whisper por turno si Deepgram falla) ->
DeepSeek traduce -> emite evento -> TTS (Fish Speech, con fallback a edge-tts) reproduce
en el dispositivo de salida elegido.

Half-duplex: mientras el bot habla, descarta la captura para no oírse a sí mismo.

Confiabilidad: un turno que falla (STT, red, etc.) se registra y se salta — NUNCA
termina la sesión completa. Ver _process_one().
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time

import numpy as np

from .logging_utils import configure_logging, log_turn, new_session_id

SR = 16000
FRAME = 1600
SILENCE_HANG = 0.7
MIN_SPEECH = 0.6
MAX_UTTER = 15.0

logger = logging.getLogger(__name__)


def list_inputs() -> list[dict]:
    """Dispositivos de entrada para la GUI: parlantes (loopback) + micrófonos."""
    import soundcard as sc
    items = [{"kind": "loopback", "name": s.name, "label": f"[Sistema] {s.name}"}
             for s in sc.all_speakers()]
    items += [{"kind": "mic", "name": m.name, "label": f"[Micrófono] {m.name}"}
              for m in sc.all_microphones()]
    return items


def list_outputs() -> list[dict]:
    """Dispositivos de salida (para la voz del bot) vía sounddevice."""
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            out.append({"index": i, "label": d["name"]})
    return out


class LiveEngine:
    def __init__(self, on_event, input_choice: dict, output_index: int | None,
                 tts_on: bool = True, model: str = "small", specialty: str = "general"):
        self.on_event = on_event
        self.input_choice = input_choice
        self.output_index = output_index
        self.tts_on = tts_on
        self.model = model
        self.specialty = specialty
        self._stop = threading.Event()
        self._speaking = threading.Event()   # half-duplex
        self._q: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._whisper_model = None  # cache perezoso: solo se carga si se necesita
        self._session_id = new_session_id()
        self._turn_id = 0

    # ---------- ciclo de vida ----------
    def start(self):
        configure_logging()
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._capture, daemon=True),
            threading.Thread(target=self._process, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()

    def _emit(self, kind: str, **data):
        try:
            self.on_event(kind, data)
        except Exception:  # noqa
            pass

    # ---------- captura ----------
    def _open_input(self):
        import soundcard as sc
        ch = self.input_choice
        if ch["kind"] == "loopback":
            return sc.get_microphone(id=str(ch["name"]), include_loopback=True)
        return sc.get_microphone(id=str(ch["name"]))

    def _capture(self):
        try:
            dev = self._open_input()
            with dev.recorder(samplerate=SR, channels=1, blocksize=FRAME) as rec:
                noise = np.median([self._rms(rec.record(FRAME).flatten()) for _ in range(10)])
                thresh = max(0.006, noise * 3.0)
                self._emit("status", text="Bot activo. Escuchando…")
                buf, silence, speaking = [], 0.0, False
                while not self._stop.is_set():
                    chunk = rec.record(FRAME).flatten().astype(np.float32)
                    if self._speaking.is_set():       # half-duplex: ignora mientras habla
                        buf, silence, speaking = [], 0.0, False
                        continue
                    level = self._rms(chunk)
                    if level >= thresh:
                        buf.append(chunk); silence = 0.0; speaking = True
                    elif speaking:
                        buf.append(chunk); silence += FRAME / SR
                        dur = len(buf) * FRAME / SR
                        if silence >= SILENCE_HANG and dur >= MIN_SPEECH:
                            self._q.put(np.concatenate(buf)); buf, silence, speaking = [], 0.0, False
                    if speaking and len(buf) * FRAME / SR >= MAX_UTTER:
                        self._q.put(np.concatenate(buf)); buf, silence, speaking = [], 0.0, False
        except Exception as e:  # noqa
            self._emit("error", text=f"Captura: {e}")

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2)) + 1e-9)

    # ---------- STT con fallback Deepgram -> Whisper por turno ----------
    def _get_whisper(self):
        """Carga perezosa: solo la primera vez que realmente se necesita Whisper
        (Deepgram no configurado, o Deepgram falló en este turno). Nunca recarga
        el modelo una vez cacheado."""
        if self._whisper_model is None:
            from faster_whisper import WhisperModel
            self._whisper_model = WhisperModel(self.model, device="cpu", compute_type="int8")
        return self._whisper_model

    def _transcribe(self, audio: np.ndarray, dg_key: str | None):
        """-> (text, lang, stt_confidence, speaker_id, backend_name)."""
        if dg_key:
            try:
                from .stt import transcribe_np_deepgram
                text, lang, conf, speaker = transcribe_np_deepgram(
                    audio, SR, dg_key, os.getenv("DEEPGRAM_MODEL", "nova-2-medical"))
                return text, lang, conf, speaker, "deepgram"
            except Exception as e:  # noqa
                self._emit("status", text=f"Deepgram falló ({e}); Whisper para este turno")
                logger.warning("deepgram_fallback_triggered", extra={"meta": {"error": str(e)}})
        whisper = self._get_whisper()
        segs, info = whisper.transcribe(audio, vad_filter=True, beam_size=1)
        text = " ".join(s.text.strip() for s in segs).strip()
        return text, info.language, 0.85, None, "whisper"

    # ---------- procesamiento ----------
    def _process(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from .translate import DeepSeekTranslator
            from .terminology import TerminologyIndex
            from .segmentation import detect_lang
            from .memory import ConversationMemory
            from .emergency import EmergencyClassifier
            from .confidence import ConfidenceEngine

            dg_key = os.getenv("DEEPGRAM_API_KEY")
            translator = DeepSeekTranslator()
            idx = TerminologyIndex.load(os.getenv("TERMINOLOGY_DIR", "data/terminology"))
            memory = ConversationMemory()
            emerg = EmergencyClassifier()
            confidence_engine = ConfidenceEngine()
            self._emit("status", text=f"STT={'Deepgram' if dg_key else 'Whisper'} · LLM={translator.model}")
        except Exception as e:  # noqa - fallas de arranque SÍ son fatales para la sesión
            self._emit("error", text=f"Inicialización: {e}")
            return

        while not self._stop.is_set():
            try:
                audio = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._process_one(audio, dg_key, translator, idx, detect_lang, memory,
                                  emerg, confidence_engine)
            except Exception as e:  # noqa - un turno roto NUNCA mata la sesión
                self._emit("error", text=f"Turno omitido: {e}")
                logger.exception("turn_failed")

    def _process_one(self, audio, dg_key, translator, idx, detect_lang, memory,
                     emerg, confidence_engine) -> None:
        from .emergency import ALERT_EN, ALERT_ES
        from .confidence import ConfidenceInputs

        t_stt0 = time.perf_counter()
        text, lang, stt_conf, speaker_id, stt_backend = self._transcribe(audio, dg_key)
        stt_ms = round((time.perf_counter() - t_stt0) * 1000)
        if not text:
            return

        lang = detect_lang(text, fallback=lang) or lang or "en"
        hits = idx.lookup(text, self.specialty)
        term_block = "\n".join(hits) if hits else "(sin términos relevantes)"
        self._emit("src", lang=lang, text=text, speaker=speaker_id)

        t0 = time.perf_counter()
        tr = translator.translate(text, lang, term_block, memory.context(limit=6))
        lat = round((time.perf_counter() - t0) * 1000)
        out = tr.text

        em = emerg.classify(text)
        if em["is_emergency"] and "<UNCLEAR>" not in out:
            out += f"  {ALERT_EN if tr.target_lang == 'en' else ALERT_ES}"

        if not tr.needs_clarification:
            memory.add_turn(speaker_id, lang, text, tr.text)

        inputs = ConfidenceInputs(
            asr_confidence=stt_conf,
            llm_confidence=tr.confidence,
            # proxies heurísticos basados en aciertos de terminología, no señales calibradas.
            terminology_certainty=0.9 if hits else 0.75,
            glossary_match=min(1.0, len(hits) / 5),
        )
        score = confidence_engine.score(inputs)
        route = confidence_engine.route(score)
        if em["force_qa_review"] and route == "automatic_approval":
            route = "qa_review"

        self._turn_id += 1
        self._emit("translation", src_lang=lang, tgt_lang=tr.target_lang,
                   text=out, latency_ms=lat, stt_ms=stt_ms, stt_backend=stt_backend,
                   speaker=speaker_id, confidence_score=score, route=route)
        if route == "pause_manual_approval":
            self._emit("needs_review", text=out, src_lang=lang, tgt_lang=tr.target_lang, score=score)

        tts_ms = 0
        if self.tts_on and out and "<UNCLEAR>" not in out:
            tts_ms = self._speak(out, tr.target_lang)

        log_turn(logger, session_id=self._session_id, turn_id=self._turn_id,
                 source_text=text, output_text=out, lang=lang, target_lang=tr.target_lang,
                 stt_ms=stt_ms, stt_backend=stt_backend, translation_ms=lat, tts_ms=tts_ms,
                 total_ms=stt_ms + lat + tts_ms, confidence_score=score, route=route,
                 speaker=speaker_id, is_emergency=em["is_emergency"])

    def _speak(self, text: str, lang: str) -> int:
        """Devuelve la latencia de síntesis (ms) para instrumentación; 0 si falló."""
        t0 = time.perf_counter()
        tts_ms = 0
        try:
            import sounddevice as sd
            from .tts import synthesize

            samples, sr = synthesize(text, lang)
            tts_ms = round((time.perf_counter() - t0) * 1000)
            if samples.size == 0:
                return tts_ms
            self._speaking.set()
            sd.play(samples, sr, device=self.output_index)
            sd.wait()
        except Exception as e:  # noqa
            self._emit("error", text=f"TTS: {e}")
        finally:
            time.sleep(0.15)          # deja morir el eco antes de re-escuchar
            self._speaking.clear()
        return tts_ms
