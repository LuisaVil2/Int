"""Motor del intérprete en vivo. Usado por la GUI y por el CLI.

Arquitectura pipeline (etapas desacopladas por colas): mientras el bot habla el
turno N ya está transcribiendo y traduciendo el N+1.

  captura ─frames→ STT ─texto→ proceso (terminología + DeepSeek) ─frases→ síntesis TTS ─audio→ reproducción

Streaming de punta a punta:
- STT: Deepgram live WebSocket (endpointing en el servidor) si hay key;
  si no, VAD local por energía + Whisper (Deepgram REST->Whisper por turno
  como camino alternativo, ver _transcribe()).
- LLM: la traducción se streamea; cada frase completa entra a TTS sin esperar
  el JSON entero (ver TextFieldStream en translate.py).
- TTS: Fish Speech (voz clonada) con fallback a edge-tts; la síntesis de la
  frase siguiente se solapa con la reproducción de la actual.

Half-duplex: mientras reproduce voz, la captura manda silencio (no se oye a sí mismo).

Confiabilidad: un turno que falla (STT, red, etc.) se registra y se salta — NUNCA
termina la sesión completa. Ver _process() / _process_one().
"""
from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time

import numpy as np

from .logging_utils import configure_logging, log_turn, new_session_id

SR = 16000
FRAME = 1600
SILENCE_HANG = 0.7
MIN_SPEECH = 0.6
MAX_UTTER = 15.0
MIN_SENTENCE = 12          # chars mínimos antes de mandar una frase a TTS

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


_SENT_BOUNDARY = re.compile(r'[.!?…:;]["\')\]]?\s')


class _SentenceSplitter:
    """Acumula deltas de texto y emite frases completas al callback."""

    def __init__(self, emit):
        self.buf = ""
        self.emit = emit

    def feed(self, delta: str):
        self.buf += delta
        while True:
            m = _SENT_BOUNDARY.search(self.buf)
            if not m or m.end() < MIN_SENTENCE:
                break
            sent = self.buf[:m.end()].strip()
            self.buf = self.buf[m.end():]
            if sent:
                self.emit(sent)

    def flush(self):
        sent = self.buf.strip()
        self.buf = ""
        if sent:
            self.emit(sent)


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
        self._utter_q: queue.Queue = queue.Queue()        # np audio -> Whisper
        self._text_q: queue.Queue = queue.Queue()         # (texto, lang, conf) -> proceso
        self._tts_q: queue.Queue = queue.Queue()          # (frase, lang) -> síntesis
        self._play_q: queue.Queue = queue.Queue(maxsize=8)  # (samples, sr) -> reproducción
        self._dg = None
        self._dg_key: str | None = None
        self._threads: list[threading.Thread] = []
        self._whisper_model = None  # cache perezoso: solo se carga si se necesita
        self._session_id = new_session_id()
        self._turn_id = 0

    # ---------- ciclo de vida ----------
    def start(self):
        from dotenv import load_dotenv
        load_dotenv()
        configure_logging()
        self._stop.clear()
        self._dg_key = os.getenv("DEEPGRAM_API_KEY") or None
        targets = [self._capture, self._process, self._tts_synth, self._tts_play]
        if not self._dg_key:
            targets.append(self._stt_whisper)
        self._threads = [threading.Thread(target=t, daemon=True) for t in targets]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        if self._dg:
            self._dg.close()
            self._dg = None
        try:
            import sounddevice as sd
            sd.stop()                     # corta reproducción en curso
        except Exception:  # noqa
            pass
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        self._speaking.clear()

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
            if self._dg_key:
                from .stt import DeepgramLive
                self._dg = DeepgramLive(
                    self._dg_key,
                    model=os.getenv("DEEPGRAM_LIVE_MODEL", "nova-3"), sr=SR,
                    on_utterance=lambda t, l, c: self._text_q.put((t, l, c)),
                    on_error=lambda e: self._emit("error", text=f"Deepgram: {e}"),
                )
                self._dg.start()
            with dev.recorder(samplerate=SR, channels=1, blocksize=FRAME) as rec:
                if self._dg:
                    self._capture_streaming(rec)
                else:
                    self._capture_vad(rec)
        except Exception as e:  # noqa
            self._emit("error", text=f"Captura: {e}")
        finally:
            if self._dg:
                self._dg.close()

    def _capture_streaming(self, rec):
        """Modo Deepgram: frames continuos al WebSocket; el endpointing lo hace el server."""
        self._emit("status", text="Bot activo. Escuchando… (Deepgram streaming)")
        zeros = np.zeros(FRAME, dtype=np.float32)
        while not self._stop.is_set():
            chunk = rec.record(FRAME).flatten().astype(np.float32)
            if self._speaking.is_set():   # half-duplex: silencio mantiene viva la conexión
                chunk = zeros
            self._dg.send_np(chunk)

    def _make_vad(self, rec):
        """is_speech(chunk) -> bool. Silero (ONNX, embebido en faster-whisper) con
        fallback a energía RMS si no carga. Silero aguanta música/ruido de fondo que
        rompe el umbral por energía."""
        try:
            from faster_whisper.vad import get_vad_model
            model = get_vad_model()
            thresh = float(os.getenv("VAD_THRESHOLD", "0.5"))
            win = (FRAME // 512) * 512      # silero exige múltiplos de 512 samples

            def is_speech(chunk: np.ndarray) -> bool:
                return float(model(chunk[:win]).max()) >= thresh

            self._emit("status", text="VAD: silero")
            return is_speech
        except Exception as e:  # noqa
            self._emit("status", text=f"VAD silero no disponible ({e}); uso energía RMS")
            noise = np.median([self._rms(rec.record(FRAME).flatten()) for _ in range(10)])
            level_thresh = max(0.006, noise * 3.0)
            return lambda chunk: self._rms(chunk) >= level_thresh

    def _capture_vad(self, rec):
        """Modo local: segmenta con VAD y encola utterances para Whisper."""
        is_speech = self._make_vad(rec)
        self._emit("status", text="Bot activo. Escuchando…")
        buf, silence, speaking = [], 0.0, False
        while not self._stop.is_set():
            chunk = rec.record(FRAME).flatten().astype(np.float32)
            if self._speaking.is_set():       # half-duplex: ignora mientras habla
                buf, silence, speaking = [], 0.0, False
                continue
            if is_speech(chunk):
                buf.append(chunk); silence = 0.0; speaking = True
            elif speaking:
                buf.append(chunk); silence += FRAME / SR
                dur = len(buf) * FRAME / SR
                if silence >= SILENCE_HANG and dur >= MIN_SPEECH:
                    self._utter_q.put(np.concatenate(buf)); buf, silence, speaking = [], 0.0, False
            if speaking and len(buf) * FRAME / SR >= MAX_UTTER:
                self._utter_q.put(np.concatenate(buf)); buf, silence, speaking = [], 0.0, False

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2)) + 1e-9)

    # ---------- STT con fallback Deepgram REST -> Whisper por turno ----------
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

    def _stt_whisper(self):
        """Hilo STT local (solo sin key Deepgram): consume utterances del VAD."""
        try:
            while not self._stop.is_set():
                try:
                    audio = self._utter_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                text, lang, conf, _speaker, _backend = self._transcribe(audio, None)
                if text:
                    self._text_q.put((text, lang, conf))
        except Exception as e:  # noqa
            self._emit("error", text=f"Whisper: {e}")

    # ---------- procesamiento ----------
    def _process(self):
        try:
            from .translate import DeepSeekTranslator
            from .terminology import TerminologyIndex
            from .segmentation import detect_lang
            from .memory import ConversationMemory
            from .emergency import EmergencyClassifier
            from .confidence import ConfidenceEngine

            translator = DeepSeekTranslator()
            idx = TerminologyIndex.load(os.getenv("TERMINOLOGY_DIR", "data/terminology"))
            memory = ConversationMemory()
            emerg = EmergencyClassifier()
            confidence_engine = ConfidenceEngine()
            self._emit("status",
                       text=f"STT={'Deepgram live' if self._dg_key else 'Whisper'} · "
                            f"LLM={translator.model} (streaming)")
        except Exception as e:  # noqa - fallas de arranque SÍ son fatales para la sesión
            self._emit("error", text=f"Inicialización: {e}")
            return

        stt_backend = "deepgram-live" if self._dg_key else "whisper"
        while not self._stop.is_set():
            try:
                text, lang, conf = self._text_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._process_text_turn(text, lang, conf, None, stt_backend, None,
                                        translator, idx, detect_lang, memory, emerg,
                                        confidence_engine, stream_tts=self.tts_on)
            except Exception as e:  # noqa - un turno roto NUNCA mata la sesión
                self._emit("error", text=f"Turno omitido: {e}")
                logger.exception("turn_failed")

    def _process_one(self, audio, dg_key, translator, idx, detect_lang, memory,
                     emerg, confidence_engine) -> None:
        """Camino por-utterance (audio -> STT REST/Whisper -> turno). Usado por los
        tests y como referencia del contrato; el runtime streaming usa _process()."""
        t_stt0 = time.perf_counter()
        text, lang, stt_conf, speaker_id, stt_backend = self._transcribe(audio, dg_key)
        stt_ms = round((time.perf_counter() - t_stt0) * 1000)
        if not text:
            return
        self._process_text_turn(text, lang, stt_conf, speaker_id, stt_backend, stt_ms,
                                translator, idx, detect_lang, memory, emerg,
                                confidence_engine, stream_tts=False)

    def _process_text_turn(self, text, lang, stt_conf, speaker_id, stt_backend, stt_ms,
                           translator, idx, detect_lang, memory, emerg,
                           confidence_engine, stream_tts: bool) -> None:
        """Núcleo de un turno: terminología -> traducción -> emergencia -> confianza
        -> eventos -> TTS -> log. `stream_tts` decide si las frases van saliendo a la
        cola de síntesis mientras el LLM aún genera (pipeline) o si se habla al final
        de forma bloqueante (_speak)."""
        from .emergency import ALERT_EN, ALERT_ES
        from .confidence import ConfidenceInputs

        lang = detect_lang(text, fallback=lang) or lang or "en"
        hits = idx.lookup(text, self.specialty)
        term_block = "\n".join(hits) if hits else "(sin términos relevantes)"
        self._emit("src", lang=lang, text=text, speaker=speaker_id)

        t0 = time.perf_counter()
        first_ms: list[int] = []
        spoken: list[str] = []
        tgt_guess = "en" if lang == "es" else "es"

        def _speak_sentence(sent: str):
            if "<UNCLEAR>" in sent:
                return
            if not first_ms:
                first_ms.append(round((time.perf_counter() - t0) * 1000))
            spoken.append(sent)
            self._tts_q.put((sent, tgt_guess))

        if stream_tts:
            splitter = _SentenceSplitter(_speak_sentence)
            tr = translator.translate(text, lang, term_block, memory.context(limit=6),
                                      on_text=splitter.feed)
            splitter.flush()
        else:
            tr = translator.translate(text, lang, term_block, memory.context(limit=6))
        lat = round((time.perf_counter() - t0) * 1000)
        out = tr.text

        em = emerg.classify(text)
        if em["is_emergency"] and "<UNCLEAR>" not in out:
            alert = ALERT_EN if tr.target_lang == "en" else ALERT_ES
            out += f"  {alert}"
            if stream_tts:
                self._tts_q.put((alert, tr.target_lang))

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
                   text=out, latency_ms=lat, first_ms=first_ms[0] if first_ms else None,
                   stt_ms=stt_ms, stt_backend=stt_backend, speaker=speaker_id,
                   confidence_score=score, route=route)
        if route == "pause_manual_approval":
            self._emit("needs_review", text=out, src_lang=lang, tgt_lang=tr.target_lang, score=score)

        tts_ms = 0
        if self.tts_on and out and "<UNCLEAR>" not in out:
            if stream_tts:
                # el streaming no entregó nada (p. ej. resultado vino del reintento):
                # habla el texto final completo
                if not spoken:
                    self._tts_q.put((out, tr.target_lang))
            else:
                tts_ms = self._speak(out, tr.target_lang)

        log_turn(logger, session_id=self._session_id, turn_id=self._turn_id,
                 source_text=text, output_text=out, lang=lang, target_lang=tr.target_lang,
                 stt_ms=stt_ms, stt_backend=stt_backend, translation_ms=lat, tts_ms=tts_ms,
                 total_ms=(stt_ms or 0) + lat + tts_ms, confidence_score=score, route=route,
                 speaker=speaker_id, is_emergency=em["is_emergency"])

    # ---------- TTS: síntesis y reproducción en hilos separados ----------
    def _tts_synth(self):
        from .tts import synthesize
        while not self._stop.is_set():
            try:
                sent, lang = self._tts_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                samples, sr = synthesize(sent, lang)
            except Exception as e:  # noqa
                self._emit("error", text=f"TTS: {e}")
                continue
            if samples.size == 0:
                continue
            while not self._stop.is_set():
                try:
                    self._play_q.put((samples, sr), timeout=0.3)
                    break
                except queue.Full:
                    continue

    def _tts_play(self):
        import sounddevice as sd
        while not self._stop.is_set():
            try:
                samples, sr = self._play_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._speaking.set()
                sd.play(samples, sr, device=self.output_index)
                sd.wait()
            except Exception as e:  # noqa
                self._emit("error", text=f"Audio out: {e}")
            finally:
                # re-abre la escucha solo cuando no queda nada por reproducir
                if self._play_q.empty() and self._tts_q.empty():
                    time.sleep(0.15)      # deja morir el eco antes de re-escuchar
                    self._speaking.clear()

    def _speak(self, text: str, lang: str) -> int:
        """Camino bloqueante (no-pipeline). Devuelve la latencia de síntesis (ms)
        para instrumentación; 0 si falló."""
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
