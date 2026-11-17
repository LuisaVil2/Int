"""Motor compartido por GUI y CLI, con colas FIFO por etapa y revisión antes de TTS."""
from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time

import numpy as np

from .logging_utils import configure_logging, log_turn, new_session_id
from .stt import Segment

SR = 16000
FRAME = 1600
SILENCE_HANG = 0.7
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
                 tts_on: bool = True, model: str = "small", specialty: str = "general", *, trace=None):
        self.on_event = on_event
        self.input_choice = input_choice
        self.output_index = output_index
        self.tts_on = tts_on
        self.model = model
        self.specialty = specialty
        self.trace = trace
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._playback_lock = threading.Lock()
        self._playback_epoch = 0
        self._started = False
        self._utter_q: queue.Queue = queue.Queue(maxsize=8)  # np audio -> Whisper
        self._text_q: queue.Queue = queue.Queue(maxsize=16)  # texto STT -> proceso
        self._tts_q: queue.Queue = queue.Queue(maxsize=8)  # turno validado -> síntesis
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
        if self._started or self._stop.is_set():
            raise RuntimeError("Cree una sesión nueva para reiniciar el motor")
        self._started = True
        self._dg_key = os.getenv("DEEPGRAM_API_KEY") or None
        targets = [self._process, self._capture]
        if self.tts_on:
            targets.extend((self._tts_synth, self._tts_play))
        if not self._dg_key:
            targets.append(self._stt_whisper)
        self._threads = [threading.Thread(target=t, daemon=True) for t in targets]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        if self.tts_on:
            with self._playback_lock:
                try:
                    import sounddevice as sd
                    sd.stop()
                except Exception:
                    logger.warning("audio_stop_failed")
        if self._dg is not None:
            self._dg.close()
        deadline = time.monotonic() + 2
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=max(0, deadline - time.monotonic()))
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        if self._threads:
            logger.warning("workers_still_stopping", extra={"meta": {"count": len(self._threads)}})

    def _emit(self, kind: str, **data):
        if not self._stop.is_set():
            if kind in ("translation", "needs_review"):
                self._trace("qa_decision", data.get("turn_id"))
            try:
                self.on_event(kind, data)
            except Exception:
                logger.warning("event_callback_failed")

    def _trace(self, stage: str, turn_id=None):
        if self.trace:
            try:
                self.trace(stage, time.perf_counter_ns(), turn_id)
            except Exception:
                logger.warning("trace_callback_failed")

    def _review(self, reason: str, **data) -> None:
        self._emit("needs_review", reason=reason, route="pause_manual_approval", **data)

    def _enqueue(self, work_queue, payload, overflow_reason: str) -> None:
        if not self._stop.is_set():
            try:
                work_queue.put_nowait(payload)
            except queue.Full:
                self._review(overflow_reason)

    def _on_live_text(self, text, lang, confidence):
        self._trace("stt_finalized")
        self._enqueue(self._text_q, (text, lang, confidence), "transcript_queue_full")

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
                    on_utterance=self._on_live_text,
                    on_gap=self._review,
                    on_error=lambda _: self._emit("error", text="Deepgram desconectado."),

                )
                self._dg.start()
            with dev.recorder(samplerate=SR, channels=1, blocksize=FRAME) as rec:
                if self._dg:
                    self._capture_streaming(rec)
                else:
                    self._capture_vad(rec)
        except Exception:
            self._review("capture_failed")
            self._emit("error", text="Captura detenida.")
        finally:
            if self._dg:
                self._dg.close()

    def _read_frame(self, rec):
        before = self._playback_epoch
        muted = self._speaking.is_set()
        chunk = rec.record(FRAME).flatten().astype(np.float32)
        self._trace("capture_frame")
        return chunk, muted or self._speaking.is_set() or before != self._playback_epoch

    def _capture_streaming(self, rec):
        """Modo Deepgram: frames continuos al WebSocket; el endpointing lo hace el server."""
        self._emit("status", text="Bot activo. Escuchando… (Deepgram streaming)")
        zeros = np.zeros(FRAME, dtype=np.float32)
        while not self._stop.is_set() and not self._dg.closed:
            chunk, muted = self._read_frame(rec)
            if self._stop.is_set():
                break
            self._dg.set_capture_muted(muted)
            self._dg.send_np(zeros if muted else chunk)

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
        except Exception:
            self._emit("status", text="Calibrando VAD por energía. Espere el aviso de escucha.")
            noise = np.median([self._rms(rec.record(FRAME).flatten()) for _ in range(10)])
            level_thresh = max(0.006, noise * 3.0)
            return lambda chunk: self._rms(chunk) >= level_thresh

    def _capture_vad(self, rec):
        """Modo local: segmenta con VAD y encola utterances para Whisper."""
        is_speech = self._make_vad(rec)
        self._emit("status", text="Bot activo. Escuchando…")
        buf, silence, speaking = [], 0.0, False
        while not self._stop.is_set():
            chunk, muted = self._read_frame(rec)
            if self._stop.is_set():
                break
            if muted:
                if buf:
                    self._review("vad_turn_interrupted")
                buf, silence, speaking = [], 0.0, False
                continue
            if is_speech(chunk):
                buf.append(chunk); silence = 0.0; speaking = True
            elif speaking:
                buf.append(chunk); silence += FRAME / SR
                if silence >= SILENCE_HANG:
                    self._enqueue(self._utter_q, np.concatenate(buf), "audio_queue_full")
                    buf, silence, speaking = [], 0.0, False
            if speaking and len(buf) * FRAME / SR >= MAX_UTTER:
                self._review("vad_turn_too_long")
                # Esperar el silencio antes de reanudar para evitar traducir la cola.
                while not self._stop.is_set():
                    frame, muted = self._read_frame(rec)
                    if not muted and not is_speech(frame):
                        silence += FRAME / SR
                        if silence >= SILENCE_HANG:
                            break
                    else:
                        silence = 0
                buf, silence, speaking = [], 0.0, False

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
                    audio, SR, dg_key, os.getenv("DEEPGRAM_MODEL", "nova-3"))
                return text, lang, conf, speaker, "deepgram"
            except Exception:
                self._emit("status", text="Deepgram falló; Whisper para este turno.")
                logger.warning("deepgram_fallback_triggered")
        whisper = self._get_whisper()
        segs, info = whisper.transcribe(audio, vad_filter=True, beam_size=1, multilingual=True)
        segs = list(segs)
        tags = {s.language for s in segs if getattr(s, "language", None)}
        lang = "mixed" if len(tags) > 1 else next(iter(tags), info.language)
        text = " ".join(s.text.strip() for s in segs).strip()
        return text, lang, 0.85, None, "whisper"

    def _stt_whisper(self):
        """Hilo STT local (solo sin key Deepgram): consume utterances del VAD."""
        while not self._stop.is_set():
            try:
                audio = self._utter_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                text, lang, conf, _, _ = self._transcribe(audio, None)
                if text:
                    self._on_live_text(text, lang, conf)
            except Exception:
                self._review("whisper_turn_failed")

    # ---------- procesamiento ----------
    def _process(self):
        try:
            from .translate import DeepSeekTranslator
            from .terminology import TerminologyIndex
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
                            f"LLM={translator.model}")
        except Exception:
            self._emit("error", text="Inicialización del traductor fallida. Revise la configuración.")
            self._stop.set()
            return

        stt_backend = "deepgram-live" if self._dg_key else "whisper"
        while not self._stop.is_set():
            try:
                text, lang, conf = self._text_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self.process_transcript(Segment(0, text, lang, asr_confidence=conf),
                                        translator, idx, memory, emerg, confidence_engine,
                                        stt_backend=stt_backend)
            except Exception:
                self._review("turn_failed")
                logger.warning("turn_failed")

    def process_transcript(self, segment: Segment, translator, idx, memory, emerg,
                           confidence_engine, *, stt_backend: str) -> None:
        """Process one finalized transcript without starting an audio stream."""
        from .segmentation import detect_lang, normalize_turns

        for turn in normalize_turns([segment]):
            self._process_text_turn(turn.text, turn.lang, segment.asr_confidence, turn.speaker,
                                    stt_backend, None, translator, idx, detect_lang, memory,
                                    emerg, confidence_engine, stream_tts=True,
                                    direction_resolved=True)

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
                           confidence_engine, stream_tts: bool, direction_resolved=False) -> None:
        """Valida y enruta el turno completo antes de entregarlo a TTS."""
        from .clinical_checks import meaning_flags
        from .translate import fidelity_issue

        if self._stop.is_set():
            return
        self._turn_id += 1
        turn_id = self._turn_id
        if not direction_resolved:
            lang = detect_lang(text, fallback=lang)
        self._emit("src", lang=lang, text=text, speaker=speaker_id, turn_id=turn_id)
        em = emerg.classify(text)
        review_meta = {"turn_id": turn_id, "emergency_labels": em["labels"]}
        if lang not in ("en", "es"):
            self._review("unknown_source_language", **review_meta)
            return

        hits = idx.lookup(text, self.specialty)
        bilingual_lookup = getattr(idx, "lookup_bilingual", None)
        prompt_hits = bilingual_lookup(text, self.specialty, lang) if bilingual_lookup else hits
        term_block = "\n".join(prompt_hits) if prompt_hits else "(sin términos relevantes)"
        t0 = time.perf_counter()
        self._trace("translation_started", turn_id)
        try:
            tr = translator.translate(text, lang, term_block, memory.context(limit=6))
        except Exception:
            self._review("translation_failed", **review_meta)
            logger.warning("translation_failed")
            return
        if self._stop.is_set():
            return

        lat = round((time.perf_counter() - t0) * 1000)
        self._trace("translation_complete", turn_id)
        expected = "en" if lang == "es" else "es"
        if (tr.needs_clarification or tr.target_lang != expected or not tr.text.strip()
                or "<unclear>" in tr.text.casefold()):
            self._review("translation_uncertain", **review_meta)
            return
        if issue := fidelity_issue(text, tr.text):
            self._review(issue, **review_meta)
            return
        if anchored := next(iter(meaning_flags(text, tr.text)), None):
            self._review(anchored[0], **review_meta)
            return

        score = confidence_engine.score_turn(stt_conf, tr.confidence, len(hits))
        shadow_score = confidence_engine.score_turn(stt_conf, tr.confidence, len(prompt_hits))
        route = confidence_engine.route(score)
        if em["force_qa_review"] and route == "automatic_approval":
            route = "qa_review"
        self._emit("confidence_evidence", turn_id=turn_id, legacy_hits=len(hits),
                   bilingual_hits=len(prompt_hits), legacy_score=score, bilingual_score=shadow_score,
                   would_accept=confidence_engine.route(shadow_score) == "automatic_approval"
                   and not em["force_qa_review"])

        if route != "automatic_approval":
            reason = "emergency_keyword" if em["force_qa_review"] else "confidence_review"
            # El borrador se muestra solo en el evento de revisión, nunca en TTS.
            self._emit("needs_review", reason=reason, route=route, draft=tr.text,
                       src_lang=lang, tgt_lang=tr.target_lang, score=score, **review_meta)
        else:
            # Los speaker IDs de REST sirven como metadato local de este turno.
            memory.add_turn(None, lang, text, tr.text)
            self._emit("translation", src_lang=lang, tgt_lang=tr.target_lang,
                       text=tr.text, latency_ms=lat, first_ms=None,
                       stt_ms=stt_ms, stt_backend=stt_backend, speaker=speaker_id,
                       confidence_score=score, route=route, turn_id=turn_id)
            if self.tts_on:
                self._enqueue(self._tts_q, (tr.text, tr.target_lang, turn_id), "tts_queue_full")

        log_turn(logger, session_id=self._session_id, turn_id=turn_id,
                 lang=lang, target_lang=tr.target_lang, translation_ms=lat,
                 stt_ms=stt_ms, stt_backend=stt_backend, confidence_score=score,
                 route=route, is_emergency=em["is_emergency"])

    # ---------- Síntesis y reproducción FIFO ----------
    def _tts_synth(self):
        try:
            from .tts import synthesize
        except Exception:
            self._review("tts_unavailable")
            return
        while not self._stop.is_set():
            try:
                text, lang, turn_id = self._tts_q.get(timeout=0.3)
            except queue.Empty:
                continue
            if self._stop.is_set():
                return
            try:
                self._trace("tts_started", turn_id)
                samples, sr = synthesize(text, lang)
                if samples.size == 0:
                    self._review("tts_empty", turn_id=turn_id)
                    continue
                self._trace("tts_synthesized", turn_id)
            except Exception:
                self._review("tts_failed", turn_id=turn_id)
                continue
            while not self._stop.is_set():
                try:
                    self._play_q.put((samples, sr, turn_id), timeout=0.3)
                    break
                except queue.Full:
                    continue

    def _tts_play(self):
        try:
            import sounddevice as sd
        except ImportError:
            self._review("audio_output_unavailable")
            return
        while not self._stop.is_set():
            try:
                samples, sr, turn_id = self._play_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                with self._playback_lock:
                    if self._stop.is_set():
                        return
                    self._playback_epoch += 1
                    self._speaking.set()
                    sd.play(samples, sr, device=self.output_index)
                self._trace("playback_started", turn_id)
                self._emit("status", text="Micrófono en pausa durante la reproducción.")
                sd.wait()
                if not self._stop.is_set():
                    self._trace("playback_complete", turn_id)
            except Exception:
                self._review("audio_output_failed", turn_id=turn_id)
            finally:
                self._stop.wait(0.15)  # cola de eco
                self._speaking.clear()
                self._playback_epoch += 1
                self._emit("status", text="Escuchando. Repita si habló durante la reproducción.")
