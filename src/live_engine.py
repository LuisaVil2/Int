"""Motor del intérprete en vivo. Usado por la GUI y por el CLI.

Arquitectura pipeline (etapas desacopladas por colas): mientras el bot habla el
turno N ya está transcribiendo y traduciendo el N+1.

  captura ─frames→ STT ─texto→ proceso (terminología + DeepSeek) ─frases→ síntesis TTS ─audio→ reproducción

Streaming de punta a punta:
- STT: Deepgram live WebSocket (endpointing en el servidor) si hay key;
  si no, VAD local por energía + Whisper (como antes).
- LLM: la traducción se streamea; cada frase completa entra a TTS sin esperar
  el JSON entero (ver TextFieldStream en translate.py).
- TTS: la síntesis de la frase siguiente se solapa con la reproducción de la actual.

Half-duplex: mientras reproduce voz, la captura manda silencio (no se oye a sí mismo).
"""
from __future__ import annotations

import os
import queue
import re
import threading
import time

import numpy as np

SR = 16000
FRAME = 1600
SILENCE_HANG = 0.7
MIN_SPEECH = 0.6
MAX_UTTER = 15.0
MIN_SENTENCE = 12          # chars mínimos antes de mandar una frase a TTS


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

    # ---------- ciclo de vida ----------
    def start(self):
        from dotenv import load_dotenv
        load_dotenv()
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

    # ---------- STT local (solo sin key Deepgram) ----------
    def _stt_whisper(self):
        try:
            from faster_whisper import WhisperModel
            whisper = WhisperModel(self.model, device="cpu", compute_type="int8")
            while not self._stop.is_set():
                try:
                    audio = self._utter_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                segs, info = whisper.transcribe(audio, vad_filter=True, beam_size=1)
                text = " ".join(s.text.strip() for s in segs).strip()
                if text:
                    self._text_q.put((text, info.language, 0.85))
        except Exception as e:  # noqa
            self._emit("error", text=f"Whisper: {e}")

    # ---------- traducción ----------
    def _process(self):
        try:
            from .translate import DeepSeekTranslator
            from .terminology import TerminologyIndex
            from .segmentation import detect_lang
            from .memory import ConversationMemory
            from .emergency import EmergencyClassifier, ALERT_EN

            translator = DeepSeekTranslator()
            idx = TerminologyIndex.load(os.getenv("TERMINOLOGY_DIR", "data/terminology"))
            memory = ConversationMemory()
            emerg = EmergencyClassifier()
            self._emit("status",
                       text=f"STT={'Deepgram live' if self._dg_key else 'Whisper'} · "
                            f"LLM={translator.model} (streaming)")

            while not self._stop.is_set():
                try:
                    text, lang, _ = self._text_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                lang = detect_lang(text, fallback=lang) or lang or "en"
                hits = idx.lookup(text, self.specialty)
                term_block = "\n".join(hits) if hits else "(sin términos relevantes)"
                self._emit("src", lang=lang, text=text)

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

                splitter = _SentenceSplitter(_speak_sentence)
                tr = translator.translate(text, lang, term_block, memory.context(),
                                          on_text=splitter.feed if self.tts_on else None)
                if self.tts_on:
                    splitter.flush()
                lat = round((time.perf_counter() - t0) * 1000)
                out = tr.text
                if emerg.classify(text)["is_emergency"] and tr.target_lang == "en" \
                        and "<UNCLEAR>" not in out:
                    out += f"  {ALERT_EN}"
                    if self.tts_on:
                        self._tts_q.put((ALERT_EN, "en"))
                if not tr.needs_clarification:
                    memory.add_turn(None, lang, text, tr.text)
                self._emit("translation", src_lang=lang, tgt_lang=tr.target_lang,
                           text=out, latency_ms=lat,
                           first_ms=first_ms[0] if first_ms else None)

                # el streaming no entregó nada (p. ej. resultado vino del reintento):
                # habla el texto final completo
                if self.tts_on and not spoken and "<UNCLEAR>" not in out:
                    self._tts_q.put((out, tr.target_lang))
        except Exception as e:  # noqa
            self._emit("error", text=f"Proceso: {e}")

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
