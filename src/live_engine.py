"""Motor del intérprete en vivo. Usado por la GUI y por el CLI.

Captura audio (loopback del sistema o micrófono) -> segmenta por energía ->
STT (Deepgram si hay key, si no Whisper) -> DeepSeek traduce -> emite evento ->
TTS (edge-tts) reproduce en el dispositivo de salida elegido.

Half-duplex: mientras el bot habla, descarta la captura para no oírse a sí mismo.
"""
from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np

SR = 16000
FRAME = 1600
SILENCE_HANG = 0.7
MIN_SPEECH = 0.6
MAX_UTTER = 15.0


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

    # ---------- ciclo de vida ----------
    def start(self):
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

    # ---------- procesamiento ----------
    def _process(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from .translate import DeepSeekTranslator
            from .terminology import TerminologyIndex
            from .segmentation import detect_lang
            from .memory import ConversationMemory
            from .emergency import EmergencyClassifier, ALERT_EN

            dg_key = os.getenv("DEEPGRAM_API_KEY")
            whisper = None
            if not dg_key:
                from faster_whisper import WhisperModel
                whisper = WhisperModel(self.model, device="cpu", compute_type="int8")
            translator = DeepSeekTranslator()
            idx = TerminologyIndex.load(os.getenv("TERMINOLOGY_DIR", "data/terminology"))
            memory = ConversationMemory()
            emerg = EmergencyClassifier()
            self._emit("status", text=f"STT={'Deepgram' if dg_key else 'Whisper'} · LLM={translator.model}")

            while not self._stop.is_set():
                try:
                    audio = self._q.get(timeout=0.3)
                except queue.Empty:
                    continue
                # STT
                if dg_key:
                    from .stt import transcribe_np_deepgram
                    text, lang, _ = transcribe_np_deepgram(audio, SR, dg_key,
                                                           os.getenv("DEEPGRAM_MODEL", "nova-2-medical"))
                else:
                    segs, info = whisper.transcribe(audio, vad_filter=True, beam_size=1)
                    text = " ".join(s.text.strip() for s in segs).strip()
                    lang = info.language
                if not text:
                    continue
                lang = detect_lang(text, fallback=lang) or lang or "en"
                hits = idx.lookup(text, self.specialty)
                term_block = "\n".join(hits) if hits else "(sin términos relevantes)"
                self._emit("src", lang=lang, text=text)

                t0 = time.perf_counter()
                tr = translator.translate(text, lang, term_block, memory.context())
                lat = round((time.perf_counter() - t0) * 1000)
                out = tr.text
                if emerg.classify(text)["is_emergency"] and tr.target_lang == "en" and "<UNCLEAR>" not in out:
                    out += f"  {ALERT_EN}"
                if not tr.needs_clarification:
                    memory.add_turn(None, lang, text, tr.text)
                self._emit("translation", src_lang=lang, tgt_lang=tr.target_lang,
                           text=out, latency_ms=lat)

                if self.tts_on and out and "<UNCLEAR>" not in out:
                    self._speak(out, tr.target_lang)
        except Exception as e:  # noqa
            self._emit("error", text=f"Proceso: {e}")

    def _speak(self, text: str, lang: str):
        try:
            import sounddevice as sd
            from .tts import synthesize

            samples, sr = synthesize(text, lang)
            if samples.size == 0:
                return
            self._speaking.set()
            sd.play(samples, sr, device=self.output_index)
            sd.wait()
        except Exception as e:  # noqa
            self._emit("error", text=f"TTS: {e}")
        finally:
            time.sleep(0.15)          # deja morir el eco antes de re-escuchar
            self._speaking.clear()
