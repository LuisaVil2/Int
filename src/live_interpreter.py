"""Bot intérprete EN<->ES EN VIVO. Oye el audio del sistema (el video de YouTube) y
traduce en tiempo real a texto en pantalla.

Uso:
    # 1) Enciende el bot:
    python -m src.live_interpreter
    # 2) Reproduce el video de YouTube en el navegador. El bot lo oye por loopback.

    python -m src.live_interpreter --mic            # usa micrófono en vez de loopback
    python -m src.live_interpreter --list-devices   # lista dispositivos de audio
    python -m src.live_interpreter --model small     # tamaño Whisper (tiny/base/small/medium)

Requiere DEEPSEEK_API_KEY en .env. STT = Whisper local (lag ~3-5s). Para baja latencia
real, migrar a Deepgram streaming (ver src/stt.py ruta Deepgram).

Ctrl+C para apagar.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import numpy as np
from dotenv import load_dotenv

SR = 16000                 # sample rate
FRAME = 1600               # 0.1s por bloque de captura
SILENCE_HANG = 0.7         # s de silencio para cerrar un turno
MIN_SPEECH = 0.6           # s mínimos de voz para procesar
MAX_UTTER = 15.0           # s máx por turno (corta monólogos largos)


def list_devices():
    import soundcard as sc
    print("== Parlantes (loopback) ==")
    for s in sc.all_speakers():
        print(f"  {s.name}")
    print("== Micrófonos ==")
    for m in sc.all_microphones():
        print(f"  {m.name}")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def capture_loop(out_q: "queue.Queue", use_mic: bool, stop: threading.Event):
    """Productor: captura audio, segmenta por energía, encola turnos (np.float32 mono 16k)."""
    import soundcard as sc

    if use_mic:
        dev = sc.default_microphone()
        src = "micrófono"
    else:
        spk = sc.default_speaker()
        dev = sc.get_microphone(id=str(spk.name), include_loopback=True)
        src = f"loopback ({spk.name})"
    print(f"[audio] Capturando de: {src}\n")

    with dev.recorder(samplerate=SR, channels=1, blocksize=FRAME) as rec:
        # calibra piso de ruido ~1s
        noise = np.median([_rms(rec.record(numframes=FRAME).flatten()) for _ in range(10)])
        thresh = max(0.006, noise * 3.0)

        buf: list[np.ndarray] = []
        silence = 0.0
        speaking = False
        while not stop.is_set():
            chunk = rec.record(numframes=FRAME).flatten().astype(np.float32)
            level = _rms(chunk)
            if level >= thresh:
                buf.append(chunk)
                silence = 0.0
                speaking = True
            elif speaking:
                buf.append(chunk)               # cola de silencio dentro del turno
                silence += FRAME / SR
                dur = len(buf) * FRAME / SR
                if silence >= SILENCE_HANG and dur >= MIN_SPEECH:
                    out_q.put(np.concatenate(buf)); buf, silence, speaking = [], 0.0, False
            # corta turnos demasiado largos
            if speaking and len(buf) * FRAME / SR >= MAX_UTTER:
                out_q.put(np.concatenate(buf)); buf, silence, speaking = [], 0.0, False


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true", help="usar micrófono en vez de loopback")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--model", default="small", help="tamaño Whisper")
    ap.add_argument("--specialty", default="general")
    args = ap.parse_args()

    if args.list_devices:
        list_devices(); return

    # carga perezosa (tras --list-devices)
    from faster_whisper import WhisperModel
    from .translate import DeepSeekTranslator
    from .terminology import TerminologyIndex
    from .segmentation import detect_lang
    from .memory import ConversationMemory
    from .emergency import EmergencyClassifier, ALERT_EN, ALERT_ES
    from .logging_utils import configure_logging, log_turn, new_session_id
    import logging

    configure_logging()
    logger = logging.getLogger(__name__)
    session_id = new_session_id()
    turn_id = 0

    print("== Cargando intérprete en vivo ==")
    whisper = WhisperModel(args.model, device="cpu", compute_type="int8")
    translator = DeepSeekTranslator()
    idx = TerminologyIndex.load("data/terminology")
    memory = ConversationMemory()
    emerg = EmergencyClassifier()
    print(f"  Whisper={args.model}  DeepSeek={translator.model}  términos={len(idx.terms)}")

    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    cap = threading.Thread(target=capture_loop, args=(out_q, args.mic, stop), daemon=True)
    cap.start()

    print("== BOT ACTIVO ==  Reproduce el video de YouTube. Ctrl+C para apagar.\n")
    try:
        while True:
            audio = out_q.get()
            try:
                t_stt0 = time.perf_counter()
                segs, info = whisper.transcribe(audio, vad_filter=True, beam_size=1)
                text = " ".join(s.text.strip() for s in segs).strip()
                stt_ms = round((time.perf_counter() - t_stt0) * 1000)
                if not text:
                    continue
                lang = detect_lang(text) or info.language or "en"
                hits = idx.lookup(text, args.specialty)
                term_block = "\n".join(hits) if hits else "(sin términos relevantes)"
                t0 = time.perf_counter()
                tr = translator.translate(text, lang, term_block, memory.context(limit=6))
                lat = round((time.perf_counter() - t0) * 1000)
                out_text = tr.text
                em = emerg.classify(text)
                if em["is_emergency"] and "<UNCLEAR>" not in out_text:
                    out_text += f"  {ALERT_EN if tr.target_lang == 'en' else ALERT_ES}"
                if not tr.needs_clarification:
                    memory.add_turn(None, lang, text, tr.text)

                turn_id += 1
                log_turn(logger, session_id=session_id, turn_id=turn_id,
                         source_text=text, output_text=out_text, lang=lang,
                         target_lang=tr.target_lang, stt_ms=stt_ms, translation_ms=lat,
                         is_emergency=em["is_emergency"])

                arrow = f"{lang.upper()}→{tr.target_lang.upper()}"
                print(f"  [{arrow}] {text}")
                print(f"     ➜ {out_text}   ({lat}ms)\n")
            except Exception as e:  # noqa - un turno roto no debe matar la sesión
                print(f"  ✗ Turno omitido: {e}")
                logger.exception("turn_failed")
    except KeyboardInterrupt:
        stop.set()
        print("\n== BOT APAGADO ==")


if __name__ == "__main__":
    main()
