"""Consola de interpretación de texto con el mismo motor que la GUI."""
from __future__ import annotations

import argparse
import queue
import sys

from .live_engine import LiveEngine, list_inputs


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic", action="store_true", help="usar micrófono en vez de loopback")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--model", default="small", help="tamaño Whisper")
    parser.add_argument("--specialty", default="general")
    args = parser.parse_args()

    if args.list_devices:
        for device in list_inputs():
            print(device["label"])
        return

    import soundcard as sc

    device = sc.default_microphone() if args.mic else sc.default_speaker()
    events = queue.Queue()
    engine = LiveEngine(lambda kind, data: events.put((kind, data)),
                        {"kind": "mic" if args.mic else "loopback", "name": device.name},
                        output_index=None, tts_on=False, model=args.model, specialty=args.specialty)
    engine.start()
    try:
        while True:
            kind, data = events.get()
            if kind == "src":
                print(f"[{data['lang'] or '?'}] {data['text']}")
            elif kind == "translation":
                print(f"  → [{data['tgt_lang']}] {data['text']}")
            elif kind == "needs_review":
                print(f"  Revisión requerida ({data['reason']})")
                if data.get("emergency_labels"):
                    print("  Coincidencia con términos de emergencia.")
                if data.get("draft"):
                    print(f"  Borrador: {data['draft']}")
            elif kind in ("status", "error"):
                print(data["text"])
                if kind == "error":
                    break
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
