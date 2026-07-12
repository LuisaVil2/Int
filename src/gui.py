"""GUI del bot intérprete médico en vivo.

    python -m src.gui

Elegís ENTRADA (de dónde oye: audio del sistema/loopback o micrófono) y SALIDA
(por dónde habla la voz), activás el bot, y reproducís el video de YouTube.
"""
from __future__ import annotations

import queue
import tkinter as tk
from tkinter import ttk, scrolledtext

from .live_engine import LiveEngine, list_inputs, list_outputs


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.engine: LiveEngine | None = None
        self.events: queue.Queue = queue.Queue()
        root.title("Intérprete Médico EN↔ES — En vivo")
        root.geometry("820x560")

        self.inputs = list_inputs()
        self.outputs = list_outputs()

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Entra (oye):").grid(row=0, column=0, sticky="w", pady=3)
        self.in_var = tk.StringVar()
        self.in_combo = ttk.Combobox(top, textvariable=self.in_var, width=58, state="readonly",
                                     values=[d["label"] for d in self.inputs])
        self.in_combo.grid(row=0, column=1, sticky="w", padx=6)
        # default: primer loopback
        for i, d in enumerate(self.inputs):
            if d["kind"] == "loopback":
                self.in_combo.current(i); break

        ttk.Label(top, text="Sale (voz):").grid(row=1, column=0, sticky="w", pady=3)
        self.out_var = tk.StringVar()
        self.out_combo = ttk.Combobox(top, textvariable=self.out_var, width=58, state="readonly",
                                      values=[d["label"] for d in self.outputs])
        self.out_combo.grid(row=1, column=1, sticky="w", padx=6)
        if self.outputs:
            self.out_combo.current(0)

        opts = ttk.Frame(top)
        opts.grid(row=2, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(opts, text="Especialidad:").pack(side="left")
        self.spec_var = tk.StringVar(value="general")
        ttk.Combobox(opts, textvariable=self.spec_var, width=14, state="readonly",
                     values=self._specialties()).pack(side="left", padx=(4, 14))
        ttk.Label(opts, text="Whisper (sin key Deepgram):").pack(side="left")
        self.model_var = tk.StringVar(value="small")
        ttk.Combobox(opts, textvariable=self.model_var, width=8, state="readonly",
                     values=["tiny", "base", "small", "medium"]).pack(side="left", padx=4)

        self.tts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Hablar la traducción (voz)", variable=self.tts_var)\
            .grid(row=3, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(top, text="⚠ Si la voz sale por el MISMO dispositivo que el bot oye, puede "
                            "oírse a sí mismo.\nUsa salidas distintas (ej: oye el sistema, habla por audífonos).",
                  foreground="#a05000").grid(row=4, column=1, sticky="w", padx=6)

        btns = ttk.Frame(root, padding=(10, 0))
        btns.pack(fill="x")
        self.start_btn = ttk.Button(btns, text="▶ Activar bot", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="■ Apagar", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.status = ttk.Label(btns, text="Apagado", foreground="#555")
        self.status.pack(side="left", padx=12)

        self.log = scrolledtext.ScrolledText(root, wrap="word", font=("Consolas", 11))
        self.log.pack(fill="both", expand=True, padx=10, pady=10)
        self.log.tag_config("src", foreground="#666")
        self.log.tag_config("tr", foreground="#0a7d00", font=("Consolas", 12, "bold"))
        self.log.tag_config("st", foreground="#1565c0")
        self.log.tag_config("err", foreground="#c62828")

        self.root.after(100, self._drain)

    @staticmethod
    def _specialties() -> list[str]:
        """Especialidades presentes en los glosarios (más 'general')."""
        import os
        try:
            from .terminology import TerminologyIndex
            idx = TerminologyIndex.load(os.getenv("TERMINOLOGY_DIR", "data/terminology"))
            specs = {t.specialty for t in idx.terms} - {"drug"}
        except Exception:  # noqa
            specs = set()
        return sorted(specs | {"general"})

    # eventos del engine (en hilos) -> cola; la GUI los drena en el hilo principal
    def _on_event(self, kind: str, data: dict):
        self.events.put((kind, data))

    def _drain(self):
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "src":
                self._append(f"[{data['lang'].upper()}] {data['text']}\n", "src")
            elif kind == "translation":
                arrow = f"{data['src_lang'].upper()}→{data['tgt_lang'].upper()}"
                lat = f"{data['latency_ms']}ms"
                if data.get("first_ms"):
                    lat = f"1ª frase {data['first_ms']}ms · total {lat}"
                self._append(f"  ➜ [{arrow}] {data['text']}   ({lat})\n\n", "tr")
            elif kind == "status":
                self.status.config(text=data["text"]); self._append(f"· {data['text']}\n", "st")
            elif kind == "error":
                self._append(f"✖ {data['text']}\n", "err")
        self.root.after(100, self._drain)

    def _append(self, text: str, tag: str):
        self.log.insert("end", text, tag)
        self.log.see("end")

    def start(self):
        in_choice = self.inputs[self.in_combo.current()]
        out_index = self.outputs[self.out_combo.current()]["index"] if self.outputs else None
        self.engine = LiveEngine(self._on_event, in_choice, out_index,
                                 tts_on=self.tts_var.get(), model=self.model_var.get(),
                                 specialty=self.spec_var.get())
        self.engine.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.in_combo.config(state="disabled"); self.out_combo.config(state="disabled")
        self._append("=== BOT ACTIVO — reproduce el video de YouTube ===\n", "st")

    def stop(self):
        if self.engine:
            self.engine.stop(); self.engine = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.in_combo.config(state="readonly"); self.out_combo.config(state="readonly")
        self.status.config(text="Apagado")
        self._append("=== BOT APAGADO ===\n\n", "st")


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
