"""Tk desktop reviewer for saved fictional interpretation sessions."""
from __future__ import annotations

import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .export import build_export, write_json
from .schemas import CATEGORIES, HumanError, Review
from .store import ReviewStore

LABELS = {
    "omission": "Omission", "addition": "Addition", "mistranslation": "Mistranslation",
    "negation_reversal": "Negation reversal", "medication_or_dose": "Medication or dose",
    "numbers_or_units": "Numbers or units", "speaker_attribution": "Speaker attribution",
    "wrong_direction": "Wrong direction", "unjustified_inference": "Unjustified inference",
    "uncertainty_or_failure": "Uncertainty or failure",
}


class Reviewer:
    def __init__(self, root: tk.Tk, store: ReviewStore):
        self.root, self.store = root, store
        self.sessions = []
        self.session = None
        self.turn = None
        self.review = None

        root.title("Interpretation QA · Fictional simulations")
        root.geometry("1160x800")
        root.minsize(900, 640)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        bar = ttk.Frame(root, padding=(12, 10))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="Session").grid(row=0, column=0, padx=(0, 8))
        self.session_picker = ttk.Combobox(bar, state="readonly")
        self.session_picker.grid(row=0, column=1, sticky="ew")
        self.session_picker.bind("<<ComboboxSelected>>", self.select_session)
        ttk.Button(bar, text="Reload", command=self.reload).grid(row=0, column=2, padx=8)
        ttk.Button(bar, text="Export session", command=self.export).grid(row=0, column=3)
        panes = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        panes.grid(row=1, column=0, sticky="nsew", padx=12)
        left = ttk.Frame(panes)
        self.turn_list = tk.Listbox(left, width=29, exportselection=False)
        self.turn_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(left, command=self.turn_list.yview)
        scroll.pack(side="right", fill="y")
        self.turn_list.configure(yscrollcommand=scroll.set)
        self.turn_list.bind("<<ListboxSelect>>", self.select_turn)
        panes.add(left, weight=1)
        body = ttk.Frame(panes, padding=(12, 0, 0, 0))
        panes.add(body, weight=5)
        body.columnconfigure((0, 1), weight=1, uniform="halves")
        body.rowconfigure(2, weight=1)

        self.context = ttk.Label(body, text="Select a session and turn.")
        self.context.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.source = self.text_panel(body, "Source", 1, 0, readonly=True)
        self.candidate = self.text_panel(body, "Engine candidate", 1, 1, readonly=True)
        self.reference = self.text_panel(body, "Reference", 2, 0, readonly=True)
        self.evidence = self.text_panel(body, "Automated flags and model suggestions", 2, 1, readonly=True)
        audio = ttk.Frame(body)
        audio.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        self.audio_buttons: dict[str, ttk.Button] = {}
        for kind in ("source", "output"):
            button = ttk.Button(audio, text=f"Play {kind}", command=lambda k=kind: self.play(k))
            button.pack(side="left", padx=(0, 8))
            self.audio_buttons[kind] = button

        labels = ttk.LabelFrame(body, text="Human error labels and severity", padding=8)
        labels.grid(row=4, column=0, columnspan=2, sticky="ew")
        labels.columnconfigure((0, 1), weight=1, uniform="labels")
        self.errors = {}
        for index, category in enumerate(CATEGORIES):
            cell = ttk.Frame(labels)
            cell.grid(row=index % 5, column=index // 5, sticky="ew", padx=4, pady=2)
            ttk.Label(cell, text=LABELS[category]).pack(side="left")
            value = tk.StringVar()
            self.errors[category] = value
            ttk.Combobox(cell, textvariable=value, values=("", "minor", "major", "critical"),
                         state="readonly", width=10).pack(side="right")
        self.correction = self.text_panel(body, "Corrected interpretation", 5, 0)
        self.notes = self.text_panel(body, "Reviewer notes", 5, 1)
        actions = ttk.Frame(body)
        actions.grid(row=6, column=0, columnspan=2, sticky="ew", pady=8)
        self.draft_button = ttk.Button(actions, text="Save draft", command=lambda: self.save("pending"))
        self.draft_button.pack(side="left", padx=(0, 8))
        self.done_button = ttk.Button(actions, text="Mark reviewed", command=lambda: self.save("reviewed"))
        self.done_button.pack(side="left")
        self.status = ttk.Label(root, padding=(12, 5))
        self.status.grid(row=2, column=0, sticky="ew")
        root.protocol("WM_DELETE_WINDOW", lambda: root.destroy() if self.can_leave() else None)
        self.reload()

    @staticmethod
    def text_panel(parent, label, row, column, readonly=False):
        pane = ttk.LabelFrame(parent, text=label, padding=5)
        pane.grid(row=row, column=column, sticky="nsew", padx=(0, 6), pady=3)
        box = ScrolledText(pane, height=4, width=36, wrap="word", undo=not readonly)
        box.pack(fill="both", expand=True)
        if readonly:
            box.configure(state="disabled")
        return box

    @staticmethod
    def fill(box, text, readonly=False):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        if readonly:
            box.configure(state="disabled")

    def can_leave(self):
        if self.turn is None:
            return True
        before = self.review
        correction = self.correction.get("1.0", "end-1c")
        notes = self.notes.get("1.0", "end-1c")
        errors = {key: value.get() for key, value in self.errors.items() if value.get()}
        saved = {error.category: error.severity for error in before.errors} if before else {}
        changed = (correction != (before.corrected_interpretation if before else "")
                   or notes != (before.notes if before else "") or errors != saved)
        return not changed or messagebox.askyesno("Unsaved edits", "Discard unsaved review edits?",
                                                  parent=self.root)

    def reload(self):
        if not self.can_leave():
            return
        current_id = self.session.id if self.session else None
        self.sessions = self.store.sessions()
        self.session_picker["values"] = [
            f"{s.title} · {s.adapter} · {str(s.id)[:8]}" for s in self.sessions]
        if self.sessions:
            selected = next((i for i, s in enumerate(self.sessions) if s.id == current_id), 0)
            self.session_picker.current(selected)
            self.load_session(self.sessions[selected])
        else:
            self.show_turn(None)
            self.status.configure(text="No sessions yet. Run a fictional scenario from the command line.")

    def select_session(self, _event=None):
        selected = self.session_picker.current()
        if selected < 0:
            return
        if not self.can_leave():
            self.session_picker.current(self.sessions.index(self.session))
            return
        self.load_session(self.sessions[selected])

    def load_session(self, session):
        self.session = session
        self.saved = self.store.reviews(session.id)
        self.refresh_turn_list(0)
        self.show_turn(session.turns[0])

    def refresh_turn_list(self, selected):
        self.turn_list.delete(0, "end")
        for turn in self.session.turns:
            review = self.saved.get(turn.id)
            status = "demo" if review and review.origin == "demo" else (
                "reviewed" if review and review.status == "reviewed" else "pending")
            self.turn_list.insert("end", f"{turn.id}  {turn.role}  ·  {status}")
        self.turn_list.selection_set(selected)
        self.turn_list.see(selected)

    def select_turn(self, _event=None):
        selection = self.turn_list.curselection()
        if not selection or self.session is None:
            return
        selected = selection[0]
        if self.session.turns[selected] is self.turn:
            return
        if not self.can_leave():
            self.refresh_turn_list(self.session.turns.index(self.turn))
            return
        self.show_turn(self.session.turns[selected])

    def show_turn(self, turn):
        self.turn = turn
        self.review = self.saved.get(turn.id) if turn else None
        for button in (self.draft_button, self.done_button):
            button.configure(state="normal" if turn else "disabled")
        for kind, button in self.audio_buttons.items():
            enabled = bool(turn and self.store.audio_path(self.session.id, turn.id, kind).is_file())
            button.configure(state="normal" if enabled else "disabled")
        if turn is None:
            return
        self.context.configure(text=f"{turn.speaker_id} · {turn.role} · "
                                    f"{turn.source_language.upper()} → {turn.target_language.upper()} · "
                                    f"{turn.event.replace('_', ' ')} · {turn.disposition}")
        candidate = "\n\n".join(
            f"[{(item.target_language or '?').upper()}"
            f"{', ' + item.role if item.role else ''}] {item.text}" for item in turn.interpretations)
        flags = [f"Rule · {LABELS[flag.category]} ({flag.suggested_severity}): "
                 f"{flag.rule.replace('_', ' ')}" for flag in turn.flags]
        flags.extend(f"Model suggestion · {LABELS[item.category]} ({item.suggested_severity}): "
                     f"{item.rationale}" for item in turn.model_suggestions)
        if turn.review_reasons:
            flags.append("Engine hold · " + ", ".join(reason.replace("_", " ")
                                                      for reason in turn.review_reasons))
        self.fill(self.source, turn.source, readonly=True)
        self.fill(self.candidate, candidate or "No candidate returned.", readonly=True)
        self.fill(self.reference, turn.reference or "Expected disposition: review.", readonly=True)
        self.fill(self.evidence, "\n".join(flags) or "No automated flags or suggestions.", readonly=True)
        self.fill(self.correction, self.review.corrected_interpretation if self.review else "")
        self.fill(self.notes, self.review.notes if self.review else "")
        saved = {error.category: error.severity for error in self.review.errors} if self.review else {}
        for category, field in self.errors.items():
            field.set(saved.get(category, ""))
        status = self.review.status if self.review else "pending"
        if self.review and self.review.origin == "demo":
            status = "simulated review"
        self.status.configure(text=f"Review status · {status} · {self.session.adapter} adapter")

    def save(self, status):
        if self.turn is None:
            return
        try:
            review = self.store.save_review(Review(
                session_id=self.session.id, turn_id=self.turn.id, status=status,
                revision=self.review.revision if self.review else 0,
                errors=tuple(HumanError(category=key, severity=value.get())
                             for key, value in self.errors.items() if value.get()),
                corrected_interpretation=self.correction.get("1.0", "end-1c"),
                notes=self.notes.get("1.0", "end-1c"),
            ))
        except ValueError as error:
            # Validation errors can contain free-text input.
            text = str(error) if type(error) is ValueError else "Review text or labels are invalid."
            messagebox.showerror("Could not save review", text, parent=self.root)
            return
        except sqlite3.Error:
            messagebox.showerror("Could not save review", "The local database could not be updated.",
                                 parent=self.root)
            return
        self.saved[review.turn_id] = review
        self.refresh_turn_list(self.session.turns.index(self.turn))
        self.show_turn(self.turn)

    def play(self, kind):
        if self.turn is None:
            return
        path = self.store.audio_path(self.session.id, self.turn.id, kind)
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except (ImportError, RuntimeError, OSError):
            messagebox.showerror("Could not play audio", "WAV playback requires Windows and an audio device.",
                                 parent=self.root)

    def export(self):
        if self.session is None:
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json",
                                            filetypes=[("JSON", "*.json")], confirmoverwrite=False)
        if not path:
            return
        try:
            write_json(Path(path), build_export(self.store.snapshot(self.session.id)))
        except OSError:
            messagebox.showerror("Could not export", "Choose a new, writable JSON filename.", parent=self.root)
            return
        self.status.configure(text="Structured export saved. Reviewer free text stays in the local database.")


def open_reviewer(store: ReviewStore) -> None:
    root = tk.Tk()
    Reviewer(root, store)
    root.mainloop()
