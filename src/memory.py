"""Memoria conversacional — contexto inyectado al LLM. Salvado de LuisaVil2/Int.

Mejora la calidad: el intérprete ve los turnos previos (anáfora, terminología
consistente, hilo clínico) en vez de traducir frases aisladas.
"""
from __future__ import annotations

from collections import deque


class ConversationMemory:
    def __init__(self, max_turns: int = 12):
        self.turns: deque[dict] = deque(maxlen=max_turns)

    def add_turn(self, speaker, source_lang: str, source_text: str, interpretation: str):
        self.turns.append({"speaker": speaker, "source_lang": source_lang,
                           "source_text": source_text, "interpretation": interpretation})

    def context(self, limit: int | None = None) -> str:
        if not self.turns:
            return "(inicio de la conversación, sin contexto previo)"
        turns = list(self.turns)[-limit:] if limit else list(self.turns)
        return "\n".join(
            (f"[{t['source_lang']}] (S{t['speaker']}) {t['source_text']} -> {t['interpretation']}"
             if t.get("speaker") is not None
             else f"[{t['source_lang']}] {t['source_text']} -> {t['interpretation']}")
            for t in turns
        )
