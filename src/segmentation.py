"""Capa de normalización de turnos — FIX Hallazgo #1.

El STT (Whisper VAD, o incluso Deepgram sin diarización perfecta) mezcla el final
de una frase EN con el inicio de la respuesta ES en un mismo segmento. Eso rompe el
modelo turno-por-turno: el LLM no puede elegir una sola dirección -> <UNCLEAR>.

Solución: partir cada segmento en frases, detectar idioma POR FRASE, y reagrupar
frases consecutivas del mismo idioma en turnos monolingües. Funciona para cualquier
backend STT y es complementario a la diarización de Deepgram.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0  # determinismo

from .stt import Segment


@dataclass
class Turn:
    start: float
    lang: str          # 'en' | 'es'
    text: str
    speaker: int | None = None


# corta en límites de frase conservando el delimitador; conserva ¿ ¡ iniciales
_SENT = re.compile(r"[¿¡]?[^.!?¿¡]*[.!?]+|[¿¡]?\S[^.!?¿¡]*$")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [m.group().strip() for m in _SENT.finditer(text)]
    return [p for p in parts if p]


def detect_lang(text: str, fallback: str | None = None) -> str | None:
    """EN/ES solamente. Para fragmentos muy cortos, usa fallback (idioma del turno previo)."""
    alpha = re.sub(r"[^a-záéíóúñü ]", "", text.lower())
    if len(alpha.replace(" ", "")) < 4:
        return fallback  # "No.", "¿Cómo?" -> hereda contexto
    try:
        ranked = detect_langs(text)
    except Exception:  # noqa
        return fallback
    for cand in ranked:
        if cand.lang in ("en", "es"):
            return cand.lang
    # si detectó otro idioma (pt/it confundido con es), elige el más cercano
    top = ranked[0].lang if ranked else None
    return "es" if top in ("pt", "it", "ca", "gl") else ("en" if top else fallback)


def normalize_turns(segments: list[Segment]) -> list[Turn]:
    """Aplana segmentos -> frases monolingües -> turnos por idioma contiguo."""
    turns: list[Turn] = []
    last_lang: str | None = None
    for seg in segments:
        seg_speaker = getattr(seg, "speaker", None)
        for sent in split_sentences(seg.text):
            lang = detect_lang(sent, fallback=last_lang) or seg.lang or "en"
            last_lang = lang
            if turns and turns[-1].lang == lang and turns[-1].speaker == seg_speaker:
                # misma dirección y hablante -> fusiona con el turno abierto
                turns[-1].text = (turns[-1].text + " " + sent).strip()
            else:
                turns.append(Turn(start=seg.start, lang=lang, text=sent, speaker=seg_speaker))
    return turns
