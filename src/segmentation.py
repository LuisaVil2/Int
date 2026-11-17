from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # determinismo

from .stt import Segment


@dataclass
class Turn:
    start: float
    lang: str | None   # EN, ES o dirección incierta
    text: str
    speaker: int | None = None


# No separar decimales, abreviaturas habituales ni iniciales de nombres.
_BOUNDARY = re.compile(r"[.!?…]+(?=\s|[¿¡]|$)|(?<=[^\s¿¡])\s+(?=[¿¡])")
_ABBREVIATION = re.compile(r"\b(?:dr|dra|mr|mrs|sr|sra|vs|etc|p\.ej|[A-Z])\.$", re.I)
_SHORT = {"yes": "en", "yeah": "en", "yep": "en", "nope": "en", "sure": "en",
          "hello": "en", "hi": "en", "thanks": "en", "thank you": "en", "sorry": "en",
          "i see": "en", "of course": "en", "sí": "es", "gracias": "es", "claro": "es",
          "cómo": "es", "hola": "es", "perdón": "es", "no sé": "es", "por favor": "es",
          "de acuerdo": "es", "muchas gracias": "es", "no gracias": "es",
          "no thanks": "en"}
_SHARED = {"no", "ok", "okay", "diabetes", "insulin", "insulina"}


def split_sentences(text: str) -> list[str]:
    parts = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        part = text[start:match.end()].strip()
        if not part or _ABBREVIATION.search(part):
            continue
        parts.append(part)
        start = match.end()
    if tail := text[start:].strip():
        parts.append(tail)
    return parts


def detect_lang(text: str, fallback: str | None = None) -> str | None:
    """Usa solo evidencia del texto y, para fragmentos, del mismo tramo de audio."""
    tag = (fallback or "").lower().split("-")[0]
    hint = tag if tag in ("en", "es") else None
    words = re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)
    phrase = " ".join(words)
    if phrase in _SHORT:
        return _SHORT[phrase]
    if phrase in _SHARED or not words:
        return hint
    if text.lstrip().startswith(("¿", "¡")):
        return "es"
    if len(words) < 2:
        return hint
    # La estructura de la frase prevalece sobre un término médico prestado.
    if re.match(r"^(?:(?:no|sí|hola) )?(?:tengo|tomo|tomé|puedo|tiene|siento|me duele|soy|"
                r"buenos días|buenas tardes)\b", phrase):
        return "es"
    if re.match(r"^(?:(?:yes|hello|hi) )?(?:i (?:am|have|take|took|do|did|can|feel|don|didn|m)|"
                r"my|let me|please|good morning)\b", phrase):
        return "en"
    if re.match(r"^(?:are|is|was|were|do|does|did|can|could|have|has|will|would) "
                r"(?:you|i|he|she|they|we|it)\b", phrase):
        return "en"
    try:
        ranked = detect_langs(text)
    except LangDetectException:
        return hint
    if ranked and ranked[0].lang in ("en", "es") and ranked[0].prob >= 0.85:
        return ranked[0].lang
    return hint if tag and (not ranked or ranked[0].prob < 0.85) else None


def normalize_turns(segments: list[Segment]) -> list[Turn]:
    """Divide cambios de idioma detectables y conserva los tramos inciertos."""
    turns: list[Turn] = []
    for seg in segments:
        first_turn_in_segment = len(turns)
        sentences = split_sentences(seg.text)
        if seg.lang == "mixed":
            detected = {detect_lang(sentence) for sentence in sentences}
            if not {"en", "es"} <= detected:
                turns.append(Turn(seg.start, None, seg.text, seg.speaker))
                continue
        hint = seg.lang if len(sentences) == 1 else None
        for sent in sentences:
            lang = detect_lang(sent, fallback=hint)
            if (lang is not None and turns and turns[-1].lang == lang
                    and turns[-1].speaker == seg.speaker
                    and (seg.speaker is None or len(turns) > first_turn_in_segment)):
                turns[-1].text += " " + sent
            else:
                turns.append(Turn(start=seg.start, lang=lang, text=sent, speaker=seg.speaker))
    return turns
