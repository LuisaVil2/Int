"""Motor de confianza multi-señal + ruteo QA. Salvado de LuisaVil2/Int (limpiado).

Idea valiosa que Instrucciones.md no tenía: combinar señales (ASR, LLM, glosario...)
en un score 0-100 y rutear: aprobación automática / revisión QA / pausa manual.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConfidenceInputs:
    asr_confidence: float = 0.85
    llm_confidence: float = 0.9
    terminology_certainty: float = 0.85
    conversation_consistency: float = 0.9
    glossary_match: float = 0.8


class ConfidenceEngine:
    weights = {
        "asr_confidence": 0.30,
        "llm_confidence": 0.30,
        "terminology_certainty": 0.15,
        "conversation_consistency": 0.15,
        "glossary_match": 0.10,
    }

    def score(self, i: ConfidenceInputs) -> int:
        return round(sum(max(0.0, min(1.0, getattr(i, k))) * w
                         for k, w in self.weights.items()) * 100)

    def route(self, score: int) -> str:
        if score >= 95:
            return "automatic_approval"
        if score >= 80:
            return "qa_review"
        return "pause_manual_approval"
