"""Clasificador de emergencias EN/ES — §11. Adaptado de LuisaVil2/Int (limpiado).

El bot completa el turno y AÑADE una alerta para el proveedor si detecta keywords.
"""
from __future__ import annotations

import re

EMERGENCY_TERMS: dict[str, list[str]] = {
    "chest_pain": ["chest pain", "dolor de pecho", "dolor en el pecho"],
    "difficulty_breathing": ["can't breathe", "cannot breathe", "shortness of breath",
                              "no puedo respirar", "dificultad para respirar", "me ahogo"],
    "heart_attack": ["heart attack", "infarto"],
    "stroke": ["stroke", "derrame cerebral", "accidente cerebrovascular"],
    "cardiac_arrest": ["cardiac arrest", "paro cardíaco", "paro cardiaco"],
    "suicidal_ideation": ["suicidal", "suicide", "suicidio", "quitarme la vida", "matarme"],
    "anaphylaxis": ["anaphylaxis", "anafilaxia", "reacción alérgica grave"],
    "seizure": ["seizure", "convulsión", "convulsion"],
    "overdose": ["overdose", "sobredosis"],
    "hemorrhage": ["hemorrhage", "hemorragia", "bleeding out", "sangrando mucho"],
}

ALERT_EN = "[Interpreter alert: patient mentioned possible emergency keywords]"


class EmergencyClassifier:
    def classify(self, text: str) -> dict:
        low = text.lower()
        labels = [label for label, terms in EMERGENCY_TERMS.items()
                  if any(re.search(rf"\b{re.escape(t)}\b", low) for t in terms)]
        return {"is_emergency": bool(labels), "labels": labels,
                "force_qa_review": bool(labels)}
