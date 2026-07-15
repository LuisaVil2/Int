"""Clasificador de emergencias EN/ES — §11. Adaptado de LuisaVil2/Int (limpiado).

El bot completa el turno y AÑADE una alerta para el proveedor si detecta keywords.
"""
from __future__ import annotations

import re

EMERGENCY_TERMS: dict[str, list[str]] = {
    "chest_pain": ["chest pain", "chest pressure", "chest tightness",
                    "dolor de pecho", "dolor en el pecho", "presión en el pecho",
                    "opresión en el pecho"],
    "difficulty_breathing": ["can't breathe", "cannot breathe", "can't catch my breath",
                              "shortness of breath", "gasping for air",
                              "no puedo respirar", "dificultad para respirar", "me ahogo",
                              "me falta el aire", "no me entra el aire"],
    "heart_attack": ["heart attack", "infarto", "ataque al corazón", "infarto al corazón",
                       "infarto de miocardio"],
    "stroke": ["stroke", "face drooping", "slurred speech", "sudden weakness on one side",
                "derrame cerebral", "accidente cerebrovascular", "se me durmió un lado",
                "se me trabó la lengua", "se me cayó la cara"],
    "cardiac_arrest": ["cardiac arrest", "no pulse", "not breathing and unresponsive",
                        "paro cardíaco", "paro cardiaco", "paro cardiorrespiratorio"],
    "suicidal_ideation": ["suicidal", "suicide", "want to kill myself", "want to end my life",
                          "suicidio", "quitarme la vida", "matarme", "acabar con mi vida",
                          "no quiero vivir"],
    "anaphylaxis": ["anaphylaxis", "throat closing", "throat closing up", "severe allergic reaction",
                     "anafilaxia", "reacción alérgica grave", "se me cierra la garganta",
                     "se me está cerrando la garganta"],
    "seizure": ["seizure", "convulsing", "convulsión", "convulsion", "convulsionando"],
    "overdose": ["overdose", "took too many pills", "sobredosis", "se tomó demasiadas pastillas"],
    "hemorrhage": ["hemorrhage", "bleeding out", "won't stop bleeding", "heavy bleeding",
                    "hemorragia", "sangrando mucho", "no para de sangrar", "sangrado abundante"],
    "sepsis": ["sepsis", "septic shock", "septicemia",
                "choque séptico", "sepsis generalizada"],
    "loss_of_consciousness": ["loss of consciousness", "unconscious", "unresponsive", "passed out",
                              "pérdida de conciencia", "inconsciente", "se desmayó",
                              "perdió el conocimiento", "no responde"],
}

ALERT_EN = "[Interpreter alert: patient mentioned possible emergency keywords]"
ALERT_ES = "[Alerta del intérprete: se mencionaron posibles palabras clave de emergencia]"


class EmergencyClassifier:
    def classify(self, text: str) -> dict:
        low = text.lower()
        labels = [label for label, terms in EMERGENCY_TERMS.items()
                  if any(re.search(rf"\b{re.escape(t)}\b", low) for t in terms)]
        return {"is_emergency": bool(labels), "labels": labels,
                "force_qa_review": bool(labels)}
