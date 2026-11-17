"""Traducción DeepSeek con validación del turno completo y estado de revisión."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .prompts import SYSTEM_PROMPT_V1, RETRY_SUFFIX, build_user_message

logger = logging.getLogger(__name__)


def _opposite_lang(source_lang: str | None, source_text: str = "") -> str | None:
    """Idioma destino = el OTRO idioma. Nunca adivina el mismo default para 'desconocido'."""
    lang = (source_lang or "").lower().split("-")[0]
    if lang not in ("en", "es"):
        from .segmentation import detect_lang

        lang = detect_lang(source_text)
    if lang is None:
        return None
    return "es" if lang == "en" else "en"


class Translation(BaseModel):
    target_lang: str | None
    text: str
    confidence: float = 0.0
    needs_clarification: bool = False

    @field_validator("target_lang")
    @classmethod
    def _lang(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.lower().strip()
        if v not in ("en", "es"):
            raise ValueError("target_lang debe ser 'en' o 'es'")
        return v


class _ProviderTranslation(Translation):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_lang: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    needs_clarification: bool


# Explicit digit values and common dose units. This is a rejection check, not a
# proof of fidelity. Spellings outside these patterns require human evaluation.
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?=\b|[a-zµμ])", re.I)
_UNITS = {
    "mg": r"mg|milligrams?|miligramos?",
    "g": r"g|grams?|gramos?",
    "kg": r"kg|kilograms?|kilogramos?",
    "mcg": r"mcg|[µμ]g|micrograms?|microgramos?",
    "ml": r"ml|milliliters?|millilitres?|mililitros?",
    "l": r"l|liters?|litres?|litros?",
    "unit": r"iu|ui|units?|unidades?",
    "tablet": r"tablets?|tabletas?|comprimidos?",
    "capsule": r"capsules?|cápsulas?",
}
_DOSE = re.compile(
    _NUMBER.pattern + r"\s*(?:" +
    "|".join(f"(?P<{unit}>{forms})" for unit, forms in _UNITS.items()) + r")\b", re.I
)
_NEGATION = re.compile(
    r"\b(?:no|not|never|none|neither|nor|without|cannot|den(?:y|ies|ied)|"
    r"nunca|jamás|sin|tampoco|ningun[oa]?|ningún|niego|niega|negó)\b|n['’]t\b", re.I
)


def fidelity_issue(source: str, target: str) -> str | None:
    def normalized_number(value):
        # Tres cifras tras un separador pueden indicar miles o milésimas.
        return value if re.search(r"[.,]\d{3}(?:[.,]|$)", value) else value.replace(",", ".")

    def numbers(text):
        return [normalized_number(value) for value in _NUMBER.findall(text)]

    def doses(text):
        return [(normalized_number(_NUMBER.match(m.group()).group()), m.lastgroup)
                for m in _DOSE.finditer(text)]

    if numbers(source) != numbers(target):
        return "numbers_changed"
    if doses(source) != doses(target):
        return "dose_units_changed"
    if bool(_NEGATION.search(source)) != bool(_NEGATION.search(target)):
        return "negation_changed"
    return None


class LocalTranslator:
    """Utilidad experimental de sustitución léxica sin uso en el motor."""
    
    def __init__(self):
        # Frases médicas multi-palabra (EN -> ES)
        self.phrases_en_es = {
            "shortness of breath": "falta de aliento",
            "chest pain": "dolor en el pecho",
            "blood pressure": "presión arterial",
            "heart rate": "frecuencia cardíaca",
            "body temperature": "temperatura corporal",
            "heart attack": "infarto",
            "blood clot": "coágulo de sangre",
            "acute abdomen": "abdomen agudo",
            "loss of consciousness": "pérdida de conciencia",
            "allergic reaction": "reacción alérgica",
            "common cold": "resfriado común",
            "high fever": "fiebre alta",
            "severe pain": "dolor severo",
        }
        # Crear reverso
        self.phrases_es_en = {v: k for k, v in self.phrases_en_es.items()}
        
        # Palabras individuales (EN -> ES)
        self.words_en_es = {
            "patient": "paciente", "doctor": "médico", "physician": "médico",
            "nurse": "enfermero", "hospital": "hospital", "clinic": "clínica",
            "headache": "dolor de cabeza", "fever": "fiebre", "pain": "dolor",
            "dizziness": "mareos", "vertigo": "vértigo", "nausea": "náusea",
            "vomit": "vómito", "vomiting": "vómitos", "cough": "tos",
            "chest": "pecho", "heart": "corazón", "lung": "pulmón", "lungs": "pulmones",
            "blood": "sangre", "pressure": "presión", "temperature": "temperatura",
            "allergy": "alergia", "allergic": "alérgico", "medication": "medicamento",
            "drug": "droga", "prescription": "receta", "dose": "dosis",
            "symptom": "síntoma", "diagnosis": "diagnóstico", "treatment": "tratamiento",
            "surgery": "cirugía", "operation": "operación", "infection": "infección",
            "fracture": "fractura", "wound": "herida", "emergency": "emergencia",
            "ambulance": "ambulancia", "urgent": "urgente", "acute": "agudo",
            "chronic": "crónico", "diabetic": "diabético", "diabetes": "diabetes",
            "hypertension": "hipertensión", "asthma": "asma", "pneumonia": "neumonía",
            "cancer": "cáncer", "stroke": "accidente cerebrovascular", "seizure": "convulsión",
            "anxiety": "ansiedad", "depression": "depresión", "liver": "hígado",
            "kidney": "riñón", "stomach": "estómago", "intestine": "intestino",
            "bone": "hueso", "muscle": "músculo", "skin": "piel", "throat": "garganta",
            "eye": "ojo", "ear": "oído", "nose": "nariz", "mouth": "boca",
            "tongue": "lengua", "tooth": "diente", "teeth": "dientes", "arm": "brazo",
            "leg": "pierna", "hand": "mano", "foot": "pie", "head": "cabeza",
            "back": "espalda", "neck": "cuello", "shoulder": "hombro",
            "swelling": "hinchazón", "bruise": "moratón", "bleed": "sangrar",
            "bleeding": "sangrado", "injury": "lesión", "burn": "quemadura",
            "cut": "corte", "poison": "veneno", "alcohol": "alcohol",
            "tobacco": "tabaco", "smoking": "fumar", "pregnant": "embarazada",
            "pregnancy": "embarazo", "baby": "bebé", "newborn": "recién nacido",
            "child": "niño", "elderly": "adulto mayor", "age": "edad", "sex": "sexo",
            "male": "hombre", "female": "mujer", "human": "humano",
            "body": "cuerpo", "health": "salud", "illness": "enfermedad",
            "disease": "enfermedad", "disorder": "trastorno", "condition": "condición",
        }
        # Crear reverso
        self.words_es_en = {v: k for k, v in self.words_en_es.items()}
    
    def translate(self, source_text: str, source_lang: str | None,
                  terminology_block: str, context: str = "") -> Translation:
        """Traducción rápida local usando diccionario médico + pattern matching."""
        
        # Detectar idioma si no se proporciona
        target_lang = _opposite_lang(source_lang, source_text)
        if target_lang is None:
            return Translation(target_lang=None, text="<UNCLEAR>", needs_clarification=True)
        
        # Diccionarios a usar basados en idioma
        phrases_dict = self.phrases_en_es if target_lang == "es" else self.phrases_es_en
        words_dict = self.words_en_es if target_lang == "es" else self.words_es_en
        
        import re
        
        result = source_text
        matched_terms = 0
        total_terms = 0
        
        # Primero reemplazar frases multi-palabra (más específico)
        for phrase, translation in phrases_dict.items():
            pattern = r'\b' + re.escape(phrase) + r'\b'
            matches = re.findall(pattern, result, flags=re.IGNORECASE)
            if matches:
                # Preservar capitalización del primer match
                for match in matches:
                    if match[0].isupper() and translation:
                        trans = translation[0].upper() + translation[1:]
                    else:
                        trans = translation
                    result = re.sub(pattern, trans, result, count=1, flags=re.IGNORECASE)
                    matched_terms += 1
                total_terms += len(matches)
        
        # Luego reemplazar palabras individuales
        words = result.split()
        translated_words = []
        
        for word in words:
            # Detectar si es un artículo, preposición, número o código
            lower_word = word.lower().strip('.,!?;:')
            if lower_word in ['a', 'an', 'the', 'and', 'or', 'of', 'el', 'la', 'los', 'las', 'de', 'y', 'o']:
                translated_words.append(word)
                continue
            
            if lower_word.replace('.', '').replace(',', '').isdigit():
                translated_words.append(word)
                continue
            
            total_terms += 1
            clean_word = re.sub(r'[^\w]', '', lower_word)
            
            # Buscar en diccionario de palabras
            found = False
            for key, value in words_dict.items():
                if clean_word == key:
                    # Preservar capitalización y puntuación
                    translated = value
                    if word and word[0].isupper() and translated:
                        translated = translated[0].upper() + translated[1:]
                    # Re-añadir puntuación original
                    punctuation = re.findall(r'[^\w]+$', word)
                    if punctuation:
                        translated += punctuation[0]
                    translated_words.append(translated)
                    matched_terms += 1
                    found = True
                    break
            
            if not found:
                translated_words.append(word)
        
        result = " ".join(translated_words)
        
        # Capitalizar si el original estaba capitalizado
        if source_text and source_text[0].isupper() and result:
            result = result[0].upper() + result[1:]
        
        # Confidence basado en términos traducidos
        confidence = min(0.85, matched_terms / max(total_terms, 1))  # cap at 0.85 for fallback
        needs_clarification = confidence < 0.5

        if needs_clarification:
            # Baja confianza: esto es sustitución palabra-por-palabra, NO una traducción
            # real. No la presentamos como si lo fuera -- se degrada a <UNCLEAR> (mismo
            # contrato que el degrade de DeepSeekTranslator) para que el resto del
            # pipeline la trate como "necesita revisión humana", no como una traducción
            # válida para hablar/mostrar con autoridad.
            return Translation(
                target_lang=target_lang,
                text="<UNCLEAR>",
                confidence=confidence,
                needs_clarification=True,
            )

        return Translation(
            target_lang=target_lang,
            text=result.strip(),
            confidence=confidence,
            needs_clarification=needs_clarification
        )



class DeepSeekTranslator:
    def __init__(self):
        from anthropic import Anthropic

        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("Falta DEEPSEEK_API_KEY en .env")
        self.model = os.getenv("DEEPSEEK_MODEL") or "deepseek-flash"
        self._no_think = True  # se desactiva solo si el endpoint rechaza el flag
        self.client = Anthropic(
            api_key=key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic"),
            timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "30")),
            max_retries=2,      # reintento a nivel transporte (red/5xx), aparte del retry JSON
        )
    def _call(self, system: str, user: str,
              on_delta: Callable[[str], None] | None = None) -> str:
        # FIX #2 latencia: non-thinking explícito + temperature 0 + max_tokens corto.
        # thinking se manda vía extra_body (el endpoint DeepSeek lo acepta aunque el SDK
        # Anthropic no lo tipe). Si el endpoint lo rechaza, reintenta sin él.
        base = dict(model=self.model, max_tokens=400, temperature=0, system=system,
                    messages=[{"role": "user", "content": user}])
        if self._no_think:
            try:
                return self._request(base, on_delta,
                                     extra_body={"thinking": {"type": "disabled"}})
            except Exception as exc:
                if (getattr(exc, "status_code", None) not in (400, 422)
                        or "thinking" not in str(exc).lower()):
                    raise
                self._no_think = False
        return self._request(base, on_delta)

    def _request(self, base: dict, on_delta: Callable[[str], None] | None, **extra) -> str:
        if on_delta is None:
            msg = self.client.messages.create(**base, **extra)
            if msg.stop_reason != "end_turn":
                raise ValueError("translation_incomplete")
            return "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text").strip()
        chunks: list[str] = []
        with self.client.messages.stream(**base, **extra) as stream:
            for delta in stream.text_stream:
                chunks.append(delta)
                on_delta(delta)
            if stream.get_final_message().stop_reason != "end_turn":
                raise ValueError("translation_incomplete")
        return "".join(chunks).strip()

    def _validate(self, raw: str, source_lang: str | None) -> Translation | None:
        """Parsea + valida forma JSON y, además, que el target_lang sea REALMENTE
        el opuesto del idioma de origen (no solo un valor válido en {en, es})."""
        try:
            result = _ProviderTranslation.model_validate_json(raw)
        except (ValidationError, ValueError):
            return None

        src_norm = (source_lang or "").lower()[:2]
        if src_norm in ("en", "es") and result.target_lang == src_norm:
            logger.warning("translation_target_equals_source_label")
            return None

        if not result.text.strip():
            return None
        if result.needs_clarification or "<unclear>" in result.text.casefold():
            return Translation(target_lang=result.target_lang, text="<UNCLEAR>",
                               needs_clarification=True)

        from .segmentation import detect_lang

        # The provider's own target label is not independent evidence.
        detected_out = detect_lang(result.text)
        shared_short = result.text.strip().casefold().rstrip(".!?") in {"no", "ok", "okay"}
        if detected_out != result.target_lang and not shared_short:
            logger.warning("translation_output_language_mismatch")
            return None
        return result

    def translate(self, source_text: str, source_lang: str | None,
                  terminology_block: str, context: str = "",
                  on_text: Callable[[str], None] | None = None) -> Translation:
        """Traduce con dirección explícita y entrega únicamente texto terminado."""
        source_lang = (source_lang or "").lower().split("-")[0]
        target = {"en": "es", "es": "en"}.get(source_lang)
        unclear = Translation(target_lang=target, text="<UNCLEAR>", needs_clarification=True)
        if target is None:
            return unclear
        user = build_user_message(source_text, source_lang, terminology_block, context)
        for suffix in ("", RETRY_SUFFIX):
            try:
                raw = self._call(SYSTEM_PROMPT_V1 + suffix, user)
                result = self._validate(raw, source_lang)
            except Exception:
                logger.warning("translation_provider_failed")
                return unclear
            if result is None:
                continue
            if result.needs_clarification:
                return unclear
            if issue := fidelity_issue(source_text, result.text):
                logger.warning("translation_fidelity_check_failed", extra={"meta": {"reason": issue}})
                continue
            if on_text is not None:
                on_text(result.text)
            return result
        return unclear


_TEXT_FIELD = re.compile(r'"text"\s*:\s*"')


class TextFieldStream:
    """Extrae incrementalmente el valor del campo "text" de un JSON que llega en chunks.

    feed(chunk) devuelve el texto NUEVO decodificado (escapes JSON resueltos) desde la
    última llamada. Antes de encontrar `"text": "` y después de la comilla de cierre
    devuelve "".
    """

    def __init__(self):
        self.raw = ""
        self._start: int | None = None
        self._end: int | None = None
        self._emitted = ""

    def feed(self, chunk: str) -> str:
        if self._end is not None:
            return ""
        self.raw += chunk
        if self._start is None:
            m = _TEXT_FIELD.search(self.raw)
            if not m:
                return ""
            self._start = m.end()
        # busca la comilla de cierre no escapada
        i = self._start
        while self._end is None:
            j = self.raw.find('"', i)
            if j == -1:
                break
            k, n = j - 1, 0
            while k >= self._start and self.raw[k] == "\\":
                n += 1
                k -= 1
            if n % 2 == 0:
                self._end = j
            else:
                i = j + 1
        seg = self.raw[self._start:self._end if self._end is not None else len(self.raw)]
        decoded = _decode_json_string_prefix(seg)
        delta = decoded[len(self._emitted):]
        self._emitted = decoded
        return delta


def _decode_json_string_prefix(seg: str) -> str:
    """Decodifica el mayor prefijo válido de un cuerpo de string JSON (el final puede
    quedar a mitad de un escape \\uXXXX mientras siguen llegando chunks)."""
    while seg:
        try:
            return json.loads(f'"{seg}"', strict=False)
        except json.JSONDecodeError:
            seg = seg[:-1]
    return ""


def _parse(raw: str) -> dict:
    """Extrae el objeto JSON. Tolera ```json fences pero NO usa regex sobre el contenido."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    # primer { ... último } por si el modelo añadió texto
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1:
        s = s[i:j + 1]
    return json.loads(s)
