"""Cliente de traducción DeepSeek V4-Flash vía formato Anthropic. Ver Instrucciones.md §7.

Reusa el SDK de Anthropic apuntando a base_url de DeepSeek. Valida salida con pydantic
y reintenta UNA vez con instrucción reforzada si el JSON falla. Nunca regex sobre el cuerpo.

Soporta streaming: `translate(..., on_text=cb)` streamea la respuesta y va entregando
el valor del campo "text" del JSON a medida que llega (para TTS por frases sin esperar
el JSON completo). La validación final con pydantic no cambia.

Si DeepSeek falla (402 insufficient balance, red, etc.), automáticamente fallback a
traductor local por diccionario médico (LocalTranslator).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable

from pydantic import BaseModel, ValidationError, field_validator

from .prompts import SYSTEM_PROMPT_V1, RETRY_SUFFIX, build_user_message

logger = logging.getLogger(__name__)


def _opposite_lang(source_lang: str | None, source_text: str = "") -> str:
    """Idioma destino = el OTRO idioma. Nunca adivina el mismo default para 'desconocido'."""
    lang = (source_lang or "").lower()[:2]
    if lang not in ("en", "es"):
        from .segmentation import detect_lang

        lang = detect_lang(source_text) or "en"
    return "es" if lang == "en" else "en"


class Translation(BaseModel):
    target_lang: str
    text: str
    confidence: float = 0.0
    needs_clarification: bool = False

    @field_validator("target_lang")
    @classmethod
    def _lang(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("en", "es"):
            raise ValueError("target_lang debe ser 'en' o 'es'")
        return v


class LocalTranslator:
    """Traductor local de fallback usando terminología + pattern matching.
    
    Usado cuando DeepSeek no está disponible (402, 429, network error, etc).
    No perfecto pero funcional para traducción médica con diccionario médico.
    """
    
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
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self._no_think = True  # se desactiva solo si el endpoint rechaza el flag
        self.client = Anthropic(
            api_key=key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic"),
            timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "30")),
            max_retries=2,      # reintento a nivel transporte (red/5xx), aparte del retry JSON
        )
        # Inicializar fallback local
        self.local_fallback = LocalTranslator()
        self._use_fallback = False  # Bandera para saber si estamos usando fallback
        logger.info("DeepSeekTranslator initialized with DeepSeek backend")

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
            except Exception as e:  # noqa
                # 402 (insufficient balance): propaga para que translate() degrade
                # al fallback local en vez de reintentar sin el flag.
                if getattr(e, "status_code", None) == 402:
                    logger.warning("DeepSeek 402 (insufficient balance); usando fallback local.")
                    self._use_fallback = True
                    raise
                # Endpoint no soporta el flag: desactívalo y sigue
                self._no_think = False
        return self._request(base, on_delta)

    def _request(self, base: dict, on_delta: Callable[[str], None] | None, **extra) -> str:
        if on_delta is None:
            msg = self.client.messages.create(**base, **extra)
            # mismo shape que Anthropic: content blocks
            return "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text").strip()
        chunks: list[str] = []
        with self.client.messages.stream(**base, **extra) as stream:
            for delta in stream.text_stream:
                chunks.append(delta)
                on_delta(delta)
        return "".join(chunks).strip()

    def _validate(self, raw: str, source_lang: str | None) -> Translation | None:
        """Parsea + valida forma JSON y, además, que el target_lang sea REALMENTE
        el opuesto del idioma de origen (no solo un valor válido en {en, es})."""
        try:
            result = Translation(**_parse(raw))
        except (ValidationError, ValueError, json.JSONDecodeError):
            return None

        src_norm = (source_lang or "").lower()[:2]
        if src_norm in ("en", "es") and result.target_lang == src_norm:
            logger.warning("translation_target_equals_source_label")
            return None

        from .segmentation import detect_lang

        detected_out = detect_lang(result.text, fallback=result.target_lang)
        if detected_out != result.target_lang:
            logger.warning("translation_output_language_mismatch")
            return None
        return result

    def translate(self, source_text: str, source_lang: str | None,
                  terminology_block: str, context: str = "",
                  on_text: Callable[[str], None] | None = None) -> Translation:
        """Traduce un turno. Si `on_text` viene, streamea y entrega incrementalmente el
        valor decodificado del campo "text" (el reintento NO streamea: evita hablar doble)."""
        user = build_user_message(source_text, source_lang, terminology_block, context)
        on_delta = None
        if on_text is not None:
            tfs = TextFieldStream()

            def on_delta(raw: str):  # noqa
                piece = tfs.feed(raw)
                if piece:
                    on_text(piece)
        try:
            raw = self._call(SYSTEM_PROMPT_V1, user, on_delta)
            result = self._validate(raw, source_lang)
            if result is not None:
                return result
            # reintento reforzado UNA vez (sin streaming al caller: evita hablar doble)
            raw2 = self._call(SYSTEM_PROMPT_V1 + RETRY_SUFFIX, user)
            result = self._validate(raw2, source_lang)
            if result is not None:
                return result
            # degradar a <UNCLEAR> en vez de crashear (§7: no inventar, pedir repetición)
            target = _opposite_lang(source_lang, source_text)
            return Translation(target_lang=target, text="<UNCLEAR>",
                               confidence=0.0, needs_clarification=True)
        except Exception as e:
            # Cualquier error en DeepSeek -> fallback a traductor local
            logger.warning(f"DeepSeek error ({type(e).__name__}): {str(e)[:100]}. Using local fallback.")
            self._use_fallback = True
            result = self.local_fallback.translate(source_text, source_lang, terminology_block, context)
            logger.info(f"Fallback translation: confidence={result.confidence:.1%}, clarification_needed={result.needs_clarification}")
            return result


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
