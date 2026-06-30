"""Cliente de traducción DeepSeek V4-Flash vía formato Anthropic. Ver Instrucciones.md §7.

Reusa el SDK de Anthropic apuntando a base_url de DeepSeek. Valida salida con pydantic
y reintenta UNA vez con instrucción reforzada si el JSON falla. Nunca regex sobre el cuerpo.
"""
from __future__ import annotations

import json
import os

from pydantic import BaseModel, ValidationError, field_validator

from .prompts import SYSTEM_PROMPT_V1, RETRY_SUFFIX, build_user_message


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
        )

    def _call(self, system: str, user: str) -> str:
        # FIX #2 latencia: non-thinking explícito + temperature 0 + max_tokens corto.
        # thinking se manda vía extra_body (el endpoint DeepSeek lo acepta aunque el SDK
        # Anthropic no lo tipe). Si el endpoint lo rechaza, reintenta sin él.
        base = dict(model=self.model, max_tokens=400, temperature=0, system=system,
                    messages=[{"role": "user", "content": user}])
        if self._no_think:
            try:
                msg = self.client.messages.create(**base,
                                                  extra_body={"thinking": {"type": "disabled"}})
            except Exception:  # noqa — endpoint no soporta el flag: desactívalo y sigue
                self._no_think = False
                msg = self.client.messages.create(**base)
        else:
            msg = self.client.messages.create(**base)
        # mismo shape que Anthropic: content blocks
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    def translate(self, source_text: str, source_lang: str | None,
                  terminology_block: str, context: str = "") -> Translation:
        user = build_user_message(source_text, source_lang, terminology_block, context)
        raw = self._call(SYSTEM_PROMPT_V1, user)
        try:
            return Translation(**_parse(raw))
        except (ValidationError, ValueError, json.JSONDecodeError):
            pass
        # reintento reforzado UNA vez
        try:
            raw2 = self._call(SYSTEM_PROMPT_V1 + RETRY_SUFFIX, user)
            return Translation(**_parse(raw2))
        except (ValidationError, ValueError, json.JSONDecodeError):
            # degradar a <UNCLEAR> en vez de crashear (§7: no inventar, pedir repetición)
            target = "en" if (source_lang or "").startswith("es") else "es"
            return Translation(target_lang=target, text="<UNCLEAR>",
                               confidence=0.0, needs_clarification=True)


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
