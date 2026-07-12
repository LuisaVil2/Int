"""Cliente de traducción DeepSeek V4-Flash vía formato Anthropic. Ver Instrucciones.md §7.

Reusa el SDK de Anthropic apuntando a base_url de DeepSeek. Valida salida con pydantic
y reintenta UNA vez con instrucción reforzada si el JSON falla. Nunca regex sobre el cuerpo.

Soporta streaming: `translate(..., on_text=cb)` streamea la respuesta y va entregando
el valor del campo "text" del JSON a medida que llega (para TTS por frases sin esperar
el JSON completo). La validación final con pydantic no cambia.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable

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
            except Exception:  # noqa — endpoint no soporta el flag: desactívalo y sigue
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
            return Translation(**_parse(raw))
        except (ValidationError, ValueError, json.JSONDecodeError):
            pass
        # reintento reforzado UNA vez (sin streaming al caller)
        try:
            raw2 = self._call(SYSTEM_PROMPT_V1 + RETRY_SUFFIX, user)
            return Translation(**_parse(raw2))
        except (ValidationError, ValueError, json.JSONDecodeError):
            # degradar a <UNCLEAR> en vez de crashear (§7: no inventar, pedir repetición)
            target = "en" if (source_lang or "").startswith("es") else "es"
            return Translation(target_lang=target, text="<UNCLEAR>",
                               confidence=0.0, needs_clarification=True)


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
