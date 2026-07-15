"""Logging estructurado no-PHI para el intérprete en vivo.

Sigue la intención de Instrucciones.md §"Logging": líneas JSON con session_id/turn_id,
solo metadatos (idioma, latencia, confianza, error codes) — NUNCA el contenido de los
turnos en producción. Contenido real solo bajo DEBUG_TRANSCRIPTS=true (flag existente).

Usa solo `logging` de stdlib (sin dependencias nuevas).
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
import uuid
from pathlib import Path

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "meta", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_dir: str | os.PathLike[str] = "logs", level: int = logging.INFO) -> None:
    """Idempotente: llamar varias veces no duplica handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    formatter = JsonFormatter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        path / "interpreter.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


def log_turn(logger: logging.Logger, *, session_id: str, turn_id: int,
             source_text: str = "", output_text: str = "", **meta) -> None:
    """Loguea metadatos de un turno. El texto real solo se incluye si
    DEBUG_TRANSCRIPTS=true (flag ya existente en .env para dev local)."""
    payload = {"session_id": session_id, "turn_id": turn_id, **meta}
    if os.getenv("DEBUG_TRANSCRIPTS", "").lower() == "true":
        payload["source_text"] = source_text
        payload["output_text"] = output_text
    logger.info("turn", extra={"meta": payload})
