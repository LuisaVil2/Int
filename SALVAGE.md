# Rescate de LuisaVil2/Int

Auditoría del repo `https://github.com/LuisaVil2/Int` para reusar lo válido antes de
reemplazarlo con este proyecto.

## ✅ Salvado (adaptado y limpiado a `src/`)
| Origen en Int | Destino | Por qué |
|---|---|---|
| `backend/confidence/emergency.py` | `src/emergency.py` | Clasificador emergencias EN/ES → implementa §11 |
| `backend/confidence/scoring.py` | `src/confidence.py` | Confianza multi-señal + ruteo QA (auto/qa/pausa). Idea que el doc no tenía |
| `backend/memory/conversation.py` | `src/memory.py` | Contexto conversacional al LLM → mejora coherencia |
| `backend/providers/deepgram.py` | `src/stt.py` (ruta DG) | `diarize=True` + `language=multi` + `nova-2-medical` → Hallazgo #1 |

## ❌ Rechazado (conflicto con Instrucciones.md)
| Item | Razón |
|---|---|
| `backend/providers/deepseek.py` | Usa SDK **OpenAI** + `deepseek-chat` + texto plano. §7 manda SDK **Anthropic** + JSON estricto |
| `backend/providers/xtts.py` | XTTS vía HuggingFace. §9 manda **ElevenLabs** voz clonada |
| WebSocket `/ws/audio` + `chrome_extension/` | Telefonía vía navegador. §6 manda **Twilio Media Streams** |
| `glosario_medico.json` único | §8 manda **CSVs versionados por especialidad** en `data/terminology/` |
| Estilo de código | One-liners, semicolons, sin tests. Reescrito limpio |

## Veredicto
El repo Int tenía buena **arquitectura modular** y 4 conceptos valiosos, pero
integraciones equivocadas (OpenAI, XTTS, sin Twilio) frente a la spec. Se rescató lo
conceptual, se descartó el resto. Este proyecto reemplaza a Int como base.
