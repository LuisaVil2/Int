# Hallazgos — test intérprete vs video GERD (2026-06-30)

Pipeline: `audio/gerd.wav` → Whisper small (STT) → terminología → DeepSeek V4-Flash → tabla.
35 segmentos. **29 traducidos (calidad alta, conf 0.95-1.0) · 6 `<UNCLEAR>`.**

## ✅ Lo que funcionó
- **DeepSeek V4-Flash vía formato Anthropic: confirmado real.** `base_url=api.deepseek.com/anthropic` + `model=deepseek-v4-flash` autentican y responden. El doc tenía razón.
- **Calidad de traducción médica: excelente** en segmentos limpios. Ejemplos reales:
  - "H2 blockers" → "bloqueadores H2"; "proton pump inhibitors PPI's" → "inhibidores de la bomba de protones (IBP)"
  - "endoscopy to assess the extent of esophageal damage" → "endoscopia para evaluar la extensión del daño esofágico"
  - Registro formal médico preservado; sin opinar; sin agregar.
- JSON estricto + pydantic + degradación a `<UNCLEAR>` funcionando.

## 🔴 Hallazgo #1 (crítico): STT segmenta MAL las vueltas
Whisper corta por VAD acústico, NO por hablante/idioma. Resultado: **casi cada segmento mezcla fin de una frase EN + inicio de la respuesta ES** (o viceversa). Ej. segmento 1: *"Good morning Mr. Escalante... Buenos días doctor, últimamente he estado"* — doctor Y paciente en el mismo chunk.
- Rompe el modelo turno-por-turno: el LLM no puede elegir una sola `target_lang` → `<UNCLEAR>`.
- Los 6 `<UNCLEAR>` son exactamente segmentos code-switched.
- **Fix:** usar Deepgram Nova-3 con `utterances` + diarización (§3/§10), o un VAD propio que corte por fin-de-turno. NO Whisper whole-file para esto.

## 🔴 Hallazgo #2: latencia muy por encima del target
- Limpios: 2-8s. Ambiguos: 14-25s. Target doc DeepSeek = 900ms.
- Causas: sin streaming (batch offline) + posible **thinking mode** en input ambiguo comiéndose `max_tokens=1024` antes del JSON (explica también algunos `<UNCLEAR>`).
- **Fix:** forzar non-thinking explícito; usar streaming + primer-chunk; subir `max_tokens` o cortar thinking.

## 🟡 Hallazgo #3: ruido de STT
Whisper small: "exophagus", "Antiasid", "gastroesofagio". Deepgram Nova-3 o Whisper medium lo limpian. No bloqueante para evaluar traducción.

## Backlog priorizado
1. Cambiar STT a Deepgram Nova-3 (utterances+diarización) → resuelve #1 y #3.
2. Enforce non-thinking + streaming en `deepseek_service` → resuelve #2.
3. Re-correr este mismo test → esperar 0 `<UNCLEAR>`, latencia <3s/segmento.
4. Recién entonces: medir vs intérprete humano del video (calidad fiel).

---

# RESUELTO (mismo día) — fixes implementados

| Métrica | Antes | Después |
|---|---|---|
| `<UNCLEAR>` | 6/35 | **0/19** |
| Latencia/turno | 2–25s | **1.2–2.7s** |
| Turnos | fragmentos code-switched | monolingües coherentes |

**Fix #1 — `src/segmentation.py` (`normalize_turns`):** parte segmentos STT en frases,
detecta idioma por frase (langdetect EN/ES), reagrupa frases contiguas del mismo idioma
en turnos monolingües. Elimina los `<UNCLEAR>` sin depender de Deepgram. Complementado
con `diarize=True` + `nova-2-medical` en la ruta Deepgram (`src/stt.py`).

**Fix #2 — `src/translate.py`:** `temperature=0`, `max_tokens=400`, thinking desactivado
vía `extra_body={"thinking":{"type":"disabled"}}` (con fallback si el endpoint lo rechaza).
Más la entrada monolingüe limpia elimina los reintentos lentos. Latencia ~10×.

**Fix #3 — STT:** ruta Deepgram default `nova-2-medical` (vocab clínico). Whisper sigue como
fallback gratis. Residual: nombres propios ("Escalante") confunden al detector en turnos
muy cortos → se resuelve con speaker IDs de diarización Deepgram.

**Bonus (salvado de LuisaVil2/Int):** memoria conversacional (contexto al LLM),
motor de confianza + ruteo QA (auto/qa/pausa), clasificador de emergencias §11.
Ver columnas `route` y `emerg` en `out/result.md`.
