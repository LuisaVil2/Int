# CLAUDE.md — Medical Interpreter Voicebot

> Este archivo es contexto persistente para Claude Code. Léelo completo antes de tocar el repo. Si una decisión técnica contradice este documento, **pregunta antes de avanzar**.

---

## 1. Qué estamos construyendo

Un **bot intérprete médico bilingüe EN↔ES** que reemplaza al intérprete humano en llamadas telefónicas clínicas (consultas, triage, follow-ups, agendamiento). El bot habla con **voz femenina clonada** del operador y se activa/desactiva desde un toggle en el dashboard.

**Flujo de uso real:**
1. El operador llega a su turno y enciende el bot (`POST /api/toggle`).
2. Una llamada entrante a su número Twilio dispara el webhook.
3. El bot contesta automáticamente con su voz femenina clonada y conduce la llamada como intérprete consecutivo: escucha al proveedor médico (EN), traduce y habla en ES al paciente, escucha la respuesta del paciente (ES), traduce y habla en EN al proveedor. Loop.
4. Cuando el operador apaga el toggle, las nuevas llamadas caen a su softphone humano.

**Quién lo usa:** intérpretes médicos profesionales que quieren multiplicar su capacidad sin perder la voz/relación con clínicas que ya los conocen.

**Lo que NO es:**
- No es un chatbot de texto.
- No es interpretación simultánea (es consecutiva, con turnos).
- No es un traductor general — está restringido a contexto médico/clínico.
- No es un sustituto certificado de un intérprete humano en emergencias (ver §11).

---

## 2. Arquitectura del pipeline

```
        ┌─────────┐    PSTN     ┌─────────┐    WebSocket    ┌─────────────┐
Paciente│ Teléfono│ ──────────► │ Twilio  │ ───────────────►│  FastAPI    │
        └─────────┘             │  Voice  │  Media Streams  │  backend    │
                                └────┬────┘   mulaw 8kHz    └──────┬──────┘
                                     │                              │
                                     │       TwiML response         │
                                     │ ◄────────────────────────────┤
                                     ▼                              │
                                  Speaker                           │
                                     ▲                              │
                                     │                              ▼
                                     │                       ┌─────────────┐
                                     │                       │ Interpreter │
                                     │                       │   Agent     │
                                     │                       └──┬───┬───┬──┘
                                     │                          │   │   │
                                     │      ┌───────────────────┘   │   └────────────────┐
                                     │      ▼                       ▼                    ▼
                                     │  ┌────────┐         ┌──────────────┐       ┌──────────┐
                                     │  │Deepgram│         │  DeepSeek    │       │ElevenLabs│
                                     │  │  STT   │         │  V4-Flash    │       │   TTS    │
                                     │  │ Nova-3 │         │  translate   │       │ voz fem. │
                                     │  └────────┘         └──────────────┘       └────┬─────┘
                                     │                            ▲                    │
                                     │                            │                    │
                                     │                   ┌────────┴────────┐           │
                                     │                   │ data/terminology│           │
                                     │                   │  glosarios médic│           │
                                     │                   └─────────────────┘           │
                                     └──────────────────────────────────────────────────┘
                                              audio mulaw 8kHz de vuelta a Twilio
```

**Presupuesto de latencia (objetivo end-to-end < 1.8s por turno):**

| Etapa                          | Target  | Tope duro |
|--------------------------------|---------|-----------|
| VAD detecta fin de turno       | 400 ms  | 700 ms    |
| Deepgram → transcripción final | 200 ms  | 400 ms    |
| DeepSeek V4-Flash traducción   | 500 ms  | 900 ms    |
| ElevenLabs primer chunk audio  | 300 ms  | 600 ms    |
| Twilio playback hasta oreja    | 100 ms  | 200 ms    |
| **Total perceptible**          | **1.5s**| **2.8s**  |

Si pasamos del tope duro, registramos como SLO violation. El usuario humano tolera ~2s de pausa entre turnos en interpretación consecutiva sin sentirla "rota".

---

## 3. Stack

- **Backend:** Python 3.11+, FastAPI, uvicorn (ASGI), `asyncio` everywhere.
- **Telefonía:** Twilio Programmable Voice + Media Streams (WebSocket bidireccional, mulaw 8kHz, base64 frames).
- **STT:** Deepgram Nova-3 streaming, multilingual (auto-detección EN/ES).
- **LLM traducción:** **DeepSeek V4-Flash** (`deepseek-v4-flash`) — motor principal. Lanzado abril 2026, 284B totales / 13B activos, contexto 1M, ~$0.14/$0.28 por millón de tokens. **Clave de integración: la API de DeepSeek es compatible con el formato Anthropic** (`ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`), así que reusamos el cliente del SDK Anthropic que ya teníamos, solo cambiando base URL y API key. Modo **non-thinking** por defecto (latencia); thinking solo si el agente marca alta ambigüedad.
- **Fallback de precisión:** modelo más fuerte (configurable) SOLO para turnos marcados como "alta ambigüedad médica". DeepSeek V4-Flash no es frontier (ranking medio en benchmarks); para dosis/fármacos/siglas raras conviene escalar. Ver §7.
- **TTS:** ElevenLabs streaming, **voz femenina clonada** del operador, modelo `eleven_turbo_v2_5` (multilingüe, baja latencia).
- **Terminología médica:** carpeta `data/terminology/` con glosarios EN↔ES, abreviaturas clínicas, nombres de fármacos. Se inyecta como contexto al LLM. Ver §7 y §8.
- **Estado:** in-memory por ahora (`BotState` singleton). Migrar a Redis cuando haya >1 worker.
- **Logging:** `structlog` JSON, todas las líneas con `call_sid` y `turn_id`.
- **Tests:** pytest + pytest-asyncio. Para audio: fixtures WAV → mulaw.

---

## 4. Estructura del proyecto

```
backend/
├── app/
│   ├── main.py                      # FastAPI app, lifespan, routers
│   ├── config.py                    # Settings (pydantic-settings, .env)
│   ├── state.py                     # BotState singleton (toggle, métricas)
│   ├── api/
│   │   ├── status.py                # GET /api/status, POST /api/toggle  ✅
│   │   └── twilio.py                # POST /api/twilio/voice → TwiML     🚧
│   ├── websockets/
│   │   └── media.py                 # WS /ws/media (Twilio Media Stream) 🚧
│   ├── services/
│   │   ├── deepgram_streamer.py     # STT streaming                       ✅
│   │   ├── deepseek_service.py      # Traducción (cliente fmt Anthropic) 🚧
│   │   └── elevenlabs_service.py    # TTS streaming voz fem. → mulaw 8kHz🚧
│   ├── agents/
│   │   └── interpreter.py           # Orquestador EN↔ES (turn-taking)    🚧
│   ├── audio/
│   │   ├── codec.py                 # mulaw <-> PCM16 helpers
│   │   └── vad.py                   # webrtcvad o silero, decisión EOT
│   ├── terminology/
│   │   └── loader.py                # Carga + indexa glosarios de data/  🚧
│   └── prompts/
│       └── medical_interpreter.py   # System prompts versionados
├── data/
│   └── terminology/                 # ← carpeta de terminología médica   🚧
│       ├── glossary_en_es.csv       # término EN → término ES + notas
│       ├── abbreviations.csv        # siglas clínicas (BP, SOB, Rx, etc.)
│       ├── drug_names.csv           # fármacos: marca ↔ genérico ↔ ES
│       ├── specialties/             # glosarios por especialidad
│       │   ├── cardiology.csv
│       │   ├── pediatrics.csv
│       │   └── oncology.csv
│       └── README.md                # formato y cómo añadir términos
├── tests/
├── pyproject.toml
└── .env.example
```

Leyenda: ✅ hecho · 🚧 pendiente

---

## 5. Estado actual (a la fecha de este archivo)

**Hecho:**
- Scaffold FastAPI + config + lifespan.
- `BotState` con `is_active`, endpoints `/api/status` y `/api/toggle`.
- `DeepgramStreamer`: conecta, envía mulaw, recibe transcripciones (interim + final).

**Pendiente, en orden sugerido:**
1. `api/twilio.py` — webhook entrante que responde TwiML con `<Connect><Stream url="wss://.../ws/media"/>`.
2. `websockets/media.py` — handler del WS de Media Stream. Parsea eventos `start`, `media`, `mark`, `stop`. Decodifica base64 → bytes mulaw → empuja a Deepgram. Encola audio TTS de vuelta como base64 mulaw.
3. **Stubs** para `deepseek_service.py` y `elevenlabs_service.py` que devuelvan eco / silencio. Validar loop completo Twilio→Deepgram→stub→Twilio antes de gastar tokens/créditos.
4. `terminology/loader.py` + poblar `data/terminology/` con un glosario base.
5. `deepseek_service.py` real (ver §7), con inyección de terminología.
6. `elevenlabs_service.py` real con la voz femenina clonada del operador.
7. `agents/interpreter.py` — orquesta turnos, decide dirección de traducción, maneja barge-in.
8. VAD propio para detectar fin de turno (Deepgram `is_final` no siempre alcanza).
9. Métricas: latencia por etapa, tasa de error, longitud de turno.

---

## 6. Twilio — detalles que duelen si los ignoras

- **Audio format:** mulaw, 8000 Hz, mono, frames de 20ms (160 bytes payload). Twilio envía base64 en el campo `media.payload` del evento WS.
- **Eventos del WS** en orden: `connected` → `start` (incluye `streamSid`, `callSid`) → muchos `media` → `stop`. Guardar `streamSid` para enviar audio de vuelta.
- **Enviar audio:** mismo formato. Estructura `{"event":"media","streamSid":"...","media":{"payload":"<base64 mulaw 20ms>"}}`. Twilio bufferea ~5s, no envíes faster-than-real-time grandes lotes.
- **Marks:** usa eventos `mark` para saber cuándo Twilio terminó de reproducir un chunk TTS. Crítico para barge-in.
- **Topología:** arrancamos con **flujo mediado** (el bot SIEMPRE habla, paciente y proveedor nunca se conectan directo) usando `<Connect><Stream>`. Migrar a **Twilio Conference** (bot como participante muted-toggleable) cuando se quiera ofrecer "pasar a humano" en vivo.
- **Webhook security:** validar firma `X-Twilio-Signature` en todo endpoint público. No skipear ni en dev — usa ngrok con la firma habilitada.

---

## 7. DeepSeek V4-Flash — cómo lo usamos (motor de traducción)

**Modelo:** `deepseek-v4-flash`, modo **non-thinking** por defecto. Modo **thinking** solo si el agente marca el turno como "alta ambigüedad médica" (siglas raras, dosis, nombres de fármacos). Si la precisión no alcanza, escalar al **modelo de fallback** configurado en `.env`.

**Integración (importante):** la API de DeepSeek habla el formato Anthropic. Reusamos el cliente del SDK de Anthropic apuntando a:
```
base_url = "https://api.deepseek.com/anthropic"
api_key  = $DEEPSEEK_API_KEY
model    = "deepseek-v4-flash"
```
Esto significa que `deepseek_service.py` es casi idéntico en forma a un cliente Anthropic — mismo shape de `messages`, mismo parsing de `content` blocks.

> ⚠️ Nota de fecha: los endpoints legacy `deepseek-chat` / `deepseek-reasoner` se retiran después del **24 jul 2026**; ya mapean a `deepseek-v4-flash` (non-thinking / thinking). Usar el nombre nuevo directamente.

**Rol:** traducir, NO conversar. El agente NUNCA debe responder preguntas que le hagan al "intérprete" — si el paciente dice "¿usted qué cree, doctora?", lo traduce literal al EN para el proveedor, no opina.

**System prompt** (vive en `app/prompts/medical_interpreter.py`, versionado):
- Eres una intérprete médica profesional. Traduces fielmente EN↔ES.
- Preservas registro: si el proveedor es formal, lo eres en ES. Si el paciente usa modismos, los reflejas con equivalentes naturales en EN.
- Usas la terminología provista en el bloque `<terminology>` (inyectado desde `data/terminology/`) como autoridad. Si un término aparece ahí, usas esa traducción.
- Términos médicos: terminología clínica estándar en cada idioma. Dosis, unidades y números NUNCA se aproximan.
- Si no entiendes algo (audio sucio, palabra desconocida), devuelves el token especial `<UNCLEAR>` y NO inventas. El agente pedirá repetición.
- No agregas, no resumes, no editorializas. No agregas saludos ni cortesías que no estén en el original.
- Si el hablante se dirige al intérprete directamente, traduces ESA instrucción al otro idioma como si la dijera él mismo.

**Inyección de terminología:** antes de cada traducción, `terminology/loader.py` selecciona los términos relevantes (match por keywords del transcript + especialidad activa de la llamada) y los inserta en un bloque `<terminology>...</terminology>` dentro del prompt. No volcamos el glosario entero cada vez — solo lo que matchea, para no inflar latencia/tokens.

**Formato de salida:** JSON estricto: `{"target_lang":"es"|"en","text":"...","confidence":0..1,"needs_clarification":bool}`. Como DeepSeek-Flash a veces no respeta JSON tan bien como modelos frontier, **validar con pydantic y reintentar una vez con instrucción reforzada** si el parse falla. Nunca regex.

**Lo que NUNCA hacemos:** dar consejo médico propio, "mejorar" lo dicho, omitir lenguaje incómodo (se traduce tal cual).

---

## 8. Carpeta de terminología — `data/terminology/`

El corazón de la calidad clínica. Glosarios curados que el LLM usa como fuente de verdad.

**Formato de archivos** (CSV, UTF-8, header obligatorio):

```csv
# glossary_en_es.csv
term_en,term_es,notes,specialty
shortness of breath,dificultad para respirar,"síntoma; NO 'corto de aliento'",general
blood pressure,presión arterial,,cardiology
miscarriage,aborto espontáneo,"sensible; tono cuidadoso",obgyn
```

```csv
# abbreviations.csv
abbr,expansion_en,expansion_es,specialty
BP,blood pressure,presión arterial,general
SOB,shortness of breath,dificultad para respirar,general
Rx,prescription,receta médica,general
NPO,nothing by mouth,nada por vía oral,general
```

```csv
# drug_names.csv
brand,generic,name_es,notes
Tylenol,acetaminophen,acetaminofén,"OTC analgésico"
Advil,ibuprofen,ibuprofeno,
Coumadin,warfarin,warfarina,"anticoagulante; precisión en dosis crítica"
```

**Reglas:**
- Todo término sensible (salud mental, embarazo, terminal, etc.) lleva nota de tono en `notes`.
- `specialty` permite cargar solo el glosario de la especialidad de la llamada (campo opcional en el webhook o detectado por el agente).
- `terminology/loader.py` carga al startup, indexa en memoria (dict + índice por keyword) y expone `lookup(transcript, specialty) -> list[term]`.
- Para añadir términos: editar el CSV correspondiente y reiniciar (o hot-reload si lo implementamos). Documentado en `data/terminology/README.md`.
- **Versionado:** los CSV van en git. Cambios de terminología = PR con revisión, igual que código. Un mal término puede causar daño clínico.

---

## 9. ElevenLabs — voz femenina del operador

- **Voice ID:** está en `.env` como `ELEVENLABS_VOICE_ID`. Es la **voz femenina clonada** de la operadora (clon entrenado por separado por cada operadora). Si aún no hay clon propio, usar una **voz femenina pre-hecha de ElevenLabs** en EN/ES como placeholder y dejar el voice_id en config.
- **Modelo:** `eleven_turbo_v2_5` (multilingüe, latencia ~250ms first byte).
- **Streaming endpoint:** `/v1/text-to-speech/{voice_id}/stream` con `output_format=ulaw_8000` (¡crítico! evitamos resampleo del lado nuestro).
- **Optimización de latencia:** `optimize_streaming_latency=3`.
- **Chunking:** ElevenLabs devuelve chunks variables. Re-empaquetamos a frames de 160 bytes (20ms mulaw) antes de mandar a Twilio.
- **Stability/similarity:** stability 0.5, similarity 0.85. Probar con la operadora real antes de producción — clones más estables suenan más planos.
- **Idiomas:** la misma voz femenina debe sonar natural en EN y ES. Verificar que el clon se entrenó con audio bilingüe o al menos que el modelo turbo multilingüe mantiene la identidad de voz en ambos idiomas.

---

## 10. Turn-taking y barge-in

Este es el problema duro. Reglas:

1. **Estado del agente:** `LISTENING_A` | `TRANSLATING` | `SPEAKING_TO_B` | `LISTENING_B` | `TRANSLATING` | `SPEAKING_TO_A`. Máquina de estados explícita, no flags sueltos.
2. **Fin de turno (EOT):** combinación de (a) Deepgram `is_final=true` Y (b) silencio detectado por VAD durante ≥600ms.
3. **Barge-in:** si estamos `SPEAKING_TO_X` y el VAD del lado contrario detecta voz sostenida >300ms, **cortamos el TTS** (mandamos `clear` al stream Twilio), pasamos a `LISTENING`. El hablante manda.
4. **Talkover de la misma persona:** el agente espera 1s, traduce, y responde con frase pre-grabada en la voz femenina ("Un momento, el doctor estaba hablando"), no generada.
5. **Silencio largo:** si pasamos >8s en `LISTENING` sin actividad, el bot pregunta en el idioma del último hablante: "¿Sigue ahí?".

---

## 11. Privacidad, compliance, ética

**Esto es PHI (Protected Health Information). Trátalo como tal aunque aún no estemos certificados.**

- **Nada de logging del contenido de los turnos** en producción. Solo metadatos (duración, idioma, latencia, error codes). Debug con contenido solo bajo flag `DEBUG_TRANSCRIPTS=true` en dev local.
- **No persistir audio.** Buffers en memoria, descartados al final del turno. QA solo con consentimiento explícito grabado y bucket cifrado, retención ≤30 días.
- **BAAs:** Twilio, Deepgram, ElevenLabs ofrecen Business Associate Agreements. **OJO con DeepSeek:** verifica disponibilidad de BAA y residencia de datos antes de cualquier llamada con PHI real — si no hay BAA o el procesamiento es en jurisdicción problemática para tu compliance, usa el modelo solo con datos de prueba o ponle un proveedor de hosting con garantías (ej. endpoint self-hosted / proveedor con acuerdo). Hasta resolver esto: **solo llamadas de prueba con voluntarios informados, sin PHI real.**
- **Disclaimer al inicio de cada llamada:** "Esta llamada está siendo asistida por un intérprete automatizado. Si necesita un intérprete humano, diga 'humano' en cualquier momento." Frase pre-grabada en la voz femenina, EN y ES.
- **Escape word:** "human", "humano", "real person", "persona real" → bot pasa a `ESCALATING`, hace dial al softphone del operador, sale del loop.
- **Emergencias:** keywords de emergencia (chest pain / dolor de pecho, can't breathe / no puedo respirar, suicide / suicidio, etc.) → el bot completa el turno actual y AÑADE en EN para el proveedor: `[Interpreter alert: patient mentioned possible emergency keywords]`. Lista en `app/prompts/emergency_keywords.py`.

---

## 12. Comandos

```bash
# Setup
poetry install
cp .env.example .env  # llenar keys
poetry run python -m app.scripts.verify_credentials  # ping a los proveedores

# Dev
poetry run uvicorn app.main:app --reload --port 8000
ngrok http 8000  # exponer para Twilio webhook

# Tests
poetry run pytest
poetry run pytest -m live          # e2e con APIs reales (cuesta créditos)
poetry run pytest --cov=app --cov-report=term-missing

# Lint / type
poetry run ruff check .
poetry run mypy app/

# Terminología
poetry run python -m app.terminology.loader --validate   # checa formato CSV

# Toggle manual (debug)
curl -X POST localhost:8000/api/toggle
curl localhost:8000/api/status
```

---

## 13. Variables de entorno

```
# Traducción — DeepSeek V4-Flash vía formato Anthropic
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic
DEEPSEEK_MODEL=deepseek-v4-flash
# Fallback de precisión para turnos de alta ambigüedad médica
FALLBACK_API_KEY=
FALLBACK_BASE_URL=
FALLBACK_MODEL=

# STT
DEEPGRAM_API_KEY=
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGES=en,es

# TTS — voz femenina
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=          # clon femenino de la operadora
ELEVENLABS_MODEL=eleven_turbo_v2_5

# Telefonía
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
PUBLIC_BASE_URL=               # ej https://abcd.ngrok.io, sin trailing slash

# Terminología
TERMINOLOGY_DIR=data/terminology
DEFAULT_SPECIALTY=general

DEBUG_TRANSCRIPTS=false
```

---

## 14. Roadmap (corto plazo)

1. **MVP de loop cerrado** con Twilio + stubs.
2. **Glosario base** poblado en `data/terminology/` + loader funcionando.
3. **Traducción real** con DeepSeek V4-Flash + inyección de terminología, sin VAD propio aún.
4. **VAD + turn-taking robusto.**
5. **Voz femenina clonada** de la primera operadora, pruebas con voluntarios bilingües.
6. **Métricas + dashboard** de latencia y errores.
7. **Modo escalation** a humano + resolver compliance/BAA de DeepSeek.
8. **Piloto** con clínica pequeña, llamadas reales con consentimiento (solo tras resolver §11).

---

## 15. Cómo trabajar en este repo (instrucciones para Claude Code)

- Antes de implementar algo nuevo, lee este archivo entero. Si algo aquí está desactualizado respecto al código, **actualiza este archivo en el mismo commit**.
- Cuando crees un servicio nuevo, sigue el patrón de `deepgram_streamer.py`: clase async, `__aenter__`/`__aexit__`, métodos `start()` / `send()` / `close()`, métricas via callback.
- `deepseek_service.py` debe construirse sobre el SDK de Anthropic apuntando a la base URL de DeepSeek — no escribir un cliente HTTP a mano salvo necesidad.
- Para cualquier decisión que afecte latencia (>50ms), mídela con un test antes de mergear.
- Los CSV de `data/terminology/` se tratan como código clínico: cambios por PR con revisión.
- Nunca commits con secretos. `.env` está en `.gitignore`; verifica con `git diff --cached`.
- En PRs incluye: qué cambió, por qué, qué probaste, métricas si aplica.

---

*Última revisión: voz femenina + DeepSeek V4-Flash como motor de traducción + carpeta data/terminology. Estado: scaffold + Deepgram STT hechos. Próximo hito: webhook Twilio + WS Media Stream con loop de eco.*