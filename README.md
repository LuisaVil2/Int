# Intérprete médico EN↔ES — prototipo de simulación

Motor Python de interpretación consecutiva experimental. La GUI y la consola comparten
captura, STT, segmentación, traducción y revisión. La GUI puede hablar las traducciones
que pasan el ruteo y las comprobaciones del turno completo.

- [Auditoría de estabilización, resultados de pruebas y plan de fase 2](ENGINE_AUDIT.md)
- [Especificación histórica](Instrucciones.md), [hallazgos históricos](FINDINGS.md) y [código rescatado](SALVAGE.md)

Pipeline: **captura → Deepgram WebSocket o Whisper local → dirección EN↔ES → DeepSeek
completo → validación y ruteo → síntesis opcional**.

---

## 1. Requisitos

- **Windows** (probado en Win 11) — la captura de audio del sistema usa WASAPI loopback.
- **Python 3.11+** (probado en 3.13).
- **ffmpeg** (para decodificar audio).
- **Claves de API**: DeepSeek para traducir y Deepgram opcional para STT en vivo. Fish Audio requiere credenciales y una voz ya configurada. Edge-tts usa voces estándar.

---

## 2. Instalación (desde cero)

```powershell
# 1) Clonar
git clone https://github.com/LuisaVil2/Int.git
cd Int

# 2) Instalar ffmpeg (si no lo tienes)
winget install -e --id Gyan.FFmpeg

# 3) Crear entorno virtual e instalar dependencias
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4) Configurar claves
copy .env.example .env
notepad .env        # DEEPSEEK_API_KEY y, para streaming, DEEPGRAM_API_KEY
```

En `.env` ajusta también `FFMPEG_PATH` con la ruta a `ffmpeg.exe` si no quedó en el PATH
(winget lo instala en `...\WinGet\Packages\Gyan.FFmpeg_...\ffmpeg-*\bin\ffmpeg.exe`).

> `.env` y `API.txt` están en `.gitignore` — las claves **nunca** se suben al repo.

---

## 3. Correr el bot (GUI — recomendado)

```powershell
.\.venv\Scripts\python.exe -m src.gui
```

En la ventana:
1. Elige la entrada. Puede ser audio del sistema por loopback o un micrófono.
2. Si activas **Hablar la traducción**, elige audífonos o una salida distinta de la capturada.
3. Activa el bot y usa únicamente un diálogo ficticio. Si Whisper calibra el micrófono
   por energía, espera el aviso de escucha.
4. El panel muestra la transcripción, traducciones aceptadas por las reglas del motor
   y borradores marcados como **Revisión requerida**. Pide repetir o revisa manualmente
   los turnos retenidos. No hay botón de aprobación para reproducir un borrador.
5. Pulsa **Apagar** o cierra la ventana para detener la sesión.

La reproducción pausa la captura útil, con 150 ms de margen para el eco. Puede perderse
el habla durante esa ventana. La interfaz indica cuándo repetir. No se recuperan
frames descartados por reconexión ni audio superpuesto a la reproducción.

La comprobación de dirección aplica indicios del texto y, cuando corresponde, la etiqueta
del mismo tramo de ASR. Un «No» sin indicio de idioma requiere revisión. El motor acumula
resultados finales de Deepgram y usa sus señales de fin de turno. Las etiquetas de
diarización solo indican separación local. La interfaz muestra coincidencias de
palabras de emergencia como motivo de revisión, incluso con síntomas negados.

---

## 4. Otras formas de correrlo

```powershell
# En vivo por consola (texto, sin GUI)
.\.venv\Scripts\python.exe -m src.live_interpreter
.\.venv\Scripts\python.exe -m src.live_interpreter --mic          # usar micrófono
.\.venv\Scripts\python.exe -m src.live_interpreter --list-devices # listar dispositivos

# Prueba batch solo con una grabación ficticia
.\.venv\Scripts\python.exe -m src.test_interpreter --audio audio/simulacion.wav --max 30
```

---

## 5. Estructura

```
src/
  gui.py                GUI (Tkinter): elegir audio in/out, activar/apagar, panel en vivo
  live_engine.py        motor en vivo: captura half-duplex -> STT -> DeepSeek -> TTS
  live_interpreter.py   versión en vivo por consola
  youtube_test.py       test de un comando desde una URL de YouTube
  test_interpreter.py   pipeline batch sobre audio/subtítulos
  stt.py                STT: Deepgram (REST, nova-3 multi) / Whisper / VTT
  translate.py          DeepSeek Flash (formato Anthropic) + JSON + reintento
  segmentation.py       divide cambios de idioma y conserva turnos inciertos
  tts/                  Fish Audio configurado y respaldo edge-tts
  terminology.py        carga e indexa glosarios médicos
  memory.py             contexto conversacional para el LLM
  emergency.py          detección de keywords de emergencia (EN/ES)
  confidence.py         confianza multi-señal + ruteo de calidad
data/terminology/       glosarios médicos CSV (EN↔ES, abreviaturas, fármacos)
```

---

## 6. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

La suite configurada por `pytest.ini` cubre módulos y regresiones deterministas sin
llamar a proveedores ni abrir dispositivos. Los diálogos ficticios y las traducciones de
referencia están en `tests/fixtures/fictional_dialogues.json`. Las respuestas simuladas
ejercitan los controles y las rutas. La calidad del modelo requiere otra evaluación.

---

## 7. Proveedores y modelos

| Componente | Por defecto | Nota |
|---|---|---|
| Traducción | DeepSeek Flash (`deepseek-flash`) vía formato Anthropic | modo sin razonamiento explícito por defecto |
| STT | Deepgram `nova-3` + `language=multi` | si se inicia sin clave, usa Whisper local. Una caída de streaming activa reconexión y revisión |
| TTS | Fish Audio si está configurado, con edge-tts como respaldo | voces estándar `es-MX-DaliaNeural` y `en-US-AriaNeural` |

---

## 8. Privacidad

Usa solo texto y audio ficticios. Deepgram recibe el audio si está configurado.
DeepSeek recibe el turno, la terminología relevante y hasta seis turnos aceptados de
contexto en vivo. El proveedor TTS recibe el texto aceptado. Los logs del motor
contienen metadatos, sin transcripciones, borradores, claves ni mensajes de excepción
del proveedor. La GUI y la consola muestran el texto para la simulación, y el
programa batch escribe texto en `out/`. Esos destinos también son solo para simulación.

Los filtros revisan formato, dirección, cifras explícitas, unidades de dosis comunes y
algunos marcadores de negación. Las comprobaciones pueden retener traducciones válidas
y dejar pasar omisiones, adiciones o errores de sentido. El puntaje de QA es heurístico.
Consulta los límites y el plan de evaluación humana en [ENGINE_AUDIT.md](ENGINE_AUDIT.md).
