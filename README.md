# Intérprete Médico EN↔ES — Bot de interpretación en vivo

Bot intérprete médico bilingüe **inglés↔español** que escucha audio en tiempo real
(p. ej. un video de YouTube o una llamada), lo transcribe, lo traduce con contexto
clínico y lo **habla** con voz femenina. Pensado para asistir interpretación médica
consecutiva.

> Especificación completa del producto en [`Instrucciones.md`](Instrucciones.md).
> Hallazgos y mejoras de los tests en [`FINDINGS.md`](FINDINGS.md).
> Qué se rescató del código previo en [`SALVAGE.md`](SALVAGE.md).

Pipeline: **captura de audio → STT (Deepgram nova-3 / Whisper) → traducción (DeepSeek V4-Flash) → TTS (edge-tts)**, con inyección de terminología médica, memoria conversacional, detección de emergencias y ruteo de calidad.

---

## 1. Requisitos

- **Windows** (probado en Win 11) — la captura de audio del sistema usa WASAPI loopback.
- **Python 3.11+** (probado en 3.13).
- **ffmpeg** (para decodificar audio).
- **Claves de API**: DeepSeek (traducción) y Deepgram (STT). El TTS (edge-tts) es gratis, sin clave.

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
notepad .env        # rellenar DEEPSEEK_API_KEY y DEEPGRAM_API_KEY
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
1. **Entra (oye):** elige `[Sistema] Altavoces (...)` por donde suena el video de YouTube
   (loopback), o `[Micrófono] ...`.
2. **Sale (voz):** elige un dispositivo **distinto** al de entrada (ej. audífonos) para que
   el bot no se oiga a sí mismo.
3. (Opcional) destilda **Hablar la traducción** si solo quieres texto.
4. Click **▶ Activar bot** y reproduce el video.
5. Verás transcripción + traducción en vivo y oirás la voz. **■ Apagar** para parar.

⚠ El modo half-duplex mitiga el eco, pero conviene usar entrada y salida separadas.

---

## 4. Otras formas de correrlo

```powershell
# En vivo por consola (texto, sin GUI)
.\.venv\Scripts\python.exe -m src.live_interpreter
.\.venv\Scripts\python.exe -m src.live_interpreter --mic          # usar micrófono
.\.venv\Scripts\python.exe -m src.live_interpreter --list-devices # listar dispositivos

# Test batch de un video de YouTube (descarga + traduce -> out/result.md)
.\.venv\Scripts\python.exe -m src.youtube_test "<URL de YouTube>" --max 30

# Test offline sobre un audio ya descargado
.\.venv\Scripts\python.exe -m src.test_interpreter --audio audio/clip.wav --max 30
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
  translate.py          DeepSeek V4-Flash (formato Anthropic) + JSON + reintento
  segmentation.py       normaliza turnos monolingües (fix code-switching)
  tts.py                TTS gratis con edge-tts (voz femenina ES/EN)
  terminology.py        carga e indexa glosarios médicos
  memory.py             contexto conversacional para el LLM
  emergency.py          detección de keywords de emergencia (EN/ES)
  confidence.py         confianza multi-señal + ruteo de calidad
data/terminology/       glosarios médicos CSV (EN↔ES, abreviaturas, fármacos)
```

---

## 6. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Cubren los módulos puros (sin red): segmentación/idioma, parser JSON + streaming
incremental del LLM, splitter de frases para TTS, emergencias, confianza, memoria,
terminología y utilidades de STT (VTT/WAV).

---

## 7. Modelos y costos

| Componente | Por defecto | Nota |
|---|---|---|
| Traducción | DeepSeek V4-Flash (`deepseek-v4-flash`) vía formato Anthropic | barato, non-thinking |
| STT | Deepgram `nova-3` + `language=multi` | si no hay clave, usa Whisper local (gratis, más lento) |
| TTS | edge-tts (voces `es-MX-DaliaNeural` / `en-US-AriaNeural`) | **gratis, sin clave**; swappeable a Piper/MeloTTS |

---

## 8. Privacidad

Esto maneja información médica sensible (PHI). **No usar con datos de pacientes reales**
hasta resolver acuerdos de tratamiento de datos (BAA) — ver §11 de `Instrucciones.md`.
Para pruebas, usar solo audio público (videos de YouTube) o voluntarios informados.
