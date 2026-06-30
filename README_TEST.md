# Intérprete médico — uso

## 🟢 GUI (recomendado) — elegir audio y activar el bot
```powershell
.\.venv\Scripts\python.exe -m src.gui
```
1. **Entra (oye):** elige `[Sistema] <tu salida>` para que el bot oiga el video de YouTube
   (loopback), o `[Micrófono] ...`.
2. **Sale (voz):** elige por dónde habla el bot. ⚠ Usa un dispositivo DISTINTO al que oye,
   o se oirá a sí mismo (el modo half-duplex lo mitiga, pero mejor separarlos).
3. Marca/desmarca **Hablar la traducción (voz)**.
4. **▶ Activar bot** → reproduce el video. Verás transcripción + traducción en vivo y la
   voz por el dispositivo elegido. **■ Apagar** para parar.

STT = Deepgram (nova-3, multilingüe) si hay `DEEPGRAM_API_KEY`; si no, Whisper local.
TTS = edge-tts (gratis, voz femenina ES/EN). Sin ElevenLabs.

## 🔴 EN VIVO por consola (sin GUI)
```powershell
.\.venv\Scripts\python.exe -m src.live_interpreter      # texto en vivo, sin voz
```
- `--mic` micrófono · `--list-devices` lista · `--model medium` más calidad.

## Test por video descargado (batch, un comando)
```powershell
.\.venv\Scripts\python.exe -m src.youtube_test "<URL de YouTube>" --max 30
# -> out/result.md con tabla SRC | BOT | route | emerg
```

---

# Test offline del intérprete médico (slice de hoy)

Prueba el núcleo de calidad — STT → terminología → traducción DeepSeek — contra
un video de YouTube de interpretación médica EN↔ES, **sin** Twilio/WS/VAD.

## Qué hay
```
src/
  prompts.py            system prompt médico (§7)
  terminology.py        loader + lookup de glosarios (§8)
  stt.py                STT: Deepgram | faster-whisper | subtítulos VTT
  translate.py          DeepSeek V4-Flash vía SDK Anthropic + pydantic + 1 retry
  test_interpreter.py   orquestador: audio → segmentos → traducción → tabla
data/terminology/       glossary_en_es / abbreviations / drug_names (CSV)
audio/                  audio descargado + subtítulos (gitignored)
out/                    result.md + result.json (gitignored)
```

## Estado
- [x] venv + deps (ffmpeg, yt-dlp, faster-whisper, anthropic, pydantic, deepgram-sdk)
- [x] Video descargado: GERD (Bridging Words) → `audio/gerd.wav` 16k mono
- [x] STT con Whisper `small` → 35 segmentos EN/ES en `out/result.md`
- [ ] **Traducción real: falta `DEEPSEEK_API_KEY` en `.env`** ← único bloqueante

## Cómo terminar el test (1 paso)
1. Pon tu key en `.env`:  `DEEPSEEK_API_KEY=sk-...`
2. Corre:
```powershell
.\.venv\Scripts\python.exe -m src.test_interpreter --audio audio/gerd.wav --max 35
```
Genera `out/result.md` con columnas SRC | BOT | confidence | latencia_ms.

## Opciones
```
--audio <wav>     transcribe con Deepgram (si hay key) o Whisper local
--vtt <archivo>   usa subtítulos en vez de audio
--max N           límite de segmentos (default 30)
--specialty X     filtra glosario por especialidad
```

## STT
- Default Whisper `small` (CPU). Cambia con `WHISPER_MODEL=medium` en `.env` para más calidad.
- Con `DEEPGRAM_API_KEY` usa Nova-3 (más rápido, detección de idioma por utterance).

## Notas
- Video público = material de prueba SIN PHI (cumple §11). No usar PHI real hasta resolver BAA de DeepSeek.
- ffmpeg instalado por winget; ruta absoluta en `.env` (`FFMPEG_PATH`).
