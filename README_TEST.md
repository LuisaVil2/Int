# Intérprete médico — uso

## 🔴 EN VIVO (el bot oye y traduce en tiempo real)
Pa probar con un video de YouTube como lo vas a usar:
```powershell
# 1) Enciende el bot
.\.venv\Scripts\python.exe -m src.live_interpreter
# 2) Reproduce el video de YouTube. El bot lo oye por el audio del sistema (loopback)
#    y va imprimiendo la traducción en vivo. Ctrl+C para apagar.
```
- El video debe sonar por el **dispositivo de salida por defecto** de Windows.
- `--mic` usa micrófono en vez del audio del sistema.
- `--list-devices` lista parlantes/micrófonos.
- `--model medium` = más calidad (más lento). STT = Whisper local, lag ~3-5s.
- Para baja latencia real: conseguir `DEEPGRAM_API_KEY` (streaming).

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
