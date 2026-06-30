"""Test de un toque: le das una URL de YouTube y corre todo el intérprete.

Uso:
    python -m src.youtube_test "https://www.youtube.com/watch?v=XXXX"
    python -m src.youtube_test "<url>" --name diabetes --max 20

Hace: descarga audio (yt-dlp+ffmpeg) -> STT -> turnos -> DeepSeek -> out/result.md
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="URL del video de YouTube")
    ap.add_argument("--name", default="clip", help="nombre base del archivo de audio")
    ap.add_argument("--max", type=int, default=40, help="máx turnos a traducir")
    args = ap.parse_args()

    py = sys.executable
    ff = os.getenv("FFMPEG_PATH", "")
    ffdir = str(Path(ff).parent) if ff and Path(ff).exists() else None
    Path("audio").mkdir(exist_ok=True)
    audio = f"audio/{args.name}.wav"

    # Solo audio: el STT (Whisper/Deepgram) trabaja sobre el wav. NO bajamos subtítulos
    # (evita HTTP 429 de YouTube y no se necesitan).
    dl = [py, "-m", "yt_dlp", "-x", "--audio-format", "wav", "--no-playlist",
          "--postprocessor-args", "-ar 16000 -ac 1",
          "-o", f"audio/{args.name}.%(ext)s", args.url]
    if ffdir:
        dl += ["--ffmpeg-location", ffdir]

    print("== 1/2 Descargando audio de YouTube ==")
    subprocess.run(dl, check=True)

    print("\n== 2/2 Corriendo intérprete médico ==")
    subprocess.run([py, "-m", "src.test_interpreter", "--audio", audio,
                    "--max", str(args.max)], check=True)

    print("\n✅ Listo. Resultado completo en:  out/result.md  (y out/result.json)")


if __name__ == "__main__":
    main()
