#!/usr/bin/env python3
import os
import sys

from dotenv import load_dotenv

# Consolas Windows con cp1252 no soportan ✓/✗
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv('.env')

env_vars = {
    'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
    'DEEPSEEK_BASE_URL': os.getenv('DEEPSEEK_BASE_URL'),
    'DEEPSEEK_MODEL': os.getenv('DEEPSEEK_MODEL'),
    'DEEPGRAM_API_KEY': os.getenv('DEEPGRAM_API_KEY'),
    'DEEPGRAM_MODEL': os.getenv('DEEPGRAM_MODEL'),
    'DEEPGRAM_LIVE_MODEL': os.getenv('DEEPGRAM_LIVE_MODEL'),
    'FISH_API_KEY': os.getenv('FISH_API_KEY'),
    'FISH_VOICE_ID': os.getenv('FISH_VOICE_ID'),
    'FFMPEG_PATH': os.getenv('FFMPEG_PATH'),
    'TERMINOLOGY_DIR': os.getenv('TERMINOLOGY_DIR'),
    'DEFAULT_SPECIALTY': os.getenv('DEFAULT_SPECIALTY'),
    'TTS_VOICE_ES': os.getenv('TTS_VOICE_ES'),
    'TTS_VOICE_EN': os.getenv('TTS_VOICE_EN'),
}

for key, value in env_vars.items():
    if value:
        print(f"{key:<25} ✓ SET")
    else:
        print(f"{key:<25} ✗ EMPTY")
