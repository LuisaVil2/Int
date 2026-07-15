#!/usr/bin/env python3
"""Test fallback translation when DeepSeek returns 402."""
import os
from dotenv import load_dotenv

load_dotenv('.env')

from src.translate import DeepSeekTranslator, LocalTranslator

print("=" * 60)
print("TESTING LOCAL TRANSLATOR (Fallback)")
print("=" * 60)

local = LocalTranslator()

test_cases = [
    ("The patient has a headache and fever", "en"),
    ("El paciente tiene dolor en el pecho", "es"),
    ("Shortness of breath and chest pain", "en"),
    ("Falta de aliento y náusea", "es"),
]

for text, lang in test_cases:
    result = local.translate(text, lang, "")
    print(f"\nSource ({lang}): {text}")
    print(f"Target ({result.target_lang}): {result.text}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Needs clarification: {result.needs_clarification}")

print("\n" + "=" * 60)
print("TESTING DEEPSEEK TRANSLATOR WITH FALLBACK")
print("=" * 60)

try:
    translator = DeepSeekTranslator()
    
    test_text = "The patient has a headache."
    terminology_block = "patient: paciente, headache: dolor de cabeza"
    
    print(f"\nAttempting DeepSeek translation...")
    print(f"Source: {test_text}")
    
    result = translator.translate(test_text, "en", terminology_block)
    
    print(f"✓ Translation completed")
    print(f"Target: {result.text}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Needs clarification: {result.needs_clarification}")
    print(f"Using fallback: {translator._use_fallback}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nNote: This is expected if DeepSeek account has no balance.")
    print("The fallback translator is now active and will handle all translations.")
