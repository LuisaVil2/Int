#!/usr/bin/env python3
"""Test translation pipeline with DeepSeek API."""
import os
from dotenv import load_dotenv

load_dotenv('.env')

from src.translate import DeepSeekTranslator
from src.terminology import TerminologyIndex

# Initialize translator
print("Initializing DeepSeekTranslator...")
translator = DeepSeekTranslator()
print(f"✓ Translator initialized with model: {translator.model}")

# Load terminology
print("\nLoading medical terminology...")
idx = TerminologyIndex.load("data/terminology")
print(f"✓ Terminology loaded")

# Test translation
test_text = "The patient has a headache and dizziness."
terminology_block = "patient: paciente, headache: dolor de cabeza, dizziness: mareos"

print(f"\nTesting translation:")
print(f"  Source (EN): {test_text}")
print(f"  Terminology: {terminology_block}")

try:
    result = translator.translate(test_text, "en", terminology_block)
    print(f"\n✓ Translation successful!")
    print(f"  Target language: {result.target_lang}")
    print(f"  Translation (ES): {result.text}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Needs clarification: {result.needs_clarification}")
except Exception as e:
    print(f"\n✗ Translation failed: {e}")
    import traceback
    traceback.print_exc()
