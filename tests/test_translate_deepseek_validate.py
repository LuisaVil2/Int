"""DeepSeekTranslator._validate: invariantes anti-echo, sin llamadas de red reales."""
import json

import pytest

from src.translate import DeepSeekTranslator


@pytest.fixture
def translator(monkeypatch):
    # Evita construir un cliente Anthropic real / exigir DEEPSEEK_API_KEY.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-used")
    return DeepSeekTranslator()


def test_validate_accepts_well_formed_opposite_language(translator):
    raw = json.dumps({"target_lang": "es", "text": "Hola, tengo dolor",
                      "confidence": 0.9, "needs_clarification": False})
    result = translator._validate(raw, "en")
    assert result is not None
    assert result.target_lang == "es"


def test_validate_rejects_target_equal_to_source_label(translator):
    # El LLM devolvió target_lang="en" para un turno de origen "en" -- eco de idioma.
    raw = json.dumps({"target_lang": "en", "text": "I have pain",
                      "confidence": 0.9, "needs_clarification": False})
    result = translator._validate(raw, "en")
    assert result is None


def test_validate_rejects_text_language_mismatch(translator):
    # target_lang dice "es" pero el texto real está en inglés -- LLM mintió sobre el idioma.
    raw = json.dumps({"target_lang": "es", "text": "The patient has severe chest pain today",
                      "confidence": 0.9, "needs_clarification": False})
    result = translator._validate(raw, "en")
    assert result is None


def test_validate_rejects_malformed_json(translator):
    assert translator._validate("not json at all", "en") is None


def test_validate_allows_none_source_lang(translator):
    raw = json.dumps({"target_lang": "es", "text": "Hola, tengo fiebre",
                      "confidence": 0.9, "needs_clarification": False})
    result = translator._validate(raw, None)
    assert result is not None
