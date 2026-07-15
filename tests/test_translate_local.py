"""LocalTranslator: ambas direcciones + regresión del fix de _opposite_lang.

No requiere red ni API keys (usa solo el diccionario local de fallback).
"""
from src.translate import LocalTranslator, _opposite_lang


def test_opposite_lang_en_to_es():
    assert _opposite_lang("en") == "es"


def test_opposite_lang_es_to_en():
    assert _opposite_lang("es") == "en"


def test_opposite_lang_none_falls_back_to_detection():
    # Antes del fix: LocalTranslator defaulteaba a "en" y el degrade de
    # DeepSeekTranslator defaulteaba a "es" para el mismo caso -- inconsistente.
    # Ahora ambos usan _opposite_lang, que intenta detectar el idioma real.
    assert _opposite_lang(None, "The patient has a headache") == "es"
    assert _opposite_lang(None, "El paciente tiene dolor de cabeza") == "en"


def test_opposite_lang_never_equals_source():
    for src in ("en", "es", None, "", "EN", "ES"):
        target = _opposite_lang(src, "some text with enough letters")
        assert target in ("en", "es")
        if src and src.lower()[:2] in ("en", "es"):
            assert target != src.lower()[:2]


def test_local_translator_en_to_es_low_confidence_degrades_to_unclear():
    # "The patient has chest pain" tiene pocos términos reconocidos (confidence=0.33,
    # por debajo del umbral 0.5) -- se degrada a <UNCLEAR> en vez de mostrar la
    # sustitución palabra-por-palabra ("The paciente has dolor en el pecho") como si
    # fuera una traducción real.
    t = LocalTranslator()
    result = t.translate("The patient has chest pain", "en", "")
    assert result.target_lang == "es"
    assert result.text == "<UNCLEAR>"
    assert result.needs_clarification is True


def test_local_translator_es_to_en():
    t = LocalTranslator()
    result = t.translate("El paciente tiene dolor de cabeza", "es", "")
    assert result.target_lang == "en"
    assert "patient" in result.text.lower()


def test_local_translator_never_echoes_source_language():
    t = LocalTranslator()
    for text, lang in [
        ("The patient has a fever", "en"),
        ("El paciente tiene fiebre", "es"),
    ]:
        result = t.translate(text, lang, "")
        assert result.target_lang != lang


def test_local_translator_confidence_capped():
    t = LocalTranslator()
    result = t.translate("The patient has chest pain and fever", "en", "")
    assert 0.0 <= result.confidence <= 0.85
