import json

import pytest
from pydantic import ValidationError

from src.translate import TextFieldStream, Translation, _parse


# ---------- _parse ----------
def test_parse_plain_json():
    assert _parse('{"target_lang": "es", "text": "hola"}') == {"target_lang": "es",
                                                               "text": "hola"}


def test_parse_json_fence():
    raw = '```json\n{"target_lang": "en", "text": "hi"}\n```'
    assert _parse(raw) == {"target_lang": "en", "text": "hi"}


def test_parse_with_surrounding_prose():
    raw = 'Aquí está: {"target_lang": "en", "text": "hi"} espero que sirva'
    assert _parse(raw) == {"target_lang": "en", "text": "hi"}


def test_parse_garbage_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse("no json aquí")


# ---------- Translation ----------
def test_translation_normalizes_lang():
    assert Translation(target_lang=" ES ", text="x").target_lang == "es"


def test_translation_rejects_bad_lang():
    with pytest.raises(ValidationError):
        Translation(target_lang="fr", text="x")


# ---------- TextFieldStream ----------
PAYLOAD = {
    "target_lang": "es",
    "text": 'Hola, señor. Tiene "ERGE".\nSiga el tratamiento.',
    "confidence": 0.97,
    "needs_clarification": False,
}


@pytest.mark.parametrize("size", [1, 3, 7, 50, 1000])
def test_textfieldstream_arbitrary_chunks(size):
    full = json.dumps(PAYLOAD, ensure_ascii=True)  # escapes \uXXXX, \n, \"
    tfs = TextFieldStream()
    got = "".join(tfs.feed(full[i:i + size]) for i in range(0, len(full), size))
    assert got == PAYLOAD["text"]


def test_textfieldstream_no_text_field_yet():
    tfs = TextFieldStream()
    assert tfs.feed('{"target_lang": "es", "confidence"') == ""


def test_textfieldstream_stops_after_close_quote():
    tfs = TextFieldStream()
    out = tfs.feed('{"text": "corto", "confidence": 0.9}')
    assert out == "corto"
    assert tfs.feed(' basura posterior "text": "otro"') == ""
