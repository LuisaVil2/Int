"""Regression fixtures and reference translations are fictional."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.emergency import EmergencyClassifier
from src.segmentation import detect_lang, normalize_turns, split_sentences
from src.stt import Segment
from src.translate import DeepSeekTranslator, fidelity_issue

CASES = json.loads((Path(__file__).parent / "fixtures" / "fictional_dialogues.json").read_text(
    encoding="utf-8"))


def mock_translator(call):
    translator = DeepSeekTranslator.__new__(DeepSeekTranslator)
    translator._call = call
    return translator


def response(target, text, **kwargs):
    return json.dumps(dict(target_lang=target, text=text, confidence=0.99,
                           needs_clarification=False) | kwargs)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_fictional_dialogues(case):
    source, lang = case["source"], case["lang"]
    if "directions" in case:
        turns = normalize_turns([Segment(0, source, lang)])
        assert [turn.lang for turn in turns] == case["directions"]
        assert " ".join(turn.text for turn in turns) == source
        return
    assert detect_lang(source) == lang
    if lang is None:
        tr = mock_translator(lambda *args: pytest.fail("unknown direction made an API call"))
        assert tr.translate(source, lang, "").needs_clarification
        return
    target_lang = "es" if lang == "en" else "en"
    reference = case["target"]
    tr = mock_translator(lambda *_: response(target_lang, reference))
    assert tr.translate(source, lang, "").text == reference
    for bad in case.get("reject", []):
        assert fidelity_issue(source, bad)
    if keyword := case.get("keyword"):
        assert keyword in EmergencyClassifier().classify(source)["labels"]


def test_sentence_split_preserves_decimal_abbreviation_and_unpunctuated_clause():
    assert split_sentences("Dr. Lee, tomé 2.5 mg ¿Es correcto? Sí.") == [
        "Dr. Lee, tomé 2.5 mg", "¿Es correcto?", "Sí."]
    assert detect_lang("Je ne prends aucun médicament.") is None


def test_wrong_dose_first_attempt_exposes_only_validated_retry():
    replies = iter([response("es", "Tomé 15 mg de warfarina."),
                    response("es", "Tomé 50 mg de warfarina.")])
    translator = mock_translator(lambda *_: next(replies))
    delivered = []
    result = translator.translate("I took 50 mg of warfarin.", "en", "", on_text=delivered.append)
    assert delivered == [result.text] == ["Tomé 50 mg de warfarina."]


@pytest.mark.parametrize("raw", [
    '{"target_lang":"es","text":"Tengo dolor.", "confidence":0.9',
    '{"target_lang":"es","text":"Tengo dolor.", "confidence":0.9}',
    response("es", "Tengo dolor.", needs_clarification=True),
    response("es", "Tengo dolor.", confidence=float("nan")),
    response("es", "Tengo dolor.", confidence=-0.01),
    response("es", "I have pain."),
])
def test_malformed_or_uncertain_translation_stays_silent(raw):
    translator = mock_translator(lambda *_: raw)
    delivered = []
    result = translator.translate("I have pain.", "en", "", on_text=delivered.append)
    assert result.needs_clarification and result.text == "<UNCLEAR>"
    assert delivered == []


def test_transport_failure_does_not_use_dictionary_or_log_payload(caplog, monkeypatch):
    from src.logging_utils import log_turn
    import logging

    private_fixture = "mock-provider-response-must-not-enter-log"

    def fail(*args):
        raise RuntimeError(private_fixture)

    translator = mock_translator(fail)
    result = translator.translate("Fever pain nausea.", "en", "")
    assert result.text == "<UNCLEAR>"
    monkeypatch.setenv("DEBUG_TRANSCRIPTS", "true")
    with caplog.at_level(logging.INFO):
        log_turn(logging.getLogger("src"), session_id="fixture", turn_id=1,
                 source_text=private_fixture, output_text=private_fixture)
    assert private_fixture not in caplog.text


def test_provider_stop_reason_must_indicate_completion():
    translator = DeepSeekTranslator.__new__(DeepSeekTranslator)
    message = SimpleNamespace(stop_reason="max_tokens", content=[
        SimpleNamespace(type="text", text=response("es", "Tengo dolor."))])
    translator.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: message))
    with pytest.raises(ValueError, match="translation_incomplete"):
        translator._request({}, None)
