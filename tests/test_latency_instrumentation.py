"""Verifica que los eventos emitidos por _process_one llevan los campos de
instrumentación de latencia por etapa (stt_ms, latency_ms, etc.), sin red/audio real."""
import types

import numpy as np

from src.confidence import ConfidenceEngine
from src.emergency import EmergencyClassifier
from src.live_engine import LiveEngine
from src.memory import ConversationMemory


class _FakeTranslation:
    def __init__(self, text="hola", target_lang="es", confidence=0.9, needs_clarification=False):
        self.text = text
        self.target_lang = target_lang
        self.confidence = confidence
        self.needs_clarification = needs_clarification


class _FakeTranslator:
    def translate(self, text, lang, term_block, context):
        return _FakeTranslation()


class _FakeIdx:
    def lookup(self, text, specialty, limit=25):
        return ["- fever = fiebre"]


def _detect_lang(text, fallback=None):
    return "en"


def test_events_carry_latency_and_speaker_fields(monkeypatch):
    events = []

    def on_event(kind, data):
        events.append((kind, data))

    engine = LiveEngine(on_event=on_event, input_choice={"kind": "mic", "name": "dummy"},
                        output_index=None, tts_on=False)

    def fake_transcribe(audio, dg_key):
        return "I have a fever", "en", 0.92, 1, "deepgram"

    monkeypatch.setattr(engine, "_transcribe", fake_transcribe)

    engine._process_one(
        audio=np.zeros(10, dtype=np.float32),
        dg_key="fake",
        translator=_FakeTranslator(),
        idx=_FakeIdx(),
        detect_lang=_detect_lang,
        memory=ConversationMemory(),
        emerg=EmergencyClassifier(),
        confidence_engine=ConfidenceEngine(),
    )

    src_events = [d for k, d in events if k == "src"]
    tr_events = [d for k, d in events if k == "translation"]
    assert src_events and src_events[0]["speaker"] == 1

    assert tr_events, "expected a translation event"
    data = tr_events[0]
    for key in ("latency_ms", "stt_ms", "stt_backend", "speaker",
                "confidence_score", "route"):
        assert key in data
    assert data["stt_backend"] == "deepgram"
    assert data["speaker"] == 1
    assert isinstance(data["confidence_score"], int)


def test_emergency_utterance_forces_non_automatic_route(monkeypatch):
    events = []
    engine = LiveEngine(on_event=lambda k, d: events.append((k, d)),
                        input_choice={"kind": "mic", "name": "dummy"}, output_index=None,
                        tts_on=False)

    def fake_transcribe(audio, dg_key):
        return "I have chest pain and can't breathe", "en", 1.0, None, "deepgram"

    monkeypatch.setattr(engine, "_transcribe", fake_transcribe)

    class _PerfectTranslator:
        def translate(self, text, lang, term_block, context):
            return _FakeTranslation(confidence=1.0, target_lang="en")

    engine._process_one(
        audio=np.zeros(10, dtype=np.float32), dg_key="fake",
        translator=_PerfectTranslator(), idx=_FakeIdx(), detect_lang=_detect_lang,
        memory=ConversationMemory(), emerg=EmergencyClassifier(),
        confidence_engine=ConfidenceEngine(),
    )

    tr_events = [d for k, d in events if k == "translation"]
    assert tr_events[0]["route"] != "automatic_approval"
    review_events = [d for k, d in events if k == "needs_review"]
    # con confianza perfecta pero force_qa_review, la ruta sube a qa_review (no pausa),
    # así que no necesariamente hay needs_review -- pero la traducción SÍ trae el alert.
    assert "Interpreter alert" in tr_events[0]["text"]
