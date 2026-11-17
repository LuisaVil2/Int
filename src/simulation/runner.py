"""Run a scripted conversation through the existing transcript processing path."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4

from ..confidence import ConfidenceEngine
from ..emergency import EmergencyClassifier
from ..live_engine import LiveEngine
from ..memory import ConversationMemory
from ..stt import Segment
from ..terminology import TerminologyIndex
from ..translate import DeepSeekTranslator, Translation
from .qa import check_turn
from .schemas import (ConfidenceEvidence, Interpretation, OPPOSITE, Scenario, Session,
                      SeverityPolicy, StageMark, Suggestions, TurnResult)

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "simulation" / "scenarios.json"
GLOSSARY_SPECIALTIES = {
    "primary_care": "internal_medicine",
    "emergency_medicine": "emergency",
    "respiratory": "internal_medicine",
    "obstetrics": "obgyn",
    "mental_health": "psychiatry",
}


class TranscriptSTT:
    """Supply a finalized transcript and an explicit fixture language hypothesis."""

    def transcribe(self, utterance, index: int) -> Segment:
        # Scripted speaker IDs are independent of ASR metadata.
        return Segment(index, utterance.text, utterance.asr_language, asr_confidence=1.0)


class FixtureTranslator:
    """Replay authored candidates, including intentionally faulty responses."""

    def __init__(self, utterance):
        self.utterance = utterance

    def translate(self, text, lang, terminology_block, context):
        reply = self.utterance.mock_reply
        if reply is None or text != self.utterance.text:
            return Translation(target_lang=OPPOSITE[lang], text="<UNCLEAR>", needs_clarification=True)
        return Translation(text=reply.text, target_lang=reply.target_language,
                           confidence=reply.confidence)


class RecordedTranslator:
    """Capture the adapter response before the engine decides whether to hold it."""

    def __init__(self, translator, role=None):
        self.translator = translator
        self.role = role
        self.attempts: list[Interpretation] = []

    def translate(self, *args):
        result = self.translator.translate(*args)
        self.attempts.append(Interpretation(text=result.text, target_language=result.target_lang,
                                            role=self.role))
        return result


def run_scenario(scenario: Scenario, *, adapter="fixture", allow_network=False,
                 measure_latency=False, audio="none", policy: SeverityPolicy | None = None,
                 suggestions: Suggestions | None = None) -> Session:
    if adapter not in {"fixture", "deepseek"} or audio not in {"none", "mock", "edge"}:
        raise ValueError("Unknown simulation adapter or audio mode")
    if (adapter == "deepseek" or audio == "edge") and not allow_network:
        raise ValueError("External providers require --allow-network")
    if suggestions and (suggestions.scenario_id != scenario.id
                        or {s.turn_id for s in suggestions.suggestions} - {t.id for t in scenario.turns}):
        raise ValueError("Model suggestions do not match this scenario and its turns")

    policy = policy or SeverityPolicy()
    stt = TranscriptSTT()
    cloud = DeepSeekTranslator() if adapter == "deepseek" else None
    events = []
    marks = []
    trace = (lambda *items: marks.append(items)) if measure_latency else None
    engine = LiveEngine(lambda kind, data: events.append((kind, data)), {}, None,
                        tts_on=False, specialty=GLOSSARY_SPECIALTIES.get(
                            scenario.specialty, scenario.specialty), trace=trace)
    glossary = TerminologyIndex.load(os.getenv("TERMINOLOGY_DIR") or
                                     DEFAULT_CORPUS.parents[1] / "terminology")
    memory = ConversationMemory()
    emergency = EmergencyClassifier()
    confidence = ConfidenceEngine()
    roles = {speaker.id: speaker.role for speaker in scenario.speakers}
    rows = []
    for index, turn in enumerate(scenario.turns):
        events.clear()
        marks.clear()
        role = roles[turn.speaker_id]
        target = OPPOSITE[turn.source_language]
        recorded_role = turn.mock_reply.reported_role if cloud is None and turn.mock_reply else None
        recorded = RecordedTranslator(cloud or FixtureTranslator(turn), recorded_role)
        started = perf_counter_ns() if measure_latency else None

        if turn.event == "interruption":
            # A deterministic capture gap. Do not interpret a truncated source fragment.
            reasons = ("simulated_capture_interruption",)
        else:
            engine.process_transcript(stt.transcribe(turn, index), recorded, glossary, memory,
                                      emergency, confidence, stt_backend="fixture-transcript")
            reasons = tuple(data["reason"] for kind, data in events if kind == "needs_review")

        elapsed = (perf_counter_ns() - started) / 1e6 if started is not None else None
        accepted = any(kind == "translation" for kind, _ in events) and not reasons
        attempts = tuple(recorded.attempts)
        rows.append(TurnResult(
            id=turn.id, speaker_id=turn.speaker_id, role=role, source_language=turn.source_language,
            target_language=target, event=turn.event, source=turn.text, reference=turn.reference,
            review_expected=turn.review_expected, interpretations=attempts,
            disposition="accepted" if accepted else "held", review_reasons=reasons,
            confidence_evidence=tuple(ConfidenceEvidence.model_validate(
                {k: v for k, v in data.items() if k != "turn_id"})
                for kind, data in events if kind == "confidence_evidence"),
            stage_marks=tuple(StageMark(stage=stage, offset_ms=(ns - started) / 1e6, engine_turn=engine_id)
                              for stage, ns, engine_id in marks) if started is not None else (),
            flags=check_turn(turn, attempts, target, role, accepted=accepted, policy=policy),
            model_suggestions=tuple(item for item in suggestions.suggestions if item.turn_id == turn.id)
            if suggestions else (), elapsed_ms=elapsed,
        ))

    return Session(id=uuid4(), created_at=datetime.now(timezone.utc), scenario_id=scenario.id,
                   title=scenario.title, specialty=scenario.specialty, difficulty=scenario.difficulty,
                   adapter=adapter, audio=audio, severity_policy=policy, turns=tuple(rows))
