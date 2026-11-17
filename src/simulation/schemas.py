"""Versioned local records shared by the simulation runner and reviewer."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Language = Literal["en", "es"]
LanguageHint = Literal["en", "es", "mixed"]
Role = Literal["clinician", "patient"]
Difficulty = Literal["basic", "intermediate", "advanced"]
Specialty = Literal["primary_care", "emergency_medicine", "cardiology", "respiratory",
                    "obstetrics", "pediatrics", "medication_reconciliation", "allergies",
                    "informed_consent", "mental_health"]
Event = Literal["speech", "interruption", "short_reply", "correction", "clarification", "code_switch"]
Category = Literal["omission", "addition", "mistranslation", "negation_reversal",
                   "medication_or_dose", "numbers_or_units", "speaker_attribution",
                   "wrong_direction", "unjustified_inference", "uncertainty_or_failure"]
Severity = Literal["minor", "major", "critical"]
CATEGORIES = ("omission", "addition", "mistranslation", "negation_reversal",
              "medication_or_dose", "numbers_or_units", "speaker_attribution",
              "wrong_direction", "unjustified_inference", "uncertainty_or_failure")
OPPOSITE = {"en": "es", "es": "en"}

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")]
UtteranceText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Speaker(Record):
    id: Identifier
    role: Role
    language: Language

    @model_validator(mode="after")
    def role_language(self):
        if self.language != ("en" if self.role == "clinician" else "es"):
            raise ValueError("A clinician speaks English and a patient speaks Spanish")
        return self


class MockReply(Record):
    text: UtteranceText
    target_language: Language
    confidence: Confidence = 0.99
    reported_role: Role | None = None


class Utterance(Record):
    id: Identifier
    speaker_id: Identifier
    text: UtteranceText
    source_language: Language
    asr_language: LanguageHint | None = None
    event: Event = "speech"
    reference: UtteranceText | None = None
    required_target_terms: tuple[Annotated[tuple[UtteranceText, ...], Field(min_length=1)], ...] = ()
    required_target_units: tuple[Annotated[tuple[UtteranceText, ...], Field(min_length=1)], ...] = ()
    review_expected: bool = False
    mock_reply: MockReply | None = None

    @model_validator(mode="after")
    def reference_or_hold(self):
        if self.reference is None and not self.review_expected:
            raise ValueError("A turn needs a reference or an explicit expected review")
        return self


class Scenario(Record):
    schema_version: Literal[1] = 1
    fictional: Literal[True]
    id: Identifier
    title: UtteranceText
    specialty: Specialty
    difficulty: Difficulty
    speakers: Annotated[tuple[Speaker, ...], Field(min_length=2, max_length=2)]
    turns: Annotated[tuple[Utterance, ...], Field(min_length=2, max_length=40)]

    @model_validator(mode="after")
    def identities(self):
        speakers = {speaker.id: speaker for speaker in self.speakers}
        if set(s.role for s in speakers.values()) != {"clinician", "patient"}:
            raise ValueError("A scenario needs one clinician and one patient with distinct IDs")
        if len({turn.id for turn in self.turns}) != len(self.turns):
            raise ValueError("Turn IDs must be unique")
        if {turn.speaker_id for turn in self.turns} != set(speakers):
            raise ValueError("Both scripted speakers must appear")
        for turn in self.turns:
            if turn.source_language != speakers[turn.speaker_id].language:
                raise ValueError("Turn source language must match the intended speaker language")
        return self


class Corpus(Record):
    schema_version: Literal[1] = 1
    scenarios: tuple[Scenario, ...]

    @model_validator(mode="after")
    def unique_ids(self):
        if len({s.id for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("Scenario IDs must be unique")
        return self


class SeverityPolicy(Record):
    schema_version: Literal[1] = 1
    minor: tuple[Category, ...] = ()
    critical: tuple[Category, ...] = ("negation_reversal", "medication_or_dose", "wrong_direction")

    @model_validator(mode="after")
    def unique_overrides(self):
        entries = self.minor + self.critical
        if len(entries) != len(set(entries)):
            raise ValueError("Each category may appear only once in severity overrides")
        return self

    def severity(self, category: Category) -> Severity:
        if category in self.critical:
            return "critical"
        return "minor" if category in self.minor else "major"


class Flag(Record):
    category: Category
    suggested_severity: Severity
    rule: Identifier


class ModelSuggestion(Record):
    turn_id: Identifier
    category: Category
    suggested_severity: Severity
    rationale: UtteranceText


class Suggestions(Record):
    schema_version: Literal[1] = 1
    scenario_id: Identifier
    suggestions: tuple[ModelSuggestion, ...]


class Interpretation(Record):
    text: str
    target_language: Language | None
    role: Role | None = None


class ConfidenceEvidence(Record):
    legacy_hits: int
    bilingual_hits: int
    legacy_score: int
    bilingual_score: int
    would_accept: bool


class StageMark(Record):
    stage: Identifier
    offset_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    engine_turn: int | None = None


class TurnResult(Record):
    id: Identifier
    speaker_id: Identifier
    role: Role
    source_language: Language
    target_language: Language
    event: Event
    source: str
    reference: str | None
    review_expected: bool
    interpretations: tuple[Interpretation, ...]
    disposition: Literal["accepted", "held"]
    review_reasons: tuple[str, ...] = ()
    flags: tuple[Flag, ...] = ()
    model_suggestions: tuple[ModelSuggestion, ...] = ()
    confidence_evidence: tuple[ConfidenceEvidence, ...] = ()
    stage_marks: tuple[StageMark, ...] = ()
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None


class Session(Record):
    schema_version: Literal[1] = 1
    fictional: Literal[True] = True
    id: UUID
    created_at: datetime
    scenario_id: Identifier
    title: str
    specialty: Specialty
    difficulty: Difficulty
    adapter: Literal["fixture", "deepseek"]
    audio: Literal["none", "mock", "edge"] = "none"
    severity_policy: SeverityPolicy
    turns: tuple[TurnResult, ...]


class HumanError(Record):
    category: Category
    severity: Severity


class Review(Record):
    schema_version: Literal[1] = 1
    session_id: UUID
    turn_id: Identifier
    status: Literal["pending", "reviewed"] = "reviewed"
    origin: Literal["human", "demo"] = "human"
    errors: tuple[HumanError, ...] = ()
    corrected_interpretation: Annotated[str, StringConstraints(max_length=4000)] = ""
    notes: Annotated[str, StringConstraints(max_length=4000)] = ""
    revision: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def unique_errors(self):
        if len({error.category for error in self.errors}) != len(self.errors):
            raise ValueError("Record each error category at most once per turn")
        return self
