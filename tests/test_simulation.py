"""Phase 2 simulations exercise the shared engine with no live services."""
from __future__ import annotations

import json
import wave
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.simulation import audio, generate, runner
from src.simulation.schemas import Corpus, Scenario, SeverityPolicy, Suggestions
from src.simulation.store import ReviewStore
from src.translate import Translation


@pytest.fixture(scope="module")
def corpus():
    return Corpus.model_validate_json(runner.DEFAULT_CORPUS.read_bytes())


def lookup(corpus, scenario_id):
    return next(case for case in corpus.scenarios if case.id == scenario_id)


def test_corpus_covers_domains_and_events(corpus):
    assert len(corpus.scenarios) >= 20
    assert len({case.specialty for case in corpus.scenarios}) == 10
    assert {case.difficulty for case in corpus.scenarios} == {"basic", "intermediate", "advanced"}
    turns = [turn for case in corpus.scenarios for turn in case.turns]
    assert {"interruption", "short_reply", "correction", "clarification", "code_switch"} <= {
        turn.event for turn in turns}
    assert {turn.asr_language for turn in turns} >= {None, "en", "es", "mixed"}


@pytest.mark.parametrize("change", ["version", "fiction", "turn_id", "role", "speaker"])
def test_scenarios_reject_invalid_identity_and_unversioned_variants(corpus, change):
    data = corpus.scenarios[0].model_dump(mode="json")
    if change == "version":
        data["schema_version"] = 2
    elif change == "fiction":
        data["fictional"] = False
    elif change == "turn_id":
        data["turns"][1]["id"] = data["turns"][0]["id"]
    elif change == "role":
        data["speakers"][0]["language"] = "es"
    else:
        data["turns"][0]["speaker_id"] = "unknown"
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_replays_same_utterances_and_keeps_scripted_speaker_ids(corpus, monkeypatch):
    def unexpected_cloud(*args):
        pytest.fail("Offline run constructed a cloud adapter")

    monkeypatch.setattr(runner, "DeepSeekTranslator", unexpected_cloud)
    case = lookup(corpus, "pc-follow-up")
    first, second = runner.run_scenario(case), runner.run_scenario(case)
    assert first.id != second.id and first.turns == second.turns
    assert [turn.role for turn in first.turns] == ["clinician", "patient", "clinician", "patient"]
    assert [turn.target_language for turn in first.turns] == ["es", "en", "es", "en"]
    assert all(turn.elapsed_ms is None for turn in first.turns)
    assert first.turns[0].disposition == "accepted"


@pytest.mark.parametrize(("case_id", "turn_id", "rule"), [
    ("em-breathing", "t2", "negation_changed"),
    ("em-breathing", "t3", "candidate_missing_or_unclear"),
    ("med-dose", "t2", "dose_units_changed"),
    ("med-name", "t2", "required_medication_token_missing"),
    ("pc-pain-scale", "t4", "numbers_changed"),
    ("ped-temperature", "t2", "required_measurement_unit_missing"),
    ("resp-inhaler", "t2", "reported_target_mismatch"),
    ("consent-speaker", "t2", "reported_role_mismatch"),
    ("resp-bilingual", "t3", "candidate_missing_or_unclear"),
    ("resp-bilingual", "t5", "candidate_missing_or_unclear"),
])
def test_exact_flags_and_held_turns(corpus, case_id, turn_id, rule):
    session = runner.run_scenario(lookup(corpus, case_id))
    turn = next(turn for turn in session.turns if turn.id == turn_id)
    assert turn.disposition == "held"
    assert rule in {flag.rule for flag in turn.flags}


def test_scope_and_laterality_are_held_with_specific_evidence(corpus):
    scope = runner.run_scenario(lookup(corpus, "obs-bleeding")).turns[1]
    side = runner.run_scenario(lookup(corpus, "card-pressure")).turns[0]
    for turn, rule in ((scope, "negation_scope_changed"), (side, "laterality_changed")):
        assert turn.interpretations[0].text != turn.reference
        assert rule in {flag.rule for flag in turn.flags}
        assert turn.disposition == "held" and rule in turn.review_reasons


@pytest.mark.parametrize("options", [{"adapter": "deepseek"}, {"audio": "edge"}])
def test_network_modes_require_opt_in(corpus, options):
    with pytest.raises(ValueError, match="allow-network"):
        runner.run_scenario(corpus.scenarios[0], **options)
    with pytest.raises(ValueError, match="allow-network"):
        generate.generate_scenario("primary_care", "basic")


def test_optional_provider_and_generator_can_be_mocked(corpus, monkeypatch):
    case = lookup(corpus, "pc-follow-up")
    turns = tuple(turn.model_copy(update={"mock_reply": None}) for turn in case.turns)
    draft = case.model_copy(update={"turns": turns})
    made = []

    def create(**kwargs):
        made.append(json.loads(kwargs["messages"][0]["content"]))
        return SimpleNamespace(stop_reason="end_turn", content=[
            SimpleNamespace(type="text", text=draft.model_dump_json())])

    monkeypatch.setattr(generate, "DeepSeekTranslator", lambda: SimpleNamespace(
        model="fixture", client=SimpleNamespace(messages=SimpleNamespace(create=create))))
    assert generate.generate_scenario("primary_care", "basic", allow_network=True) == draft
    assert made == [{"specialty": "primary_care", "difficulty": "basic"}]
    with pytest.raises(ValueError, match="requested settings"):
        generate.generate_scenario("cardiology", "basic", allow_network=True)

    simulated_requests = []

    def translate(text, lang, terms, context):
        simulated_requests.append((text, lang))
        return Translation(text="<UNCLEAR>", needs_clarification=True, target_lang="en" if lang == "es" else "es")

    monkeypatch.setattr(runner, "DeepSeekTranslator", lambda: SimpleNamespace(translate=translate))
    session = runner.run_scenario(draft, adapter="deepseek", allow_network=True)
    assert session.adapter == "deepseek"
    assert simulated_requests[0] == (draft.turns[0].text, "en")
    assert all(turn.disposition == "held" for turn in session.turns)


def test_mock_audio_creates_playable_wav_and_skips_held_outputs(corpus, tmp_path):
    session = runner.run_scenario(lookup(corpus, "pc-follow-up"), audio="mock")
    store = ReviewStore(tmp_path / "qa.sqlite3")
    store.add_session(session)
    saved, failed = audio.render_tracks(session, store)
    assert saved == len(session.turns) + 1 and failed == 0
    with wave.open(str(store.audio_path(session.id, "t1", "output")), "rb") as wav:
        assert wav.getnframes() == 2000 and wav.getframerate() == 8000
    assert store.audio_path(session.id, "t2", "source").is_file()
    assert not store.audio_path(session.id, "t2", "output").exists()
    with pytest.raises(ValueError):
        store.audio_path(session.id, "..", "source")


def test_policy_and_model_suggestions_are_stored_separately(corpus):
    case = lookup(corpus, "med-dose")
    policy = SeverityPolicy(critical=(), minor=("numbers_or_units",))
    suggestions = Suggestions(scenario_id=case.id, suggestions=[{
        "turn_id": "t2", "category": "omission", "suggested_severity": "major", "rationale": "Fixture guess"}])
    session = runner.run_scenario(case, policy=policy, suggestions=suggestions, measure_latency=True)
    result = session.turns[1]
    assert result.elapsed_ms >= 0
    assert {f.category: f.suggested_severity for f in result.flags} == {
        "numbers_or_units": "minor", "medication_or_dose": "major"}
    assert [s.category for s in result.model_suggestions] == ["omission"]
    with pytest.raises(ValueError, match="do not match"):
        runner.run_scenario(corpus.scenarios[0], suggestions=suggestions)
