import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.clinical_checks import meaning_flags
from src.live_engine import LiveEngine
from src.simulation.audio_bench import Heard, Span, edits, evaluate_stt
from src.simulation.panel import Assignment, Decision, Panel
from src.simulation.qa import check_turn
from src.simulation.runner import DEFAULT_CORPUS, run_scenario
from src.simulation.schemas import Corpus, HumanError, Interpretation, Review, SeverityPolicy, Utterance
from src.simulation.store import ReviewStore
from src.simulation.validation import audio_run, options
from src.terminology import TerminologyIndex

EXAMPLES = json.loads((Path(__file__).parent / "fixtures" / "clinical_meaning.json").read_bytes())


@pytest.fixture(scope="module")
def case():
    return Corpus.model_validate_json(DEFAULT_CORPUS.read_bytes()).scenarios[0]


@pytest.mark.parametrize(("_id", "source", "candidate", "expected"), EXAMPLES,
                         ids=[row[0] for row in EXAMPLES])
def test_anchored_evidence(_id, source, candidate, expected):
    rules = dict(meaning_flags(source, candidate))
    if expected:
        assert expected in rules
    else:
        assert not rules


def test_qa_accepts_equivalent_words_for_anchored_quantities():
    turn = Utterance(id="t1", speaker_id="patient", source_language="es",
                     text="Tomo quinientos miligramos de metformina.",
                     reference="I take 500 mg of metformin.")
    answer = Interpretation(text="I take five hundred milligrams of metformin.", target_language="en")
    assert check_turn(turn, (answer,), "en", "patient", accepted=False, policy=SeverityPolicy()) == ()


def test_bilingual_glossary_preserves_legacy_hits_and_threshold():
    from src.confidence import ConfidenceEngine

    idx = TerminologyIndex.load("data/terminology")
    source = "Tengo fiebre, tos, mareo, náusea y sarpullido."
    assert len(idx.lookup_bilingual(source, "general", "es")) > len(idx.lookup(source, "general"))
    assert idx.lookup_bilingual(source, "general", "es") == idx.lookup_bilingual(
        source.replace("á", "a"), "general", "es")
    assert idx.lookup_bilingual("Fever and cough", "general", "en") == idx.lookup("Fever and cough", "general")
    engine = ConfidenceEngine()
    assert engine.route(94) == "qa_review" and engine.route(95) == "automatic_approval"


def test_meaning_gate_blocks_high_confidence_laterality_swap():
    from src.confidence import ConfidenceEngine
    from src.emergency import EmergencyClassifier
    from src.memory import ConversationMemory
    from src.translate import Translation

    events = []
    engine = LiveEngine(lambda kind, data: events.append((kind, data)), {}, None, tts_on=True)
    translator = SimpleNamespace(translate=lambda *args: Translation(
        target_lang="es", text="Le tomé la presión en el brazo derecho.", confidence=1))
    engine._process_text_turn("I took your blood pressure in the left arm.", "en", 1, None,
                              "mock", None, translator, SimpleNamespace(lookup=lambda *args: ["term"] * 5),
                              lambda *args: "en", ConversationMemory(), EmergencyClassifier(),
                              ConfidenceEngine(), stream_tts=True, direction_resolved=True)
    assert engine._tts_q.empty()
    assert any(kind == "needs_review" and data["reason"] == "laterality_changed"
               for kind, data in events)


def test_numeric_equivalence_keeps_legacy_gate_for_other_digits():
    from src.translate import fidelity_issue

    assert fidelity_issue("Tomo 500 mg.", "I take five hundred milligrams.") is None
    assert fidelity_issue("Tomo 10 mg.", "I take five five milligrams.") == "numbers_changed"
    assert fidelity_issue("Tomo 5 mg.", "I take -5 mg.") == "numbers_changed"
    assert fidelity_issue("Tomo 500 mg a las 8.", "I take five hundred milligrams at 9.") == "numbers_changed"


def start_panel(tmp_path, case):
    store = ReviewStore(tmp_path / "review.sqlite3")
    session = run_scenario(case)
    store.add_session(session)
    panel = Panel(store)
    panel.assign(session.id, "a", Assignment(reviewer="interpreter-a", qualified=True))
    return store, session, panel


def test_blinded_panel_stores_revisions_and_marks_stale_adjudication(tmp_path, case):
    store, session, panel = start_panel(tmp_path, case)
    sid = session.id
    with pytest.raises(ValueError, match="distinct"):
        panel.assign(sid, "b", Assignment(reviewer="interpreter-a", qualified=True))
    panel.assign(sid, "b", Assignment(reviewer="interpreter-b", qualified=True))
    a = panel.view(sid, "a", "interpreter-a")
    b = panel.view(sid, "b", "interpreter-b")
    with pytest.raises(ValueError, match="not assigned"):
        panel.view(sid, "b", "interpreter-a")
    error = HumanError(category="mistranslation", severity="critical")
    review = Review(session_id=sid, turn_id="t1", errors=(error,), corrected_interpretation="Proposed correction")
    first = a.save_review(review)
    assert not b.reviews(sid)
    with pytest.raises(ValueError, match="two completed"):
        panel.adjudication_input(sid, "t1")
    b.save_review(Review(session_id=sid, turn_id="t1"))
    assert panel.metrics(sid)["binary_agreement"] == 0
    assert panel.metrics(sid)["disagreements"] == ["t1"]
    assert "reviews" in panel.adjudication_input(sid, "t1")
    panel.adjudicate(Decision(session_id=sid, turn_id="t1", revision_a=1, revision_b=1,
                             adjudicator="independent-adjudicator", errors=(error,)))
    assert panel.metrics(sid)["critical_errors_escaping_review"] == 1
    a.save_review(first)
    assert panel.metrics(sid)["stale_adjudications"] == 1
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM panel_reviews").fetchone()[0] == 3
    assert store.get_session(sid) == session and not store.reviews(sid)


@pytest.mark.parametrize(("qualified", "origin"), [(False, "human"), (True, "demo")])
def test_ineligible_reviews_do_not_count(tmp_path, case, qualified, origin):
    _, session, panel = start_panel(tmp_path, case)
    sid = session.id
    panel.assign(sid, "b", Assignment(reviewer="interpreter-b", qualified=qualified))
    for slot, reviewer in (("a", "interpreter-a"), ("b", "interpreter-b")):
        panel.view(sid, slot, reviewer).save_review(Review(session_id=sid, turn_id="t1", origin=origin))
    assert panel.metrics(sid)["qualified_human_pairs"] == 0
    assert panel.metrics(sid)["binary_kappa"] is None


def test_audio_replay_measures_injected_loss_and_duplicate(tmp_path, case):
    args = options().parse_args(["audio", "--scenario", case.id, "--output", str(tmp_path / "report.json"),
                                 "--fault", "drop-duplicate", "--pause-ms", "40", "--noise-dbfs", "-40"])
    report = audio_run(args, Corpus(scenarios=(case,)))
    assert report["evidence"] == "transcript_replay"
    assert report["stt"]["missing_turns"] == report["stt"]["duplicate_segments"] == 1
    assert report["stt"]["wer"]["rate"] > 0
    assert report["stage_latency_ms"]["source_tts"]["n"] == len(case.turns)
    assert report["end_to_end_latency_ms"] is None
    assert (tmp_path / "report-audio" / "dialogue.wav").is_file()


def test_stt_metrics_keep_timing_language_and_speakers_separate(case):
    spans = tuple(Span(turn_id=t.id, start=i, end=i + 0.8) for i, t in enumerate(case.turns))
    bad = [Heard(case.turns[0].text, 0, 1.5, lang="es", speaker=0),
           Heard(case.turns[1].text, 1, 1.7, lang=None, speaker=0)]
    score = evaluate_stt(case, spans, bad)
    assert score["merged_segments"] == 1 and score["missing_turns"] == 2
    assert score["language_mismatches"] == score["language_unknown"] == 1
    assert score["speaker_cluster_errors"] == 1
    assert edits(["a", "b"], ["a", "c", "d"]) == {"substitutions": 1, "deletions": 0, "insertions": 1}


def test_speech_renderer_passes_voice_rate_and_utf8_locally(tmp_path, case, monkeypatch):
    from src.simulation.audio_bench import render, write_pcm

    calls = []

    def synthesize(command, *, input, **kwargs):
        calls.append((command, input.decode("utf-8")))
        write_pcm(command[command.index("-w") + 1], np.ones(160, dtype=np.int16))

    monkeypatch.setattr("src.simulation.audio_bench.subprocess.run", synthesize)
    path, spans, timing = render(case, tmp_path, renderer="espeak", executable="fixture-espeak",
                                 voices={"en": "en-us", "es": "es"}, rates={"en": 180, "es": 140},
                                 pause_ms=20, noise_dbfs=None, seed=42)
    assert path.is_file() and len(spans) == len(timing) == len(case.turns)
    assert calls[1][0][1:7] == ["-b", "1", "-v", "es", "-s", "140"]
    assert calls[1][1] == case.turns[1].text


def test_trace_mock_capture_mutes_playback_echo_and_stops(monkeypatch):
    trace, sent = [], []
    engine = LiveEngine(lambda *args: None, {}, None, tts_on=False, trace=lambda *args: trace.append(args))
    engine._speaking.set()
    engine._dg = SimpleNamespace(closed=False, set_capture_muted=lambda _: None,
                                 send_np=sent.append, close=lambda: None)

    def fake_record(size):
        if sent:
            engine.stop()
        return np.ones(size, dtype=np.float32)

    engine._capture_streaming(SimpleNamespace(record=fake_record))
    assert len(sent) == 1 and not sent[0].any()
    assert [entry[0] for entry in trace] == ["capture_frame", "capture_frame"]
    monkeypatch.setattr("src.live_engine.configure_logging", lambda: None)
    with pytest.raises(RuntimeError, match="sesión nueva"):
        engine.start()
    fresh = LiveEngine(lambda *args: None, {}, None, tts_on=False)
    assert not fresh._stop.is_set()
