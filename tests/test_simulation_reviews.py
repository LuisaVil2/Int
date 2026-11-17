"""Exercise durable review edits and the local workflow from CLI to export."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest

from src.simulation.export import build_export
from src.simulation.qa import calculate
from src.simulation.runner import DEFAULT_CORPUS, run_scenario
from src.simulation.schemas import Corpus, HumanError, Review
from src.simulation.store import ReviewStore


@pytest.fixture
def run(tmp_path):
    corpus = Corpus.model_validate_json(DEFAULT_CORPUS.read_bytes())
    case = next(case for case in corpus.scenarios if case.id == "med-dose")
    store = ReviewStore(tmp_path / "qa.sqlite3")
    session = run_scenario(case)
    store.add_session(session)
    return store, session


def test_reviews_are_independent_of_originals_and_revision_checked(run):
    store, session = run
    with pytest.raises(sqlite3.IntegrityError):
        store.add_session(session)
    draft = Review(session_id=session.id, turn_id="t2", status="pending",
                   corrected_interpretation="I take 500 mg of metformin with dinner.",
                   notes="The source and reference each retain 500 mg.")
    saved = store.save_review(draft)
    assert saved.revision == 1
    assert store.get_session(session.id) == session
    with pytest.raises(ValueError, match="changed elsewhere"):
        store.save_review(draft)
    with pytest.raises(ValueError, match="does not belong"):
        store.save_review(Review(session_id=session.id, turn_id="not-a-turn"))
    completed = save_final(store, saved)
    assert completed.revision == 2 and completed.status == "reviewed"
    assert store.reviews(session.id)["t2"] == completed


def save_final(store, draft):
    labels = (HumanError(category="medication_or_dose", severity="critical"),)
    return store.save_review(draft.model_copy(update={"status": "reviewed", "errors": labels}))


def test_rates_count_only_completed_human_reviews(run):
    store, session = run
    completed = Review(session_id=session.id, turn_id="t2")
    save_final(store, completed)
    store.save_review(Review(session_id=session.id, turn_id="t1", origin="demo"))
    store.save_review(Review(session_id=session.id, turn_id="t3", status="pending",
                             errors=(HumanError(category="omission", severity="major"),)))
    result = calculate(store.snapshot(session.id))
    assert result["turns"] == 4 and result["reviewed_turns"] == 1 and result["demo_reviewed_turns"] == 1
    assert result["human_review_completion_rate"] == 0.25
    assert result["human_confirmed"]["critical_error_count"] == 1
    assert result["human_confirmed"]["count"] == 1
    assert result["model_suggestions"]["count"] == 0
    assert result["confirmed_error_rate"] == 1.0
    assert result["by_direction"]["en->es"]["confirmed_error_rate"] is None
    assert result["by_scenario"]["med-dose"]["automated_flag_rate"] == 0.25
    assert result["measured_turn_latency_ms"] == {}


def test_export_omits_personal_fields_and_local_text_by_default(run):
    store, session = run
    store.save_review(Review(session_id=session.id, turn_id="t2", notes="private reviewer note",
                             corrected_interpretation="private reviewer correction"))
    default = build_export(store.snapshot(session.id))
    sample = default["sessions"][0]["turns"][1]
    assert default["schema_version"] == 1 and default["fictional"]
    assert "source" not in sample and "reference" not in sample
    assert "text" not in sample["interpretations"][0]
    assert sample["review"]["status"] == "reviewed"
    opt_in = build_export(store.snapshot(session.id), include_fictional_text=True)
    assert opt_in["sessions"][0]["turns"][1]["source"] == session.turns[1].source
    assert "private reviewer" not in json.dumps(opt_in)
    assert str(store.path) not in json.dumps(opt_in)


def test_cli_runs_reviews_and_exports_without_provider_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    database = tmp_path / "demo.sqlite3"

    def cli(*args):
        process = subprocess.run([sys.executable, "-m", "src.simulation", "--db", str(database), *args],
                                 capture_output=True, text=True, encoding="utf-8", check=True, timeout=30)
        return json.loads(process.stdout)

    selected = cli("list", "--specialty", "medication_reconciliation", "--difficulty", "advanced")
    assert {item["id"] for item in selected} == {"med-dose", "med-name"}
    result = cli("run", "--scenario", "med-dose", "--measure-latency")[0]
    key = result["session"]
    reviewed = cli("record-review", "--session", key, "--turn", "t2",
                   "--error", "medication_or_dose:critical", "--correction", "Fictional correction", "--demo")
    assert reviewed["status"] == "reviewed"
    metrics = cli("metrics", "--session", key)
    assert metrics["measured_turn_latency_ms"]["fixture"]["n"] == 4
    assert metrics["demo_reviewed_turns"] == 1 and metrics["human_confirmed"]["count"] == 0
    path = tmp_path / "evaluation.json"
    cli("export", "--session", key, "--output", str(path))
    exported = json.loads(path.read_bytes())
    assert exported["sessions"][0]["id"] == key
    assert "Fictional correction" not in path.read_text(encoding="utf-8")
