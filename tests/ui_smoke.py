"""Opt-in desktop smoke. Run with python -m pytest -q tests/ui_smoke.py."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from src.simulation.audio import render_tracks
from src.simulation.runner import DEFAULT_CORPUS, run_scenario
from src.simulation.schemas import Corpus
from src.simulation.store import ReviewStore


def test_desktop_reviewer_saves_and_exports_with_mock_audio(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from src.simulation.review_ui import Reviewer

    store = ReviewStore(tmp_path / "review.sqlite3")
    corpus = Corpus.model_validate_json(DEFAULT_CORPUS.read_bytes())
    case = next(scenario for scenario in corpus.scenarios if scenario.id == "pc-follow-up")
    session = run_scenario(case, audio="mock")
    store.add_session(session)
    assert render_tracks(session, store) == (5, 0)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("A desktop display is unavailable")
    root.withdraw()
    try:
        ui = Reviewer(root, store)
        root.update()
        assert ui.source.get("1.0", "end-1c") == session.turns[0].source
        assert ui.candidate.get("1.0", "end-1c").endswith(session.turns[0].interpretations[0].text)
        assert not ui.audio_buttons["source"].instate(["disabled"])
        assert not ui.audio_buttons["output"].instate(["disabled"])

        played = []
        monkeypatch.setitem(sys.modules, "winsound", SimpleNamespace(
            PlaySound=lambda path, flags: played.append(path), SND_FILENAME=1, SND_ASYNC=2, SND_NODEFAULT=4))
        ui.audio_buttons["output"].invoke()
        assert Path(played[0]).is_file()

        ui.turn_list.selection_clear(0, "end")
        ui.turn_list.selection_set(1)
        ui.select_turn()
        assert ui.audio_buttons["output"].instate(["disabled"])
        ui.errors["omission"].set("minor")
        ui.correction.insert("1.0", "Fictional correction from the UI smoke")
        ui.draft_button.invoke()
        assert store.reviews(session.id)["t2"].status == "pending"
        ui.done_button.invoke()
        assert store.reviews(session.id)["t2"].status == "reviewed"
        assert store.get_session(session.id) == session

        export = tmp_path / "export.json"
        monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **_: str(export))
        ui.export()
        assert export.is_file()
        assert "Fictional correction" not in export.read_text(encoding="utf-8")
    finally:
        root.destroy()


def test_blind_desktop_view_excludes_other_review_and_qa_evidence(tmp_path):
    from src.simulation.panel import Assignment, Panel
    from src.simulation.review_ui import Reviewer
    from src.simulation.schemas import Review

    tk = pytest.importorskip("tkinter")
    scenario = Corpus.model_validate_json(DEFAULT_CORPUS.read_bytes()).scenarios[0]
    session = run_scenario(scenario)
    store = ReviewStore(tmp_path / "panel.sqlite3")
    store.add_session(session)
    panel = Panel(store)
    panel.assign(session.id, "a", Assignment(reviewer="reviewer-a"))
    panel.assign(session.id, "b", Assignment(reviewer="reviewer-b"))
    panel.view(session.id, "a", "reviewer-a").save_review(Review(
        session_id=session.id, turn_id="t1", notes="Reviewer A's fictional notes"))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("A desktop display is unavailable")
    root.withdraw()
    try:
        ui = Reviewer(root, panel.view(session.id, "b", "reviewer-b"))
        assert not ui.saved and not ui.notes.get("1.0", "end-1c")
        assert "decisions are hidden" in ui.evidence.get("1.0", "end-1c")
        assert "accepted" not in ui.context.cget("text")
    finally:
        root.destroy()
