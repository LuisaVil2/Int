"""Independent study records. Reviewer views see only one assigned slot."""
from __future__ import annotations

from collections import Counter
from math import isclose
from typing import Literal
from uuid import UUID

from pydantic import Field

from .schemas import HumanError, Identifier, Record, Review
from .store import ReviewStore


class Assignment(Record):
    reviewer: Identifier
    qualified: bool = False


class Decision(Record):
    schema_version: Literal[1] = 1
    session_id: UUID
    turn_id: Identifier
    revision_a: int = Field(gt=0)
    revision_b: int = Field(gt=0)
    adjudicator: Identifier
    errors: tuple[HumanError, ...] = ()
    corrected_interpretation: str = Field(default="", max_length=4000)
    notes: str = Field(default="", max_length=4000)
    hold_justified: bool | None = None


class Panel:
    def __init__(self, store: ReviewStore):
        self.store = store
        with store.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS panel_assignments (
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    slot TEXT NOT NULL CHECK(slot IN ('a','b')),
                    reviewer TEXT NOT NULL,
                    record TEXT NOT NULL,
                    PRIMARY KEY(session_id, slot), UNIQUE(session_id, reviewer)
                );
                CREATE TABLE IF NOT EXISTS panel_reviews (
                    session_id TEXT NOT NULL, slot TEXT NOT NULL, turn_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, record TEXT NOT NULL,
                    PRIMARY KEY(session_id, slot, turn_id, revision),
                    FOREIGN KEY(session_id, slot) REFERENCES panel_assignments(session_id, slot)
                );
                CREATE TABLE IF NOT EXISTS panel_decisions (
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    turn_id TEXT NOT NULL, sequence INTEGER NOT NULL, record TEXT NOT NULL,
                    PRIMARY KEY(session_id, turn_id, sequence)
                );
            """)

    def assign(self, session_id: UUID, slot: str, assignment: Assignment):
        if slot not in {"a", "b"}:
            raise ValueError("Choose reviewer slot a or b")
        self.store.get_session(session_id)
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            assignments = self._assigned(db, session_id)
            if slot in assignments:
                if assignments[slot] != assignment:
                    raise ValueError("A reviewer assignment cannot be replaced after registration")
                return
            if any(a.reviewer == assignment.reviewer for a in assignments.values()):
                raise ValueError("The two independent slots require distinct reviewers")
            db.execute("INSERT INTO panel_assignments VALUES (?, ?, ?, ?)",
                       (str(session_id), slot, assignment.reviewer, assignment.model_dump_json()))

    @staticmethod
    def _assigned(db, session_id):
        return {slot: Assignment.model_validate_json(raw) for slot, raw in db.execute(
            "SELECT slot, record FROM panel_assignments WHERE session_id=?", (str(session_id),))}

    @staticmethod
    def _latest(db, session_id, slot):
        rows = db.execute("""
            SELECT turn_id, record FROM panel_reviews WHERE session_id=? AND slot=?
            ORDER BY revision
        """, (str(session_id), slot))
        return {turn: Review.model_validate_json(raw) for turn, raw in rows}

    def view(self, session_id: UUID, slot: str, reviewer: str):
        with self.store.connection() as db:
            assignment = self._assigned(db, session_id).get(slot)
        if assignment is None or assignment.reviewer != reviewer:
            raise ValueError("Reviewer is not assigned to this session and slot")
        return BlindView(self, session_id, slot)

    def _save(self, slot, review):
        session = self.store.get_session(review.session_id)
        if review.turn_id not in {t.id for t in session.turns}:
            raise ValueError("Turn does not belong to this session")
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._latest(db, review.session_id, slot).get(review.turn_id)
            revision = current.revision if current else 0
            if review.revision != revision:
                raise ValueError("Your review changed elsewhere. Reload before saving.")
            saved = review.model_copy(update={"revision": revision + 1})
            db.execute("INSERT INTO panel_reviews VALUES (?, ?, ?, ?, ?)", (
                str(review.session_id), slot, review.turn_id, saved.revision, saved.model_dump_json()))
        return saved

    @staticmethod
    def _pair(assignments, first, second, turn_id):
        a, b = first.get(turn_id), second.get(turn_id)
        if (a and b and a.status == b.status == "reviewed" and a.origin == b.origin == "human"
                and len(assignments) == 2 and all(item.qualified for item in assignments.values())):
            return a, b
        return None

    def adjudication_input(self, session_id, turn_id):
        with self.store.connection() as db:
            db.execute("BEGIN")
            pair = self._pair(self._assigned(db, session_id), self._latest(db, session_id, "a"),
                              self._latest(db, session_id, "b"), turn_id)
        if pair is None:
            raise ValueError("Adjudication requires two completed, qualified human reviews")
        turn = next(t for t in self.store.get_session(session_id).turns if t.id == turn_id)
        return {"turn": turn.model_dump(mode="json"), "reviews": {
            slot: review.model_dump(mode="json") for slot, review in zip(("a", "b"), pair)
        }}

    def adjudicate(self, decision: Decision):
        if len({item.category for item in decision.errors}) != len(decision.errors):
            raise ValueError("Record each adjudicated category once")
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            assignments = self._assigned(db, decision.session_id)
            if any(x.reviewer == decision.adjudicator for x in assignments.values()):
                raise ValueError("Use a separate adjudicator")
            pair = self._pair(assignments, self._latest(db, decision.session_id, "a"),
                              self._latest(db, decision.session_id, "b"), decision.turn_id)
            if not pair or tuple(r.revision for r in pair) != (decision.revision_a, decision.revision_b):
                raise ValueError("Adjudication requires the two current qualified review revisions")
            sequence = db.execute("SELECT coalesce(max(sequence),0)+1 FROM panel_decisions "
                                  "WHERE session_id=? AND turn_id=?",
                                  (str(decision.session_id), decision.turn_id)).fetchone()[0]
            db.execute("INSERT INTO panel_decisions VALUES (?, ?, ?, ?)", (
                str(decision.session_id), decision.turn_id, sequence, decision.model_dump_json()))
        return sequence

    def metrics(self, session_id):
        session = self.store.get_session(session_id)
        with self.store.connection() as db:
            db.execute("BEGIN")
            assignments = self._assigned(db, session_id)
            first, second = (self._latest(db, session_id, slot) for slot in ("a", "b"))
            decisions = {turn: Decision.model_validate_json(raw) for turn, raw in db.execute(
                "SELECT turn_id, record FROM panel_decisions WHERE session_id=? ORDER BY sequence",
                (str(session_id),))}
        pairs = {t.id: pair for t in session.turns
                 if (pair := self._pair(assignments, first, second, t.id))}
        binary_matches = exact_matches = a_positive = b_positive = 0
        final_labels, final_severities = Counter(), Counter()
        active = stale = 0
        accepted_reviewed = held_assessed = false_holds = critical_leaks = 0
        directions = {}
        turns = {turn.id: turn for turn in session.turns}
        disagreements = []
        for turn_id, (a, b) in pairs.items():
            binary_matches += bool(a.errors) == bool(b.errors)
            a_positive += bool(a.errors)
            b_positive += bool(b.errors)
            same = {(e.category, e.severity) for e in a.errors} == {(e.category, e.severity) for e in b.errors}
            exact_matches += same
            if not same:
                disagreements.append(turn_id)
            if decision := decisions.get(turn_id):
                if (decision.revision_a, decision.revision_b) == (a.revision, b.revision):
                    active += 1
                    final_labels.update(item.category for item in decision.errors)
                    final_severities.update(item.severity for item in decision.errors)
                    original = turns[turn_id]
                    direction = f"{original.source_language}->{original.target_language}"
                    group = directions.setdefault(direction, Counter())
                    group["adjudicated_turns"] += 1
                    group["confirmed_major_or_critical"] += sum(
                        item.severity in {"major", "critical"} for item in decision.errors)
                    if original.disposition == "accepted":
                        accepted_reviewed += 1
                        group["accepted_adjudicated"] += 1
                        leaked = any(item.severity == "critical" for item in decision.errors)
                        critical_leaks += leaked
                        group["critical_errors_escaping_review"] += leaked
                    elif decision.hold_justified is not None:
                        held_assessed += 1
                        group["held_assessed"] += 1
                        false_holds += decision.hold_justified is False
                        group["false_holds"] += decision.hold_justified is False
                else:
                    stale += 1
        n = len(pairs)
        observed = binary_matches / n if n else None
        expected = (a_positive * b_positive + (n-a_positive)*(n-b_positive)) / n**2 if n else None
        kappa = (observed - expected) / (1 - expected) if n and not isclose(expected, 1) else None
        return {"schema_version": 1, "turns": len(session.turns), "qualified_human_pairs": n,
                "binary_agreement": observed, "binary_kappa": kappa,
                "exact_label_severity_agreement": exact_matches / n if n else None,
                "disagreements": disagreements, "active_adjudications": active,
                "stale_adjudications": stale + len(decisions.keys() - pairs.keys()),
                "adjudicated_by_category": dict(final_labels),
                "adjudicated_by_severity": dict(final_severities),
                "accepted_adjudicated": accepted_reviewed,
                "held_assessed": held_assessed,
                "critical_errors_escaping_review": critical_leaks if accepted_reviewed else None,
                "false_holds": false_holds if held_assessed else None,
                "by_direction": {key: dict(value) for key, value in sorted(directions.items())}}


class BlindView:
    blind = True

    def __init__(self, panel, session_id, slot):
        self.panel, self.session_id, self.slot = panel, session_id, slot

    def sessions(self):
        return [self.panel.store.get_session(self.session_id)]

    def reviews(self, session_id):
        if session_id != self.session_id:
            raise ValueError("Session is outside the assigned scope")
        with self.panel.store.connection() as db:
            return self.panel._latest(db, session_id, self.slot)

    def save_review(self, review):
        if review.session_id != self.session_id:
            raise ValueError("Session is outside the assigned scope")
        return self.panel._save(self.slot, review)

    def audio_path(self, session_id, turn_id, kind):
        if session_id != self.session_id:
            raise ValueError("Session is outside the assigned scope")
        return self.panel.store.audio_path(session_id, turn_id, kind)
