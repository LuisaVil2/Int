"""SQLite persistence with append-only originals and revisioned review edits."""
from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import UUID

from .schemas import Review, Session

DEFAULT_DB = Path("out/simulation/reviews.sqlite3")


class ReviewStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    record TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    turn_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    record TEXT NOT NULL,
                    PRIMARY KEY (session_id, turn_id)
                );
            """)

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("PRAGMA foreign_keys = ON")
            yield db

    def add_session(self, session: Session) -> None:
        with self.connection() as db:
            db.execute("INSERT INTO sessions VALUES (?, ?)", (str(session.id), session.model_dump_json()))

    def get_session(self, session_id: UUID) -> Session:
        with self.connection() as db:
            row = db.execute("SELECT record FROM sessions WHERE id = ?", (str(session_id),)).fetchone()
        if row is None:
            raise ValueError("Session does not exist")
        return Session.model_validate_json(row[0])

    def sessions(self) -> list[Session]:
        with self.connection() as db:
            records = db.execute("SELECT record FROM sessions ORDER BY rowid DESC").fetchall()
        return [Session.model_validate_json(row[0]) for row in records]

    def reviews(self, session_id: UUID) -> dict[str, Review]:
        with self.connection() as db:
            rows = db.execute("SELECT turn_id, record FROM reviews WHERE session_id = ?",
                              (str(session_id),)).fetchall()
        return {turn_id: Review.model_validate_json(record) for turn_id, record in rows}

    def save_review(self, review: Review) -> Review:
        session = self.get_session(review.session_id)
        if review.turn_id not in {turn.id for turn in session.turns}:
            raise ValueError("Turn does not belong to this session")
        keys = (str(review.session_id), review.turn_id)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT revision FROM reviews WHERE session_id = ? AND turn_id = ?",
                             keys).fetchone()
            previous = old[0] if old else 0
            if review.revision != previous:
                raise ValueError("This review changed elsewhere. Reload the turn before saving.")
            updated = review.model_copy(update={"revision": previous + 1})
            db.execute("""
                INSERT INTO reviews VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, turn_id) DO UPDATE SET revision=excluded.revision, record=excluded.record
            """, (*keys, updated.revision, updated.model_dump_json()))
        return updated

    def snapshot(self, session_id: UUID | None = None) -> list[tuple[Session, dict[str, Review]]]:
        with self.connection() as db:
            db.execute("BEGIN")
            originals = db.execute("SELECT id, record FROM sessions ORDER BY rowid DESC"
                                   if session_id is None else "SELECT id, record FROM sessions WHERE id = ?",
                                   () if session_id is None else (str(session_id),)).fetchall()
            if not originals and session_id is not None:
                raise ValueError("Session does not exist")
            return [
                (Session.model_validate_json(record), {
                    turn: Review.model_validate_json(review)
                    for turn, review in db.execute("SELECT turn_id, record FROM reviews WHERE session_id = ?",
                                                   (key,))
                })
                for key, record in originals
            ]

    def audio_path(self, session_id: UUID, turn_id: str, kind: str) -> Path:
        session = self.get_session(session_id)
        if kind not in {"source", "output"} or turn_id not in {turn.id for turn in session.turns}:
            raise ValueError("Unknown audio track")
        return self.path.parent / "audio" / str(session.id) / f"{turn_id}-{kind}.wav"
