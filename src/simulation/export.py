"""Export structured evaluation data without local paths or reviewer free text."""
from __future__ import annotations

import json
from pathlib import Path

from .qa import calculate


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def build_export(snapshot, *, include_fictional_text=False) -> dict:
    sessions = []
    for session, reviews in snapshot:
        rows = []
        for turn in session.turns:
            row = turn.model_dump(mode="json", exclude={"source", "reference", "interpretations",
                                                        "model_suggestions"})
            row["interpretations"] = [
                item.model_dump(mode="json", exclude=set() if include_fictional_text else {"text"})
                for item in turn.interpretations
            ]
            row["model_suggestions"] = [
                item.model_dump(mode="json", exclude={"rationale"}) for item in turn.model_suggestions
            ]
            if include_fictional_text:
                row.update(source=turn.source, reference=turn.reference)
            review = reviews.get(turn.id)
            row["review"] = {"status": "pending", "origin": None, "errors": []}
            if review:
                row["review"] = review.model_dump(mode="json", include={"status", "origin", "errors"})
                if review.status == "pending":
                    row["review"]["errors"] = []
            rows.append(row)
        sessions.append({
            "id": str(session.id), "scenario_id": session.scenario_id, "specialty": session.specialty,
            "difficulty": session.difficulty, "adapter": session.adapter, "audio": session.audio,
            "severity_policy": session.severity_policy.model_dump(mode="json"), "turns": rows,
        })
    return {"schema_version": 1, "fictional": True, "includes_fictional_text": include_fictional_text,
            "sessions": sessions, "metrics": calculate(snapshot)}
