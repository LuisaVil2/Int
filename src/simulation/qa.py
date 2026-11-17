"""Exact checks and metrics. Semantic labels come from separate reviews."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from ..translate import fidelity_issues
from .schemas import Category, Flag, Interpretation, Review, Session, SeverityPolicy, Utterance

_RULES: dict[str, Category] = {
    "numbers_changed": "numbers_or_units",
    "dose_units_changed": "medication_or_dose",
    "negation_changed": "negation_reversal",
}


def check_turn(turn: Utterance, interpretations: tuple[Interpretation, ...], target: str,
               role: str, *, accepted: bool, policy: SeverityPolicy) -> tuple[Flag, ...]:
    flags: list[Flag] = []

    def add(category: Category, rule: str) -> None:
        flags.append(Flag(category=category, rule=rule, suggested_severity=policy.severity(category)))

    rendered = [item for item in interpretations if item.text.strip()
                and "<unclear>" not in item.text.casefold()]
    if len(rendered) != len(interpretations) or not rendered:
        add("uncertainty_or_failure", "candidate_missing_or_unclear")
    if accepted and turn.review_expected:
        add("unjustified_inference", "expected_review_bypassed")
    if any(item.role is not None and item.role != role for item in interpretations):
        add("speaker_attribution", "reported_role_mismatch")
    if any(item.target_language is not None and item.target_language != target
           for item in interpretations):
        add("wrong_direction", "reported_target_mismatch")
    if rendered and all(item.target_language == target for item in rendered):
        candidate = " ".join(item.text for item in rendered)
        if turn.reference:
            for rule in fidelity_issues(turn.reference, candidate):
                add(_RULES[rule], rule)
        for groups, category, rule in (
            (turn.required_target_terms, "medication_or_dose", "required_medication_token_missing"),
            (turn.required_target_units, "numbers_or_units", "required_measurement_unit_missing"),
        ):
            if any(not any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", candidate, re.I)
                           for alias in aliases) for aliases in groups):
                add(category, rule)
    return tuple(flags)


def calculate(sessions: Iterable[tuple[Session, dict[str, Review]]]) -> dict:
    """Aggregate reviews and evidence channels with explicit, separate denominators."""
    totals = Counter()
    automated = Counter()
    model = Counter()
    confirmed = Counter()
    severity = Counter()
    by_scenario: dict[str, Counter] = {}
    by_direction: dict[str, Counter] = {}
    timing: dict[str, list[float]] = {}
    for session, reviews in sessions:
        for turn in session.turns:
            direction = f"{turn.source_language}->{turn.target_language}"
            groups = (totals, by_scenario.setdefault(session.scenario_id, Counter()),
                      by_direction.setdefault(direction, Counter()))
            review = reviews.get(turn.id)
            human_done = bool(review and review.origin == "human" and review.status == "reviewed")
            demo_done = bool(review and review.origin == "demo" and review.status == "reviewed")
            for group in groups:
                group["turns"] += 1
                group["flagged_turns"] += bool(turn.flags)
                group["reviewed_turns"] += human_done
                group["turns_with_confirmed_error"] += bool(human_done and review.errors)
                group["demo_reviewed_turns"] += demo_done
            automated.update(flag.category for flag in turn.flags)
            model.update(suggestion.category for suggestion in turn.model_suggestions)
            if human_done:
                confirmed.update(error.category for error in review.errors)
                severity.update(error.severity for error in review.errors)
            if turn.elapsed_ms is not None:
                timing.setdefault(session.adapter, []).append(turn.elapsed_ms)

    def rates(counts: Counter) -> dict:
        count, reviewed = counts["turns"], counts["reviewed_turns"]
        errors = counts["turns_with_confirmed_error"]
        return {"turns": count, "flagged_turns": counts["flagged_turns"], "reviewed_turns": reviewed,
                "demo_reviewed_turns": counts["demo_reviewed_turns"],
                "turns_with_confirmed_error": errors,
                "automated_flag_rate": counts["flagged_turns"] / count if count else None,
                "confirmed_error_rate": errors / reviewed if reviewed else None,
                "human_review_completion_rate": reviewed / count if count else None}

    def distribution(values: list[float]) -> dict:
        values = sorted(values)
        return {"n": len(values), "min": values[0], "max": values[-1],
                "p50": values[math.ceil(0.50 * len(values)) - 1],
                "p95": values[math.ceil(0.95 * len(values)) - 1]}

    return {
        "schema_version": 1,
        **rates(totals),
        "automated_flags": {"count": sum(automated.values()), "by_category": dict(automated)},
        "model_suggestions": {"count": sum(model.values()), "by_category": dict(model)},
        "human_confirmed": {"count": sum(confirmed.values()), "by_category": dict(confirmed),
                            "by_severity": dict(severity), "critical_error_count": severity["critical"]},
        "by_scenario": {key: rates(value) for key, value in sorted(by_scenario.items())},
        "by_direction": {key: rates(value) for key, value in sorted(by_direction.items())},
        "measured_turn_latency_ms": {key: distribution(values) for key, values in sorted(timing.items())},
    }
