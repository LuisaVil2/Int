"""Generate fixture-replay evidence from a checked-out source tree."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

def duration_stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(ordered), "p50_ms": ordered[math.ceil(len(ordered) * .50) - 1],
            "p95_ms": ordered[math.ceil(len(ordered) * .95) - 1]}


def summarize(scenarios, run_scenario, *, measure_latency=False):
    totals, reasons, flags = Counter(), Counter(), Counter()
    scenario_rows, directions = {}, {}
    scenario_reasons, direction_reasons = {}, {}
    details = []
    timings = {}
    for case in scenarios:
        session = run_scenario(case, measure_latency=measure_latency)
        intended = {t.id: t for t in case.turns}
        scenario_rows[case.id] = Counter()
        for turn in session.turns:
            direction = f"{turn.source_language}->{turn.target_language}"
            spec = intended[turn.id]
            # All unequal authored candidates in the bundled Phase 2 corpus are injected examples.
            mutant = bool(spec.mock_reply and spec.mock_reply.text != spec.reference)
            confidence = getattr(turn, "confidence_evidence", ())
            shadow = (turn.review_reasons == ("confidence_review",) and bool(confidence)
                      and all(item.would_accept for item in confidence))
            for group in (totals, scenario_rows[case.id], directions.setdefault(direction, Counter())):
                group["turns"] += 1
                group[turn.disposition] += 1
                group["scripted_mutants"] += mutant
                group["accepted_scripted_mutants"] += mutant and turn.disposition == "accepted"
                group["flagged_scripted_mutants"] += mutant and bool(turn.flags)
                group["shadow_additional_accepts"] += shadow
                group["shadow_guarded_additional_accepts"] += shadow and not turn.flags
                group["shadow_additional_scripted_mutants"] += shadow and mutant
                group["shadow_guarded_additional_scripted_mutants"] += shadow and mutant and not turn.flags
                group["legacy_term_hits"] += sum(item.legacy_hits for item in confidence)
                group["bilingual_term_hits"] += sum(item.bilingual_hits for item in confidence)
            reasons.update(turn.review_reasons)
            scenario_reasons.setdefault(case.id, Counter()).update(turn.review_reasons)
            direction_reasons.setdefault(direction, Counter()).update(turn.review_reasons)
            flags.update(item.rule for item in turn.flags)
            for label, earlier, later in (
                ("fixture_translation", "translation_started", "translation_complete"),
                ("fixture_qa_after_translation", "translation_complete", "qa_decision"),
            ):
                marks = {(m.stage, m.engine_turn): m.offset_ms for m in getattr(turn, "stage_marks", ())}
                for (stage, key), end in marks.items():
                    if stage == later and (earlier, key) in marks:
                        timings.setdefault(label, []).append(end - marks[earlier, key])
            if turn.elapsed_ms is not None:
                timings.setdefault("fixture_processing", []).append(turn.elapsed_ms)
            details.append({"scenario": case.id, "turn": turn.id, "direction": direction,
                            "disposition": turn.disposition, "reasons": turn.review_reasons,
                            "rule_flags": [item.rule for item in turn.flags], "scripted_mutant": mutant})
    return {
        "schema_version": 1, "mode": "fixture_replay", "totals": dict(totals),
        "shadow_evidence": any(entry["legacy_term_hits"] or entry["bilingual_term_hits"]
                               for entry in directions.values()),
        "hold_reasons": dict(reasons), "rule_flags": dict(flags),
        "by_scenario": {key: {**dict(value), "hold_reasons": dict(scenario_reasons[key])}
                        for key, value in scenario_rows.items()},
        "by_direction": {key: {**dict(value), "hold_reasons": dict(direction_reasons[key])}
                         for key, value in sorted(directions.items())},
        "stage_latency_ms": {label: duration_stats(samples) for label, samples in sorted(timings.items())},
        "false_holds": None, "critical_errors_escaping_review": None,
        "confirmed_major_or_critical": None, "stt": None, "end_to_end_latency_ms": None,
        "reviewer_agreement": None, "adjudication": None, "turns": details,
    }


if __name__ == "__main__":
    import argparse
    import sys

    # Deliberately import the engine from the current checkout, even when
    # this benchmark driver lives in another worktree at a newer revision.
    sys.path.insert(0, str(Path.cwd()))
    from src.simulation.runner import DEFAULT_CORPUS, run_scenario
    from src.simulation.schemas import Corpus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure-latency", action="store_true")
    args = parser.parse_args()
    corpus = Corpus.model_validate_json(DEFAULT_CORPUS.read_bytes())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summarize(corpus.scenarios, run_scenario, measure_latency=args.measure_latency),
                  handle, indent=2, ensure_ascii=False)
