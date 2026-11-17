"""Command-line entry point for local fictional simulations and QA."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import get_args
from uuid import UUID

from pydantic import ValidationError

from .audio import render_tracks
from .export import build_export, write_json
from .generate import generate_scenario
from .qa import calculate
from .runner import DEFAULT_CORPUS, run_scenario
from .schemas import Corpus, Difficulty, HumanError, Review, Scenario, SeverityPolicy, Specialty, Suggestions
from .store import DEFAULT_DB, ReviewStore


def selection(parser):
    parser.add_argument("--specialty", choices=get_args(Specialty))
    parser.add_argument("--difficulty", choices=get_args(Difficulty))
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=40)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = result.add_subparsers(dest="command", required=True)
    selection(commands.add_parser("list", help="List authored fictional scenarios"))

    run = commands.add_parser("run", help="Run one scenario or a filtered group")
    choice = run.add_mutually_exclusive_group()
    choice.add_argument("--scenario")
    choice.add_argument("--scenario-file", type=Path)
    selection(run)
    run.add_argument("--adapter", choices=("fixture", "deepseek"), default="fixture")
    run.add_argument("--audio", choices=("none", "mock", "edge"), default="none")
    run.add_argument("--voice-en")
    run.add_argument("--voice-es")
    run.add_argument("--allow-network", action="store_true")
    run.add_argument("--measure-latency", action="store_true")
    run.add_argument("--policy", type=Path)
    run.add_argument("--suggestions", type=Path)

    generation = commands.add_parser("generate", help="Generate an opt-in draft scenario")
    generation.add_argument("--specialty", required=True, choices=get_args(Specialty))
    generation.add_argument("--difficulty", required=True, choices=get_args(Difficulty))
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--allow-network", action="store_true")

    commands.add_parser("review", help="Open the local desktop reviewer")
    record = commands.add_parser("record-review", help="Save a review from the command line")
    record.add_argument("--session", required=True, type=UUID)
    record.add_argument("--turn", required=True)
    record.add_argument("--error", action="append", default=[], metavar="CATEGORY:SEVERITY")
    record.add_argument("--correction", default="")
    record.add_argument("--notes", default="")
    record.add_argument("--pending", action="store_true")
    record.add_argument("--demo", action="store_true", help="Exclude a simulated review from human metrics")
    for name in ("metrics", "export"):
        report = commands.add_parser(name)
        report.add_argument("--session", type=UUID)
        if name == "export":
            report.add_argument("--output", type=Path, required=True)
            report.add_argument("--include-fictional-text", action="store_true")
    return result


def pick_scenarios(args):
    if not 2 <= args.min_turns <= args.max_turns <= 40:
        raise ValueError("Turn count must be between 2 and 40 and min-turns cannot exceed max-turns")
    external = getattr(args, "scenario_file", None)
    scenarios = (Scenario.model_validate_json(external.read_bytes()),) if external else (
        Corpus.model_validate_json(DEFAULT_CORPUS.read_bytes()).scenarios)
    return [scenario for scenario in scenarios
            if (not getattr(args, "scenario", None) or args.scenario == scenario.id)
            and (not args.specialty or args.specialty == scenario.specialty)
            and (not args.difficulty or args.difficulty == scenario.difficulty)
            and args.min_turns <= len(scenario.turns) <= args.max_turns]


def run_selected(args, scenarios):
    if (args.adapter == "deepseek" or args.audio == "edge") and not args.allow_network:
        raise ValueError("External providers require --allow-network")
    if (args.voice_en or args.voice_es) and args.audio != "edge":
        raise ValueError("Voice settings require Edge audio")
    if not scenarios:
        raise ValueError("No matching scenarios")
    if args.suggestions and len(scenarios) != 1:
        raise ValueError("Model suggestions require one matching scenario")
    if args.allow_network:
        from dotenv import load_dotenv

        load_dotenv()
    policy = SeverityPolicy.model_validate_json(args.policy.read_bytes()) if args.policy else None
    suggestions = Suggestions.model_validate_json(args.suggestions.read_bytes()) if args.suggestions else None
    store = ReviewStore(args.db)
    summaries = []
    for scenario in scenarios:
        session = run_scenario(scenario, adapter=args.adapter, allow_network=args.allow_network,
                               measure_latency=args.measure_latency, audio=args.audio,
                               policy=policy, suggestions=suggestions)
        store.add_session(session)
        saved, failed = render_tracks(session, store, allow_network=args.allow_network,
                                      voice_en=args.voice_en, voice_es=args.voice_es)
        summaries.append({"session": str(session.id), "scenario": scenario.id, "turns": len(session.turns),
                          "held": sum(turn.disposition == "held" for turn in session.turns),
                          "audio_saved": saved, "audio_failed": failed})
    return summaries


def execute(args):
    if args.command == "list":
        return [scenario.model_dump(include={"id", "title", "specialty", "difficulty"})
                for scenario in pick_scenarios(args)]
    if args.command == "run":
        return run_selected(args, pick_scenarios(args))
    if args.command == "generate":
        if not args.allow_network:
            raise ValueError("External generation requires --allow-network")
        from dotenv import load_dotenv

        load_dotenv()
        scenario = generate_scenario(args.specialty, args.difficulty, allow_network=True)
        write_json(args.output, scenario.model_dump(mode="json"))
        return {"scenario": scenario.id, "draft": True}

    store = ReviewStore(args.db)
    if args.command == "review":
        from .review_ui import open_reviewer

        open_reviewer(store)
        return None
    if args.command == "record-review":
        errors = []
        for value in args.error:
            category, sep, severity = value.partition(":")
            if not sep:
                raise ValueError("Use --error CATEGORY:SEVERITY")
            errors.append(HumanError(category=category, severity=severity))
        existing = store.reviews(args.session).get(args.turn)
        review = store.save_review(Review(
            session_id=args.session, turn_id=args.turn, errors=tuple(errors),
            status="pending" if args.pending else "reviewed", origin="demo" if args.demo else "human",
            corrected_interpretation=args.correction, notes=args.notes,
            revision=existing.revision if existing else 0,
        ))
        return {"session": str(review.session_id), "turn": review.turn_id,
                "status": review.status, "origin": review.origin, "revision": review.revision}
    snapshot = store.snapshot(args.session)
    if args.command == "metrics":
        return calculate(snapshot)
    write_json(args.output, build_export(snapshot, include_fictional_text=args.include_fictional_text))
    return {"sessions": len(snapshot), "output": str(args.output)}


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        result = execute(args)
    except ValidationError:
        print("Invalid structured simulation or review data.", file=sys.stderr)
        return 1
    except (ValueError, FileExistsError) as error:
        print(error, file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error):
        print("Could not open or save local simulation data.", file=sys.stderr)
        return 1
    except Exception:
        print("Simulation operation failed. Check provider or desktop configuration.", file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
