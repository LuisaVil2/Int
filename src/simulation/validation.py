"""Phase 3 local validation, independent review and fictional audio commands."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter_ns
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from .audio_bench import Span, evaluate_stt, read_pcm, render, transcribe
from .comparison import duration_stats, summarize
from .export import write_json
from .panel import Assignment, Decision, Panel
from .runner import DEFAULT_CORPUS, run_scenario
from .schemas import Corpus
from .store import DEFAULT_DB, ReviewStore


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixtures")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--baseline", type=Path)
    fixture.add_argument("--measure-latency", action="store_true")
    audio = commands.add_parser("audio")
    audio.add_argument("--scenario", required=True)
    audio.add_argument("--output", type=Path, required=True)
    audio.add_argument("--renderer", choices=("tone", "espeak"), default="tone")
    audio.add_argument("--espeak-executable", default="espeak-ng")
    audio.add_argument("--voice-en", default="en-us")
    audio.add_argument("--voice-es", default="es")
    audio.add_argument("--rate-en", type=int, default=175)
    audio.add_argument("--rate-es", type=int, default=160)
    audio.add_argument("--pause-ms", type=int, default=350)
    audio.add_argument("--noise-dbfs", type=float)
    audio.add_argument("--seed", type=int, default=42)
    audio.add_argument("--stt", choices=("replay", "whisper", "deepgram"), default="replay")
    audio.add_argument("--fault", choices=("none", "drop-duplicate"), default="none")
    audio.add_argument("--whisper-model", type=Path)
    audio.add_argument("--allow-network", action="store_true")
    audio.add_argument("--recording", type=Path)
    audio.add_argument("--timeline", type=Path)
    audio.add_argument("--authorized-fictional", action="store_true")
    audio.add_argument("--play", action="store_true")
    for name in ("assign", "blind-review", "adjudication-input", "panel-metrics"):
        item = commands.add_parser(name)
        item.add_argument("--session", type=UUID, required=True)
        if name in {"assign", "blind-review"}:
            item.add_argument("--slot", choices=("a", "b"), required=True)
            item.add_argument("--reviewer", required=True)
        if name == "assign":
            item.add_argument("--qualified", action="store_true")
        if name == "adjudication-input":
            item.add_argument("--turn", required=True)
            item.add_argument("--output", type=Path, required=True)
    commands.add_parser("adjudicate").add_argument("--file", type=Path, required=True)
    return parser


def verify_spans(spans, case, duration):
    if [item.turn_id for item in spans] != [turn.id for turn in case.turns]:
        raise ValueError("The timeline must contain every scripted turn in order")
    if any(not (math.isfinite(x.start) and math.isfinite(x.end)
                and 0 <= x.start < x.end <= duration + 1e-6) for x in spans):
        raise ValueError("The timeline contains an invalid audio interval")
    if list(spans) != sorted(spans, key=lambda item: item.start):
        raise ValueError("Timeline starts must be in chronological order")


def audio_run(args, corpus):
    case = next((s for s in corpus.scenarios if s.id == args.scenario), None)
    if case is None:
        raise ValueError("Unknown fictional scenario")
    if (args.recording or args.timeline) and not (args.recording and args.timeline and args.authorized_fictional):
        raise ValueError("An actor recording requires a timeline and --authorized-fictional")
    if args.stt == "deepgram" and not args.allow_network:
        raise ValueError("Deepgram audio requires --allow-network")
    if args.stt != "replay" and args.renderer == "tone" and not args.recording:
        raise ValueError("A speech recognizer requires a spoken synthetic or authorized recording")
    if args.fault != "none" and args.stt != "replay":
        raise ValueError("Injected STT faults require replay")
    if not 0 <= args.pause_ms <= 5000 or not all(80 <= x <= 350 for x in (args.rate_en, args.rate_es)):
        raise ValueError("Use pauses 0–5000 ms and speech rates 80–350 words per minute")
    if args.noise_dbfs is not None and not -90 <= args.noise_dbfs <= -5:
        raise ValueError("Noise must be between -90 and -5 dBFS")
    if args.output.exists():
        raise FileExistsError("Choose a new benchmark output filename")
    if args.stt == "deepgram":
        from dotenv import load_dotenv

        load_dotenv()
    directory = args.output.parent / f"{args.output.stem}-audio"
    voices = {"en": args.voice_en, "es": args.voice_es}
    rates = {"en": args.rate_en, "es": args.rate_es}
    synthesis = []
    if args.recording:
        path = args.recording
        timeline = json.loads(args.timeline.read_bytes())
        if not isinstance(timeline, dict) or timeline.get("schema_version") != 1 or timeline.get("scenario") != case.id:
            raise ValueError("The versioned timeline must identify this scenario")
        spans = TypeAdapter(tuple[Span, ...]).validate_python(timeline.get("turns"))
    else:
        path, spans, synthesis = render(case, directory, renderer=args.renderer,
                                       executable=args.espeak_executable, voices=voices, rates=rates,
                                       pause_ms=args.pause_ms, noise_dbfs=args.noise_dbfs, seed=args.seed)
        write_json(directory / "timeline.json", {"schema_version": 1, "scenario": case.id,
                                                  "turns": [x.model_dump() for x in spans]})
    start = perf_counter_ns()
    sr, pcm = read_pcm(path)
    read_done = perf_counter_ns()
    verify_spans(spans, case, len(pcm) / sr)
    if len(pcm) / sr > 600:
        raise ValueError("Benchmark recordings are limited to ten minutes")
    transcribe_start = perf_counter_ns()
    segments = transcribe(path, args.stt, case, spans, model=args.whisper_model,
                          allow_network=args.allow_network, fault=args.fault)
    end = perf_counter_ns()
    result = evaluate_stt(case, spans, segments)
    qa_done = perf_counter_ns()
    playback = None
    if args.play:
        import winsound

        began = perf_counter_ns()
        winsound.PlaySound(str(path.resolve()), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
        playback = (perf_counter_ns() - began) / 1e6
    return {
        "schema_version": 1, "scenario": case.id, "fictional": True,
        "evidence": "transcript_replay" if args.stt == "replay" else "offline_wav_recognition",
        "audio_source": "authorized_recording" if args.recording else args.renderer,
        "stt_adapter": args.stt, "fault": args.fault, "voices": voices if args.renderer == "espeak" else None,
        "wpm": rates if args.renderer == "espeak" else None,
        "pause_ms": args.pause_ms if not args.recording else None,
        "noise_dbfs": args.noise_dbfs if not args.recording else None, "seed": args.seed,
        "audio_duration_seconds": len(pcm) / sr,
        "timeline": [item.model_dump() for item in spans],
        "stt": result, "stage_latency_ms": {
            "source_tts": duration_stats(synthesis), "wav_read": duration_stats([(read_done - start) / 1e6]),
            "stt_batch_processing": duration_stats([(end - transcribe_start) / 1e6]),
            "stt_scoring": duration_stats([(qa_done - end) / 1e6]),
            "requested_wav_playback_call": duration_stats([playback]) if playback is not None else None,
        }, "physical_capture": None, "translation_from_audio": None, "end_to_end_latency_ms": None,
    }


def execute(args):
    if args.command in {"fixtures", "audio"}:
        corpus = Corpus.model_validate_json(DEFAULT_CORPUS.read_bytes())
        if args.command == "audio":
            result = audio_run(args, corpus)
        else:
            candidate = summarize(corpus.scenarios, run_scenario, measure_latency=args.measure_latency)
            result = {"schema_version": 1, "baseline": json.loads(args.baseline.read_bytes())
                      if args.baseline else None, "candidate": candidate}
        write_json(args.output, result)
        return {"output": str(args.output)}
    panel = Panel(ReviewStore(args.db))
    if args.command == "assign":
        panel.assign(args.session, args.slot, Assignment(reviewer=args.reviewer, qualified=args.qualified))
        return {"assigned": args.slot}
    if args.command == "blind-review":
        from .review_ui import open_reviewer

        open_reviewer(panel.view(args.session, args.slot, args.reviewer))
        return None
    if args.command == "adjudication-input":
        write_json(args.output, panel.adjudication_input(args.session, args.turn))
        return {"output": str(args.output)}
    if args.command == "adjudicate":
        return {"sequence": panel.adjudicate(Decision.model_validate_json(args.file.read_bytes()))}
    return panel.metrics(args.session)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = execute(options().parse_args(argv))
    except ValidationError:
        print("Invalid structured validation data.", file=sys.stderr)
        return 1
    except (ValueError, FileExistsError) as error:
        print(error, file=sys.stderr)
        return 1
    except Exception:
        print("Validation failed. Check local file, device, or provider setup.", file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
