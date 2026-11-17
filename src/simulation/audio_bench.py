"""Fictional audio rendering and STT measurement with explicit evidence modes."""
from __future__ import annotations

import math
import os
import re
import subprocess
import unicodedata
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from ..clinical_checks import inventory
from .comparison import duration_stats
from .schemas import Record, Scenario

SR = 16000


class Span(Record):
    turn_id: str
    start: float
    end: float


@dataclass(frozen=True)
class Heard:
    text: str
    start: float
    end: float
    lang: str | None = None
    speaker: int | None = None


def tokenize(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text.casefold()))


def edits(reference, candidate):
    """Return substitution, deletion, insertion counts from a minimum edit alignment."""
    previous = [(0, 0, j) for j in range(len(candidate) + 1)]
    for i, source in enumerate(reference, 1):
        current = [(0, i, 0)]
        for j, target in enumerate(candidate, 1):
            diagonal = previous[j - 1]
            if source == target:
                current.append(diagonal)
            else:
                s, d, a = diagonal
                ds, dd, da = previous[j]
                ins, ind, ina = current[j - 1]
                current.append(min(((s + 1, d, a), (ds, dd + 1, da), (ins, ind, ina + 1)), key=sum))
        previous = current
    return dict(zip(("substitutions", "deletions", "insertions"), previous[-1]))


def error_rate(reference, candidate):
    counts = edits(reference, candidate)
    return {**counts, "reference_count": len(reference),
            "rate": sum(counts.values()) / len(reference) if reference else None}


def entity_set(text):
    facts = inventory(text)
    terms = set()
    for key, values in facts.items():
        if isinstance(values, dict):
            terms.update((key, name, value) for name, entries in values.items() for value in entries)
        else:
            terms.update((key, value) for value in values)
    return terms


def evaluate_stt(scenario: Scenario, spans: tuple[Span, ...], hypothesis: list[Heard]):
    refs = {t.id: t for t in scenario.turns}
    by_turn = {t.id: [] for t in scenario.turns}
    ordered = sorted(hypothesis, key=lambda h: (h.start, h.end))
    merged = unassigned = duplicates = 0
    seen = set()
    for h in ordered:
        overlapping = [(max(0, min(s.end, h.end) - max(s.start, h.start)), s.turn_id) for s in spans]
        overlaps = [(n, key) for n, key in overlapping if n > 0]
        if not overlaps:
            unassigned += 1
            continue
        if len(overlaps) > 1:
            merged += 1
        key = max(overlaps, key=lambda item: item[0])[1]
        fingerprint = (key, round(h.start, 3), round(h.end, 3), tuple(tokenize(h.text)))
        if fingerprint in seen:
            duplicates += 1
        seen.add(fingerprint)
        by_turn[key].append(h)

    missing = split = language_errors = language_unknown = 0
    clusters = Counter()
    directions = {}
    for key, parts in by_turn.items():
        ref = refs[key]
        direction = f"{ref.source_language}->{ 'es' if ref.source_language == 'en' else 'en'}"
        group = directions.setdefault(direction, {"reference": [], "hypothesis": [],
                                                   "entities_missing": 0, "entities_added": 0})
        group["reference"].extend(tokenize(ref.text))
        output = " ".join(item.text for item in parts)
        group["hypothesis"].extend(tokenize(output))
        required, observed = entity_set(ref.text), entity_set(output)
        group["entities_missing"] += len(required - observed)
        group["entities_added"] += len(observed - required)
        missing += not parts
        split += len({(p.start, p.end, p.text) for p in parts}) > 1
        for h in parts:
            language_unknown += h.lang not in {"en", "es"}
            language_errors += h.lang in {"en", "es"} and h.lang != ref.source_language
            if h.speaker is not None:
                clusters[(ref.speaker_id, h.speaker)] += 1

    observed_clusters = {tag for _, tag in clusters}
    speakers = [speaker.id for speaker in scenario.speakers]
    best = max((clusters[speakers[0], a] + clusters[speakers[1], b]
                for a in (*observed_clusters, None) for b in (*observed_clusters, None)
                if a != b or a is None), default=0)
    speaker_segments = sum(clusters.values())
    ref_tokens = tokenize(" ".join(refs[s.turn_id].text for s in spans))
    hyp_tokens = tokenize(" ".join(h.text for h in ordered))
    return {"wer": error_rate(ref_tokens, hyp_tokens), "cer": error_rate("".join(ref_tokens), "".join(hyp_tokens)),
            "missing_turns": missing, "duplicate_segments": duplicates, "split_turns": split,
            "merged_segments": merged, "unassigned_segments": unassigned,
            "language_mismatches": language_errors, "language_unknown": language_unknown,
            "speaker_segments_scored": speaker_segments,
            "speaker_cluster_errors": speaker_segments - best if speaker_segments else None,
            "by_direction": {key: {"wer": error_rate(group["reference"], group["hypothesis"]),
                                   "cer": error_rate("".join(group["reference"]), "".join(group["hypothesis"])),
                                   "entities_missing": group["entities_missing"],
                                   "entities_added": group["entities_added"]}
                             for key, group in sorted(directions.items())}}


def read_pcm(path):
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError("Audio must be mono PCM16 WAV")
        return handle.getframerate(), np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")


def write_pcm(path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, SR, 0, "NONE", "not compressed"))
        handle.writeframes(np.clip(samples, -32768, 32767).astype("<i2").tobytes())


def render(scenario, directory: Path, *, renderer, executable, voices, rates, pause_ms, noise_dbfs, seed):
    from scipy.signal import resample_poly
    from .audio import MockTTS

    directory.mkdir(parents=True, exist_ok=True)
    pieces, spans, timings = [], [], []
    offset = 0
    pause = np.zeros(round(pause_ms * SR / 1000), dtype=np.int16)
    for turn in scenario.turns:
        start = perf_counter_ns()
        if renderer == "tone":
            rate, pcm = MockTTS().synthesize(turn.text, turn.source_language)
            samples = np.frombuffer(pcm, dtype="<i2")
        else:
            clip = directory / f"{turn.id}.wav"
            lang = turn.source_language
            subprocess.run([executable, "-b", "1", "-v", voices[lang], "-s", str(rates[lang]),
                            "-w", str(clip), "--stdin"], input=turn.text.encode(),
                           capture_output=True, check=True, timeout=40)
            rate, samples = read_pcm(clip)
        if rate != SR:
            divisor = math.gcd(rate, SR)
            samples = resample_poly(samples.astype(float), SR // divisor, rate // divisor)
        if not len(samples):
            raise ValueError("Renderer returned empty audio")
        timings.append((perf_counter_ns() - start) / 1e6)
        spans.append(Span(turn_id=turn.id, start=offset / SR, end=(offset + len(samples)) / SR))
        pieces.extend((samples, pause))
        offset += len(samples) + len(pause)
    result = np.concatenate(pieces)
    if noise_dbfs is not None:
        noise = np.random.default_rng(seed).normal(0, 32768 * 10 ** (noise_dbfs / 20), size=len(result))
        result = result + noise
    path = directory / "dialogue.wav"
    write_pcm(path, result)
    return path, tuple(spans), timings


def transcribe(path, adapter, scenario, spans, *, model=None, allow_network=False, fault="none"):
    if adapter == "replay":
        turns = {t.id: t for t in scenario.turns}
        speaker_ids = {speaker.id: index for index, speaker in enumerate(scenario.speakers)}
        items = [Heard(turns[s.turn_id].text, s.start, s.end, turns[s.turn_id].source_language,
                       speaker_ids[turns[s.turn_id].speaker_id]) for s in spans]
        if fault == "drop-duplicate":
            items = [items[0], items[0], *items[2:]]
        return items
    if adapter == "whisper":
        from faster_whisper import WhisperModel

        if not model or not Path(model).is_dir():
            raise ValueError("Whisper requires a local model directory")
        local = WhisperModel(str(Path(model).resolve()), device="cpu", compute_type="int8", local_files_only=True)
        segments, _ = local.transcribe(str(path), multilingual=True, vad_filter=True)
        return [Heard(item.text, item.start, item.end, getattr(item, "language", None)) for item in segments]
    if adapter != "deepgram" or not allow_network:
        raise ValueError("Deepgram audio requires explicit --allow-network")
    from ..stt import _dg_post

    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        raise ValueError("Set DEEPGRAM_API_KEY locally for this optional adapter")
    response = _dg_post(path.read_bytes(), key, os.getenv("DEEPGRAM_MODEL", "nova-3"),
                        {"language": "multi", "utterances": "true", "diarize": "true"})
    return [Heard(u["transcript"], u["start"], u["end"], u.get("language") or
                  next((w.get("language") for w in (u.get("words") or []) if w.get("language")), None),
                  u.get("speaker")) for u in response.get("results", {}).get("utterances", [])]
