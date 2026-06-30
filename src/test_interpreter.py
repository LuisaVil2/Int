"""Test offline del intérprete médico contra un audio/subtítulos.

Pipeline (con fixes de FINDINGS.md):
  audio|VTT -> STT -> normalize_turns (FIX #1: turnos monolingües) ->
  [por turno] memoria+terminología -> DeepSeek (FIX #2: non-think, temp0) ->
  emergencia (§11) + confianza/ruteo QA -> tabla.

Uso:
    python -m src.test_interpreter --audio audio/gerd.wav
    python -m src.test_interpreter --vtt audio/clip.en.vtt --max 20

Sin DEEPSEEK_API_KEY: imprime solo la transcripción normalizada (no traduce).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .stt import get_segments
from .segmentation import normalize_turns
from .terminology import TerminologyIndex
from .emergency import EmergencyClassifier, ALERT_EN
from .confidence import ConfidenceEngine, ConfidenceInputs
from .memory import ConversationMemory


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None)
    ap.add_argument("--vtt", default=None)
    ap.add_argument("--max", type=int, default=40, help="máx turnos a traducir")
    ap.add_argument("--specialty", default=os.getenv("DEFAULT_SPECIALTY", "general"))
    args = ap.parse_args()

    print("== STT ==")
    segments = get_segments(args.audio, args.vtt)
    print(f"  {len(segments)} segmentos crudos")

    turns = normalize_turns(segments)
    turns = [t for t in turns if t.text.strip()][: args.max]
    print(f"  {len(turns)} turnos monolingües (post-normalización)")

    idx = TerminologyIndex.load(os.getenv("TERMINOLOGY_DIR", "data/terminology"))
    emerg_clf = EmergencyClassifier()
    conf_engine = ConfidenceEngine()
    memory = ConversationMemory()
    print(f"  terminología: {len(idx.terms)} términos")

    have_key = bool(os.getenv("DEEPSEEK_API_KEY"))
    translator = None
    if have_key:
        from .translate import DeepSeekTranslator
        translator = DeepSeekTranslator()
        print(f"== Traducción: DeepSeek {translator.model} (non-thinking) ==")
    else:
        print("== Traducción OMITIDA: falta DEEPSEEK_API_KEY ==")

    rows = []
    for n, t in enumerate(turns, 1):
        hits = idx.lookup(t.text, args.specialty)
        term_block = "\n".join(hits) if hits else "(sin términos relevantes)"
        row = {"i": n, "t": t.start, "spk": t.speaker, "lang": t.lang, "src": t.text,
               "terms": len(hits), "out": None, "target_lang": None, "confidence": None,
               "unclear": None, "emergency": None, "route": None, "latency_ms": None}

        emer = emerg_clf.classify(t.text)
        row["emergency"] = ",".join(emer["labels"]) if emer["labels"] else ""

        if translator:
            t0 = time.perf_counter()
            try:
                tr = translator.translate(t.text, t.lang, term_block, memory.context())
                out_text = tr.text
                # §11: si emergencia, añade alerta en EN para el proveedor
                if emer["is_emergency"] and tr.target_lang == "en" and "<UNCLEAR>" not in out_text:
                    out_text = f"{out_text}  {ALERT_EN}"
                lat = round((time.perf_counter() - t0) * 1000)
                score = conf_engine.score(ConfidenceInputs(
                    asr_confidence=0.9, llm_confidence=tr.confidence,
                    terminology_certainty=0.95 if hits else 0.85,
                    glossary_match=1.0 if hits else 0.8))
                route = "qa_review" if emer["force_qa_review"] else conf_engine.route(score)
                row.update(out=out_text, target_lang=tr.target_lang, confidence=tr.confidence,
                           unclear=tr.needs_clarification, route=route, latency_ms=lat)
                row["qa_score"] = score
                if not tr.needs_clarification:
                    memory.add_turn(t.speaker, t.lang, t.text, tr.text)
            except Exception as e:  # noqa
                row["out"] = f"[ERROR: {e}]"
        rows.append(row)
        _print_row(row)

    _write_outputs(rows)


def _print_row(r: dict):
    spk = f"spk{r['spk']}" if r["spk"] is not None else "spk?"
    em = f"  ⚠EMERG:{r['emergency']}" if r["emergency"] else ""
    print(f"\n[{r['i']:>3}] t={r['t']}s {spk} {r['lang']} terms={r['terms']}{em}")
    print(f"  SRC: {r['src']}")
    if r["out"] is not None:
        lat = f"{r['latency_ms']}ms" if r["latency_ms"] else ""
        conf = f"conf={r['confidence']}" if r['confidence'] is not None else ""
        route = f" route={r['route']}" if r.get("route") else ""
        print(f"  BOT->{r['target_lang']}: {r['out']}   [{conf} {lat}{route}]")


def _write_outputs(rows: list[dict]):
    out = Path("out")
    out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Resultado test intérprete (con fixes)\n",
          "| # | t(s) | spk | lang | SRC | BOT | conf | route | emerg | ms |",
          "|---|------|-----|------|-----|-----|------|-------|-------|----|"]
    for r in rows:
        src = (r["src"] or "").replace("|", "\\|")
        bot = (r["out"] or "").replace("|", "\\|") if r["out"] else ""
        md.append(f"| {r['i']} | {r['t']} | {r['spk']} | {r['lang']} | {src} | {bot} "
                  f"| {r['confidence'] or ''} | {r.get('route') or ''} | {r['emergency'] or ''} "
                  f"| {r['latency_ms'] or ''} |")
    (out / "result.md").write_text("\n".join(md), encoding="utf-8")
    n_unclear = sum(1 for r in rows if r.get("unclear"))
    n_ok = sum(1 for r in rows if r.get("out") and not r.get("unclear") and "ERROR" not in (r["out"] or ""))
    print(f"\n== {n_ok} OK · {n_unclear} <UNCLEAR> · {len(rows)} turnos -> out/result.md ==")


if __name__ == "__main__":
    main()
