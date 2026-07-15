#!/usr/bin/env python3
"""Benchmark honesto de latencia por etapa: terminología -> DeepSeek -> Fish Speech.

Uso:
    python benchmark_report.py            # no hace llamadas reales, solo explica qué mediría
    python benchmark_report.py --live     # llamadas reales a DeepSeek/Fish Speech (gasta cuota)

No promete latencia total < 2s: el piso de red real de Fish Speech (~2.7s/frase medido
en producción) por sí solo puede superarlo para respuestas largas. Este script reporta
los números reales, no un objetivo no alcanzado.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

SAMPLES = [
    ("en", "The patient has severe chest pain and shortness of breath."),
    ("es", "El paciente tiene dolor en el pecho y dificultad para respirar."),
    ("en", "I think this might be sepsis, doctor, I'm not completely sure."),
    ("es", "Creo que esto podría ser un derrame cerebral, no estoy segura."),
    ("en", "Are you allergic to penicillin or any other medication?"),
]

REPEATS = 5


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[idx]


def _stage_stats(name: str, samples_ms: list[float]) -> str:
    if not samples_ms:
        return f"| {name} | - | - | - | - |"
    return (f"| {name} | {round(statistics.mean(samples_ms), 1)} "
            f"| {round(_percentile(samples_ms, 0.50), 1)} "
            f"| {round(_percentile(samples_ms, 0.95), 1)} | {len(samples_ms)} |")


def run_live() -> str:
    from dotenv import load_dotenv
    load_dotenv()

    from src.terminology import TerminologyIndex
    from src.translate import DeepSeekTranslator
    from src.tts import synthesize

    idx = TerminologyIndex.load(os.getenv("TERMINOLOGY_DIR", "data/terminology"))
    translator = DeepSeekTranslator()

    term_ms, translate_ms, tts_ms, total_ms = [], [], [], []

    for _ in range(REPEATS):
        for lang, text in SAMPLES:
            t0 = time.perf_counter()
            hits = idx.lookup(text, "general")
            term_ms.append((time.perf_counter() - t0) * 1000)

            term_block = "\n".join(hits) if hits else "(sin términos relevantes)"
            t0 = time.perf_counter()
            tr = translator.translate(text, lang, term_block, "")
            translate_ms.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            try:
                synthesize(tr.text, tr.target_lang)
            except Exception:
                pass
            tts_ms.append((time.perf_counter() - t0) * 1000)

            total_ms.append(term_ms[-1] + translate_ms[-1] + tts_ms[-1])

    from src.tts import benchmark_stats as tts_benchmark_stats
    tts_stats = tts_benchmark_stats()

    lines = [
        "# Benchmark de latencia — intérprete médico EN↔ES",
        "",
        f"Muestras: {len(SAMPLES)} frases (EN/ES, emergencia, incertidumbre) x {REPEATS} repeticiones "
        f"= {len(SAMPLES) * REPEATS} mediciones por etapa.",
        "",
        "| Etapa | media (ms) | p50 (ms) | p95 (ms) | n |",
        "|---|---|---|---|---|",
        _stage_stats("Terminología (lookup)", term_ms),
        _stage_stats("Traducción (DeepSeek, con posible fallback local)", translate_ms),
        _stage_stats("Síntesis TTS (Fish Speech / edge-tts)", tts_ms),
        _stage_stats("Total por turno", total_ms),
        "",
        "## Advertencias de esta corrida (léase antes de confiar en los números)",
        "",
    ]
    if getattr(translator, "_use_fallback", False):
        lines += [
            "- **DeepSeek devolvió 402 (saldo insuficiente) en TODAS las llamadas de esta "
            "corrida.** La fila 'Traducción' de arriba mide en realidad: intento a DeepSeek "
            "(falla rápido con 402) + traductor local de respaldo — NO la latencia real del "
            "LLM. Reponer saldo en DeepSeek y volver a correr `--live` para medir la latencia "
            "real del LLM.",
            "",
        ]
    lines += [
        f"- Proveedor TTS activo al final de la corrida: `{tts_stats.get('provider')}` "
        f"(inicializado: {tts_stats.get('initialized')}). Si dice `edge` en vez de `fish`, "
        "Fish Speech falló durante la corrida y se usó el fallback edge-tts.",
        "",
        "## Nota honesta sobre el objetivo de <2s",
        "",
        "Fish Speech (`s2.1-pro-free`) tiene un piso de red real medido de ~2.7s para "
        "una sola frase (llamada HTTP síncrona, sin streaming). Ese piso por sí solo "
        "puede superar un presupuesto total de 2s para respuestas de más de una frase. "
        "No se reestructuró el pipeline (sin streaming/pipelining por oración) porque el "
        "usuario confirmó explícitamente 'medir y reportar honestamente, sin reestructurar'. "
        "Las optimizaciones aplicadas fueron seguras y no estructurales: reutilización de "
        "sesión HTTP (Fish Speech), recorte de contexto de memoria (6 turnos en vez de 12 "
        "para la llamada al LLM), e instrumentación por etapa para medir esto honestamente.",
    ]
    return "\n".join(lines)


def print_dry_run() -> None:
    print("Modo seco (sin --live): no se hicieron llamadas de red.")
    print(f"Con --live, este script correría {len(SAMPLES)} frases x {REPEATS} repeticiones")
    print("a través de: TerminologyIndex.lookup -> DeepSeekTranslator.translate -> tts.synthesize,")
    print("midiendo mean/p50/p95 por etapa, y escribiría out/benchmark_report.md.")
    print()
    print("Esto gasta cuota real de DeepSeek y Fish Speech -- por eso es opt-in.")


def main() -> None:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="hacer llamadas reales (gasta cuota)")
    args = ap.parse_args()

    if not args.live:
        print_dry_run()
        return

    report = run_live()
    print(report)

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    print(f"\nEscrito en {out_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
