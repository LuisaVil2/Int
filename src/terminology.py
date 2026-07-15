"""Carga e indexa glosarios de data/terminology/. Ver Instrucciones.md §8.

lookup(transcript, specialty) -> list[term] selecciona solo términos relevantes
(match por keyword) para inyectar en el bloque <terminology>, sin volcar todo el glosario.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Term:
    src: str          # término/keyword que dispara el match (lower)
    line: str         # línea formateada para el prompt
    specialty: str


class TerminologyIndex:
    def __init__(self, terms: list[Term]):
        self.terms = terms

    @classmethod
    def load(cls, root: str | Path) -> "TerminologyIndex":
        root = Path(root)
        terms: list[Term] = []

        g = root / "glossary_en_es.csv"
        if g.exists():
            for r in _read(g):
                en, es = r.get("term_en", ""), r.get("term_es", "")
                if not en or not es:
                    continue
                notes = r.get("notes", "")
                line = f"- {en} = {es}" + (f"  ({notes})" if notes else "")
                terms.append(Term(en.lower(), line, r.get("specialty", "general")))

        a = root / "abbreviations.csv"
        if a.exists():
            for r in _read(a):
                ab = r.get("abbr", "")
                if not ab:
                    continue
                line = f"- {ab} = {r.get('expansion_en','')} / {r.get('expansion_es','')}"
                terms.append(Term(ab.lower(), line, r.get("specialty", "general")))

        d = root / "drug_names.csv"
        if d.exists():
            for r in _read(d):
                brand, generic = r.get("brand", ""), r.get("generic", "")
                if not brand:
                    continue
                notes = r.get("notes", "")
                line = f"- {brand} ({generic}) = {r.get('name_es','')}" + (f"  ({notes})" if notes else "")
                # se dispara tanto por marca como por genérico
                terms.append(Term(brand.lower(), line, "drug"))
                if generic:
                    terms.append(Term(generic.lower(), line, "drug"))

        return cls(terms)

    def lookup(self, transcript: str, specialty: str | None = None, limit: int = 25) -> list[str]:
        text = transcript.lower()
        hits: list[str] = []
        seen: set[str] = set()
        # "general" (o sin especialidad) = sin filtro: todas las especialidades son
        # candidatas. Solo se restringe cuando se pide una especialidad ESPECÍFICA
        # (ej. "cardiology"), que además de lo suyo también ve "general" y "drug".
        narrow = specialty and specialty != "general"
        for t in self.terms:
            if narrow and t.specialty not in (specialty, "general", "drug"):
                continue
            # match por palabra/substring del keyword
            if _contains(text, t.src) and t.line not in seen:
                seen.add(t.line)
                hits.append(t.line)
                if len(hits) >= limit:
                    break
        return hits


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if not (row.get(next(iter(row))) or "").startswith("#")]


def _contains(text: str, key: str) -> bool:
    if len(key) <= 4:  # siglas: match por palabra exacta para no spamear
        return re.search(rf"\b{re.escape(key)}\b", text) is not None
    return key in text


if __name__ == "__main__":
    import sys
    idx = TerminologyIndex.load("data/terminology")
    print(f"cargados {len(idx.terms)} términos")
    sample = "The patient has chest pain and high blood pressure, taking Tylenol PRN"
    for line in idx.lookup(sample):
        print(" ", line)
