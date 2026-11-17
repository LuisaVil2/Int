"""Conservative, auditable anchors for a small bilingual clinical vocabulary."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

ATOMS = {
    "bleeding": r"bleed(?:ing)?|hemorrhage|sangrado|sangr(?:e|ado)|hemorragia",
    "cramps": r"cramps?|colicos?|calambres?",
    "chest_pain": r"chest pain|dolor (?:en el|de) pecho",
    "breathing": r"short(?:ness)? of breath|trouble breathing|dificultad para respirar|falto el aire",
    "allergy": r"allerg(?:ic|y|ies)(?: to)?|alerg(?:ic[oa]s?|ias?)(?: a)?",
}
ANATOMY = {"arm": r"arm|brazo", "leg": r"leg|pierna", "eye": r"eye|ojo",
           "ear": r"ear|oido", "hand": r"hand|mano", "side": r"side|lado"}
ROUTES = {"oral": r"by mouth|orally|por via oral|por boca|via oral",
          "iv": r"intravenous(?:ly)?|intravenos[ao]|iv",
          "im": r"intramuscular(?:ly)?|im",
          "sc": r"subcutaneous(?:ly)?|subcutane[ao]|under the skin|debajo de la piel"}
FREQUENCY = {"twice_daily": r"twice (?:a|per) day|(?:2|two) times a day|(?:2|dos) veces al dia",
             "daily": r"once (?:a|per) day|once daily|(?:1|one) time a day|una vez al dia|each day|every day|cada dia",
             "weekly": r"once a week|every week|una vez a la semana|cada semana",
             "as_needed": r"as needed|segun (?:sea )?necesari[oa]"}
DRUGS = {"metformin": r"metformin|metformina", "warfarin": r"warfarin|warfarina",
         "hydroxyzine": r"hydroxyzine|hidroxizina", "hydralazine": r"hydralazine|hidralazina",
         "penicillin": r"penicillin|penicilina", "insulin": r"insulin|insulina"}
UNITS = {
    "mg": r"mg|milligrams?|miligramos?", "g": r"g|grams?|gramos?",
    "mcg": r"mcg|[µμ]g|micrograms?|microgramos?",
    "ml": r"ml|millilit(?:er|re)s?|mililitros?",
    "days": r"days?|dias?", "weeks": r"weeks?|semanas?",
    "hours": r"hours?|horas?", "months": r"months?|mes(?:es)?",
}
_EN = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_ES = "cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciseis diecisiete dieciocho diecinueve".split()
_SMALL = {word: value for items in (_EN, _ES) for value, word in enumerate(items)}
_SMALL.update({"un": 1, "una": 1, "veinte": 20, "cien": 100, "ciento": 100})
for i, word in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2):
    _SMALL[word] = i * 10
for i, word in enumerate("treinta cuarenta cincuenta sesenta setenta ochenta noventa".split(), 3):
    _SMALL[word] = i * 10
for i, word in enumerate("veintiuno veintidos veintitres veinticuatro veinticinco veintiseis veintisiete veintiocho veintinueve".split(), 21):
    _SMALL[word] = i
for i, word in enumerate("doscientos trescientos cuatrocientos quinientos seiscientos setecientos ochocientos novecientos".split(), 2):
    _SMALL[word] = i * 100
_WORD = "|".join(sorted(_SMALL, key=lambda w: (-len(w), w)))
_NUMBER = rf"(?<!\w)(?:[+-]?\d+(?:[.,]\d+)?|(?:{_WORD})(?:[ -](?:{_WORD}|hundred|and|y))*?)"
_QTY = re.compile(rf"(?P<value>{_NUMBER})\s*(?:"
                  + "|".join(f"(?P<{unit}>{aliases})" for unit, aliases in UNITS.items()) + r")\b")
_MONTHS = ("january enero|february febrero|march marzo|april abril|may mayo|june junio|july julio|"
           "august agosto|september septiembre|october octubre|november noviembre|december diciembre")
_MONTH = {word: i for i, pair in enumerate(_MONTHS.split("|"), 1) for word in pair.split()}
_MONTH_PATTERN = "|".join(_MONTH)
_DATE = re.compile(rf"(?P<day>{_NUMBER})(?:st|nd|rd|th)?\s+(?:de\s+)?(?P<month>{_MONTH_PATTERN})\b"
                   rf"|\b(?P<month_first>{_MONTH_PATTERN})\s+(?P<day_last>{_NUMBER})(?:st|nd|rd|th)?\b")
_NEG = r"\b(?:no|not|never|without|sin|nunca|jam[aá]s|deny|denies|denied|niego|niega)\b|n['’]t\b"
_CLAUSES = re.compile(r"[!?;]|(?<!\d)[.,]|[.,](?!\d)|\b(?:but|however|pero|aunque|sin embargo)\b")


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(c) != "Mn")


def contains(text, pattern):
    return re.search(rf"(?<!\w)(?:{pattern})(?!\w)", text) is not None


def number(raw):
    if raw.lstrip("+-")[0].isdigit():
        # A single separator followed by three digits may mark thousands or a decimal.
        if re.search(r"[.,]\d{3}$", raw):
            return raw
        return raw.replace(",", ".")
    words = re.split(r"[ -]+", raw)
    head = _SMALL.get(words[0])
    if len(words) == 1:
        return str(head) if head is not None else None
    hundreds = 0
    if words[1] == "hundred" and head in range(1, 10):
        hundreds, words = head * 100, words[2:]
        if words == ["and"]:
            return None
        if words[:1] == ["and"]:
            words = words[1:]
    elif head in range(100, 1000):
        hundreds, words = head, words[1:]
    if not words:
        return str(hundreds)
    head = _SMALL.get(words[0])
    if len(words) == 1 and head in range(100):
        return str(hundreds + head)
    if len(words) == 3 and words[1] == "y":
        words = [words[0], words[2]]
    if len(words) != 2:
        return None
    tail = _SMALL.get(words[1])
    if head in range(20, 100, 10) and tail in range(1, 10):
        return str(hundreds + head + tail)
    return None


def quantity_sequence(text):
    return tuple((m.lastgroup, value) for m in _QTY.finditer(fold(text))
                 if (value := number(m["value"])) is not None)


def equivalent_quantities(source, candidate):
    sequence = quantity_sequence(source)
    return (bool(sequence) and sequence == quantity_sequence(candidate)
            and all(not re.search(r"\d", _QTY.sub("", fold(text))) for text in (source, candidate)))


def dates(text):
    return {(_MONTH[m["month"] or m["month_first"]], day) for m in _DATE.finditer(fold(text))
            if (day := number(m["day"] or m["day_last"])) is not None}


def inventory(text):
    folded = fold(text)
    sides, polarity = defaultdict(set), defaultdict(set)
    for clause in _CLAUSES.split(folded):
        organs = {key for key, value in ANATOMY.items() if contains(clause, value)}
        left = contains(clause, r"left|izquierd[oa]")
        right = contains(clause, r"right|derech[oa]")
        if len(organs) == 1 and left != right:
            sides[next(iter(organs))].add("left" if left else "right")
        concepts = {key for key, value in ATOMS.items() if contains(clause, value)}
        negated = re.search(_NEG, clause)
        # A coordinated list with possible mixed scope needs human interpretation.
        if len(concepts) == 1 and not contains(clause, r"not only|no solo"):
            concept = next(iter(concepts))
            mention = re.search(ATOMS[concept], clause)
            if not negated or negated.start() < mention.start():
                polarity[concept].add(bool(negated))
    return {
        "sides": dict(sides), "polarity": dict(polarity),
        "quantity": set(quantity_sequence(text)), "dates": dates(text),
        "route": {key for key, value in ROUTES.items() if contains(folded, value)},
        "frequency": {key for key, value in FREQUENCY.items() if contains(folded, value)},
        "drugs": {key for key, value in DRUGS.items() if contains(folded, value)},
    }


def meaning_flags(source: str, candidate: str, reference: str | None = None):
    """Yield rule/category pairs only for clearly recognized evidence."""
    a, b = inventory(source), inventory(candidate)
    for key in sorted(a["sides"].keys() & b["sides"].keys()):
        if len(a["sides"][key]) == len(b["sides"][key]) == 1 and a["sides"][key] != b["sides"][key]:
            yield "laterality_changed", "mistranslation"
            break
    for key in sorted(a["polarity"].keys() & b["polarity"].keys()):
        if len(a["polarity"][key]) == len(b["polarity"][key]) == 1 and a["polarity"][key] != b["polarity"][key]:
            yield ("allergy_polarity_changed" if key == "allergy" else "negation_scope_changed"), "negation_reversal"
            break
    for key, category in (("route", "medication_or_dose"), ("frequency", "medication_or_dose"),
                          ("drugs", "medication_or_dose"),
                          ("dates", "numbers_or_units")):
        if a[key] and b[key] and a[key] != b[key]:
            yield f"{key}_changed", category
    if a["quantity"] and b["quantity"] and a["quantity"] != b["quantity"]:
        categories = {unit for unit, _ in a["quantity"] | b["quantity"]}
        yield "quantity_changed", ("medication_or_dose" if categories & {"mg", "g", "mcg", "ml"}
                                    else "numbers_or_units")
    first = r"i|yo|tomo|tome|tengo|soy|siento"
    third = r"the patient|el paciente|la paciente|he (?:takes|has|feels)|she (?:takes|has|feels)"
    if (contains(fold(source), first) and contains(fold(candidate), third)
            and not contains(fold(candidate), first)
            and (reference is None or contains(fold(reference), first))):
        yield "first_person_lost", "speaker_attribution"
