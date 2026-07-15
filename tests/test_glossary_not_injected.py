"""Regresión: el glosario/terminología debe influir la traducción como GUÍA, nunca
reescribir la oración de origen con sustitución palabra-por-palabra.

Bug original reportado: con DeepSeek caído (402), el pipeline caía 100% en
LocalTranslator (diccionario de ~100 palabras), que hace sustitución literal
palabra-por-palabra -- ej. "heart disease" -> "corazón enfermedad" -- y esa salida
rota se hablaba/mostraba como si fuera una traducción real.

Causa raíz real (no era una inyección del glosario en el texto fuente -- eso nunca
ocurrió; el bloque <terminology> siempre viaja separado del texto en el prompt del
LLM): DeepSeek devolvía 402 (sin saldo) en cada llamada. La corrección: cuando el
fallback local tiene baja confianza, se degrada a "<UNCLEAR>" (mismo contrato que el
degrade de DeepSeekTranslator) en vez de presentar la sustitución como traducción.

Este archivo verifica ambos aspectos con 20+ oraciones médicas bilingües reales.
"""
from src.prompts import build_user_message
from src.translate import LocalTranslator

BILINGUAL_MEDICAL_SENTENCES = [
    ("en", "Is there a history of diabetes, heart disease or cancer in your family history?"),
    ("en", "The patient has severe chest pain and shortness of breath."),
    ("es", "El paciente tiene dolor en el pecho y dificultad para respirar."),
    ("en", "Do you have any allergies to medications, especially penicillin?"),
    ("es", "¿Tiene alergia a algún medicamento, especialmente a la penicilina?"),
    ("en", "I need you to take this medication twice a day with food."),
    ("es", "Necesito que tome este medicamento dos veces al día con comida."),
    ("en", "Have you experienced any dizziness, nausea, or vomiting today?"),
    ("es", "¿Ha tenido mareos, náuseas o vómitos hoy?"),
    ("en", "Your blood pressure is a little high, we should monitor it."),
    ("es", "Su presión arterial está un poco alta, debemos monitorearla."),
    ("en", "The surgery went well, and you should rest for two weeks."),
    ("es", "La cirugía salió bien, y debe descansar durante dos semanas."),
    ("en", "Are you currently taking insulin for your diabetes?"),
    ("es", "¿Está tomando insulina actualmente para su diabetes?"),
    ("en", "We need to run some blood tests to check your kidney function."),
    ("es", "Necesitamos hacer unos análisis de sangre para revisar su función renal."),
    ("en", "The patient reports a family history of stroke and heart attack."),
    ("es", "El paciente reporta antecedentes familiares de derrame cerebral e infarto."),
    ("en", "Please describe the pain: is it sharp, dull, or burning?"),
    ("es", "Por favor describa el dolor: ¿es agudo, sordo o ardiente?"),
]


def test_have_at_least_twenty_sentences():
    assert len(BILINGUAL_MEDICAL_SENTENCES) >= 20


def test_llm_prompt_never_mutates_source_sentence():
    """El texto fuente debe viajar VERBATIM en el prompt del LLM -- el glosario es
    guía en un bloque <terminology> separado, nunca reemplaza palabras en el turno."""
    for lang, text in BILINGUAL_MEDICAL_SENTENCES:
        term_block = "- heart disease = enfermedad cardíaca\n- cancer = cáncer\n- diabetes = diabetes"
        prompt = build_user_message(text, lang, term_block, context="")

        assert text in prompt, "el texto original debe aparecer intacto en el prompt"
        # el texto original no debe tener sus palabras individuales reemplazadas
        # antes de llegar al prompt -- el bloque de terminología está separado.
        term_section = prompt.split("<terminology>")[1].split("</terminology>")[0]
        turno_section = prompt.split("TURNO:")[1]
        assert text.strip() in turno_section
        assert term_block in term_section


def test_local_fallback_never_presents_low_confidence_substitution_as_translation():
    """Para cada oración, o bien el fallback local tuvo confianza suficiente para
    devolver algo razonable (needs_clarification=False), o se degrada honestamente a
    <UNCLEAR> -- nunca hay un término medio donde se muestra la sustitución rota con
    needs_clarification=True (que era exactamente el bug reportado)."""
    t = LocalTranslator()
    for lang, text in BILINGUAL_MEDICAL_SENTENCES:
        result = t.translate(text, lang, "")
        assert result.target_lang != (lang.lower()[:2])
        if result.needs_clarification:
            assert result.text == "<UNCLEAR>", (
                f"baja confianza pero no se degradó a <UNCLEAR>: {text!r} -> {result.text!r}"
            )
        else:
            assert result.text != "<UNCLEAR>"


def test_exact_reported_bug_sentence_no_longer_leaks_word_salad():
    t = LocalTranslator()
    result = t.translate(
        "Is there a history of diabetes, heart disease or cancer in your family history?",
        "en", "")
    assert result.text != ("Is there a history of diabetes, corazón enfermedad or "
                           "cáncer in your family history?")
    assert "corazón enfermedad" not in result.text
