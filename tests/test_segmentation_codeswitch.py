"""normalize_turns: reagrupa segmentos con code-switching EN/ES en turnos monolingües.

Hasta ahora solo se ejercitaba desde el harness manual src/test_interpreter.py --
nunca desde pytest.
"""
from src.segmentation import detect_lang, normalize_turns, split_sentences
from src.stt import Segment


def test_split_sentences_handles_spanish_punctuation():
    parts = split_sentences("¿Cómo se siente? Me duele mucho.")
    assert len(parts) == 2
    assert "Cómo se siente" in parts[0]
    assert "duele" in parts[1]


def test_detect_lang_short_text_uses_fallback():
    assert detect_lang("No.", fallback="es") == "es"
    assert detect_lang("Ok", fallback="en") == "en"


def test_detect_lang_en_and_es():
    assert detect_lang("The patient has a fever and needs medication") == "en"
    assert detect_lang("El paciente tiene fiebre y necesita medicamento") == "es"


def test_normalize_turns_splits_code_switched_segment():
    # Un solo segmento STT mezclando el final de una frase EN con el inicio de ES.
    segs = [Segment(start=0.0, text="I have a headache. Tengo mucho dolor de cabeza.",
                    lang="en")]
    turns = normalize_turns(segs)
    assert len(turns) == 2
    assert turns[0].lang == "en"
    assert turns[1].lang == "es"


def test_normalize_turns_merges_consecutive_same_language():
    segs = [
        Segment(start=0.0, text="I have a headache.", lang="en"),
        Segment(start=1.0, text="It started yesterday.", lang="en"),
    ]
    turns = normalize_turns(segs)
    assert len(turns) == 1
    assert "headache" in turns[0].text
    assert "yesterday" in turns[0].text


def test_normalize_turns_respects_speaker_boundary():
    segs = [
        Segment(start=0.0, text="I have a headache.", lang="en", speaker=0),
        Segment(start=1.0, text="I see, tell me more.", lang="en", speaker=1),
    ]
    turns = normalize_turns(segs)
    assert len(turns) == 2
    assert turns[0].speaker == 0
    assert turns[1].speaker == 1
