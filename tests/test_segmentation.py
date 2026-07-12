from src.segmentation import detect_lang, normalize_turns, split_sentences
from src.stt import Segment


def test_split_sentences_basic():
    assert split_sentences("Hello there. How are you?") == ["Hello there.", "How are you?"]


def test_split_sentences_spanish_marks():
    out = split_sentences("Buenos días doctor. ¿Cómo está? Me duele el pecho.")
    assert out == ["Buenos días doctor.", "¿Cómo está?", "Me duele el pecho."]


def test_split_sentences_empty():
    assert split_sentences("   ") == []


def test_detect_lang_en_es():
    assert detect_lang("The patient has severe chest pain and needs treatment") == "en"
    assert detect_lang("El paciente tiene dolor de pecho desde hace tres días") == "es"


def test_detect_lang_short_uses_fallback():
    assert detect_lang("No.", fallback="es") == "es"
    assert detect_lang("Ok.", fallback="en") == "en"


def test_normalize_turns_splits_code_switched_segment():
    # Hallazgo #1: fin de frase EN + inicio ES en el mismo segmento STT
    segs = [Segment(start=0.0, lang="en",
                    text="Good morning, how can I help you today? "
                         "Buenos días doctor, últimamente he tenido mucha acidez estomacal.")]
    turns = normalize_turns(segs)
    assert [t.lang for t in turns] == ["en", "es"]
    assert turns[0].text.startswith("Good morning")
    assert turns[1].text.startswith("Buenos días")


def test_normalize_turns_merges_same_lang():
    segs = [Segment(start=0.0, lang="en", text="I see. Let me check your throat."),
            Segment(start=2.0, lang="en", text="Please open your mouth.")]
    turns = normalize_turns(segs)
    assert len(turns) == 1
    assert turns[0].lang == "en"
