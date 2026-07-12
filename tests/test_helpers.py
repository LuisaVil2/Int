"""Tests de helpers puros: splitter de frases, emergencias, confianza, memoria,
terminología y utilidades de stt (VTT, wav)."""
import io
import wave

import numpy as np

from src.confidence import ConfidenceEngine, ConfidenceInputs
from src.emergency import EmergencyClassifier
from src.live_engine import _SentenceSplitter
from src.memory import ConversationMemory
from src.stt import load_vtt, np_to_wav_bytes
from src.terminology import TerminologyIndex, _contains


# ---------- _SentenceSplitter ----------
def test_splitter_emits_full_sentences():
    out = []
    sp = _SentenceSplitter(out.append)
    for d in ["Buenos días, doct", "or. Últimamente he ten", "ido acidez. ¿Qué hago"]:
        sp.feed(d)
    sp.flush()
    assert out == ["Buenos días, doctor.", "Últimamente he tenido acidez.", "¿Qué hago"]


def test_splitter_holds_tiny_fragments():
    out = []
    sp = _SentenceSplitter(out.append)
    sp.feed("Sí. Con la comida me arde el pecho. ")
    sp.flush()
    assert out[0].startswith("Sí. Con")   # "Sí." solo no se emite: espera contexto


# ---------- emergencias ----------
def test_emergency_detects_en_and_es():
    c = EmergencyClassifier()
    assert c.classify("I have severe chest pain")["is_emergency"]
    assert "chest_pain" in c.classify("tengo dolor de pecho")["labels"]
    assert c.classify("no puedo respirar bien")["is_emergency"]


def test_emergency_negative():
    r = EmergencyClassifier().classify("me duele un poco la cabeza")
    assert not r["is_emergency"]
    assert r["labels"] == []


# ---------- confianza ----------
def test_confidence_score_and_routes():
    eng = ConfidenceEngine()
    perfect = eng.score(ConfidenceInputs(1, 1, 1, 1, 1))
    assert perfect == 100
    assert eng.route(perfect) == "automatic_approval"
    assert eng.route(85) == "qa_review"
    assert eng.route(50) == "pause_manual_approval"


def test_confidence_clamps_out_of_range():
    assert ConfidenceEngine().score(ConfidenceInputs(5, -1, 1, 1, 1)) <= 100


# ---------- memoria ----------
def test_memory_maxlen_and_context():
    m = ConversationMemory(max_turns=2)
    assert "sin contexto" in m.context()
    m.add_turn(None, "en", "hello", "hola")
    m.add_turn(None, "es", "adiós", "goodbye")
    m.add_turn(None, "en", "thanks", "gracias")
    ctx = m.context()
    assert "hello" not in ctx          # expulsado por maxlen
    assert "[en] thanks -> gracias" in ctx


# ---------- terminología ----------
def test_terminology_lookup_real_glossaries():
    idx = TerminologyIndex.load("data/terminology")
    assert len(idx.terms) > 0
    hits = idx.lookup("The patient reports chest pain and takes Tylenol")
    assert any("Tylenol" in h for h in hits)


def test_contains_short_keys_word_boundary():
    assert _contains("patient takes ppi daily", "ppi")
    assert not _contains("hippie lifestyle", "ppi")   # sigla no matchea substring


# ---------- stt utils ----------
def test_np_to_wav_roundtrip():
    samples = (np.sin(np.linspace(0, 100, 1600)) * 0.5).astype(np.float32)
    with wave.open(io.BytesIO(np_to_wav_bytes(samples, 16000)), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        back = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert back.size == samples.size
    assert np.allclose(back / 32767.0, samples, atol=1e-3)


def test_load_vtt(tmp_path):
    vtt = tmp_path / "s.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\nHello <c>there</c>\n\n"
        "00:00:03.000 --> 00:00:05.000\nHello there\n\n"   # duplicado -> dedup
        "00:00:05.500 --> 00:00:07.000\nBuenos días\n\n",
        encoding="utf-8",
    )
    segs = load_vtt(vtt)
    assert [s.text for s in segs] == ["Hello there", "Buenos días"]
    assert segs[1].start == 5.5
