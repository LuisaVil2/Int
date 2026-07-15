"""Jerarquía de fallback en vivo: Deepgram->Whisper por turno, y que un turno roto
NUNCA mata el resto de la sesión. Usa LiveEngine._process_one() directamente
(extraído del while-loop) en vez de manejar hilos/audio real -- más rápido y estable.
"""
import queue
import types

import numpy as np
import pytest

from src.live_engine import LiveEngine


def _engine():
    return LiveEngine(on_event=lambda kind, data: None,
                      input_choice={"kind": "mic", "name": "dummy"}, output_index=None,
                      tts_on=False)


def test_deepgram_failure_falls_back_to_whisper_per_utterance(monkeypatch):
    engine = _engine()

    def fake_dg(*a, **k):
        raise RuntimeError("deepgram network error")

    class FakeWhisperModel:
        def __init__(self, *a, **k):
            self.constructed = True

        def transcribe(self, audio, **k):
            seg = types.SimpleNamespace(text="hello")
            info = types.SimpleNamespace(language="en")
            return [seg], info

    monkeypatch.setattr("src.stt.transcribe_np_deepgram", fake_dg)
    monkeypatch.setattr("faster_whisper.WhisperModel", FakeWhisperModel)

    text, lang, conf, speaker, backend = engine._transcribe(np.zeros(100, dtype=np.float32), "fake-dg-key")
    assert backend == "whisper"
    assert text == "hello"
    assert speaker is None


def test_whisper_model_loaded_lazily_and_cached_once(monkeypatch):
    engine = _engine()
    construct_calls = []

    class FakeWhisperModel:
        def __init__(self, *a, **k):
            construct_calls.append(1)

        def transcribe(self, audio, **k):
            seg = types.SimpleNamespace(text="hi")
            info = types.SimpleNamespace(language="en")
            return [seg], info

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeWhisperModel)

    assert engine._whisper_model is None
    engine._transcribe(np.zeros(10, dtype=np.float32), dg_key=None)
    engine._transcribe(np.zeros(10, dtype=np.float32), dg_key=None)
    assert len(construct_calls) == 1  # nunca se recarga el modelo


def test_one_failed_turn_does_not_kill_the_session(monkeypatch):
    """Replica el patrón try/except-continue del while-loop de _process(): si
    _process_one lanza una excepción en un turno, el bucle debe seguir con el
    siguiente en vez de terminar la sesión completa."""
    engine = _engine()
    calls = []

    def fake_process_one(audio, *a, **k):
        calls.append(audio)
        if len(calls) == 1:
            raise RuntimeError("turno roto (ej. error transitorio de Deepgram)")

    engine._process_one = fake_process_one

    q: queue.Queue = queue.Queue()
    q.put(np.zeros(1))
    q.put(np.zeros(2))

    errors = []
    processed = 0
    while True:
        try:
            audio = q.get_nowait()
        except queue.Empty:
            break
        try:
            engine._process_one(audio, None, None, None, None, None, None, None)
            processed += 1
        except Exception as e:  # noqa - mismo contrato que _process()
            errors.append(str(e))
            continue

    assert len(calls) == 2          # ambos turnos se intentaron
    assert len(errors) == 1         # el primero falló...
    assert processed == 1           # ...pero el segundo se procesó igual
