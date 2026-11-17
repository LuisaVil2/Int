import json
import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np

from src.live_engine import FRAME, LiveEngine
from src.stt import DeepgramLive, _FinalResults


def event(text, start, end, *, final=True, endpoint=False, words=None, languages=None, confidence=0.9):
    return {"type": "Results", "start": start, "duration": end - start,
            "is_final": final, "speech_final": endpoint,
            "channel": {"alternatives": [{"transcript": text, "words": words or [],
                                         "languages": languages or [], "confidence": confidence}]}}


def test_endpoint_fallback_flushes_once_and_preserves_zero_confidence():
    buffer = _FinalResults()
    buffer.accept(event("Sí.", 0.0, 0.5, languages=["es-MX"], confidence=0))
    assert buffer.accept({"type": "UtteranceEnd", "last_word_end": 0.1}) == (None, None)
    boundary = {"type": "UtteranceEnd", "last_word_end": 0.5}
    assert buffer.accept(boundary) == (("Sí.", "es", 0.0), None)
    assert buffer.accept(boundary) == (None, None)
    assert buffer.accept(event("Sí.", 0.0, 0.5, endpoint=True)) == (None, None)


def test_unfinalized_tail_and_overlapping_final_ranges_require_review():
    buffer = _FinalResults()
    buffer.accept(event("I did not", 0.0, 1.0))
    buffer.accept(event("take it.", 1.0, 1.9, final=False))
    assert buffer.accept({"type": "UtteranceEnd", "last_word_end": 1.0}) == (
        None, "stt_incomplete_turn")
    buffer.accept(event("Fifty", 2.0, 3.0))
    assert buffer.accept(event("Fifty milligrams.", 2.5, 3.5, endpoint=True)) == (
        None, "stt_incomplete_turn")


def test_word_language_tags_and_speakers_are_local_to_each_endpoint():
    buffer = _FinalResults()
    words = [{"word": "tomé", "language": "es", "speaker": 0},
             {"word": "metformin", "language": "en", "speaker": 0}]
    assert buffer.accept(event("Tomé metformin.", 0.0, 0.5, endpoint=True, words=words)) == (
        ("Tomé metformin.", "es", 0.9), None)
    words[1]["speaker"] = 1
    assert buffer.accept(event("Tomé metformin.", 0.6, 1.2, endpoint=True, words=words)) == (
        None, "stt_multiple_speakers")
    assert buffer.accept(event("Yes.", 1.3, 1.8, endpoint=True,
                               words=[{"word": "yes", "language": "en", "speaker": 1}])) == (
        ("Yes.", "en", 0.9), None)


def test_reconnect_discards_both_sides_of_interrupted_turn():
    class Socket:
        def __init__(self, *messages):
            self.messages = messages
            self.closed = False

        def __iter__(self):
            return (json.dumps(msg) for msg in self.messages)

        def close(self):
            self.closed = True

    original = Socket(event("I did not", 0.0, 0.5))
    replacement = Socket(event("take it.", 0.0, 0.6, endpoint=True),
                         event("Sí.", 0.7, 1.2, endpoint=True, languages=["es"]))
    turns, reviews, errors = [], [], []
    client = DeepgramLive("fixture", on_utterance=lambda *turn: turns.append(turn),
                          on_gap=reviews.append, on_error=errors.append)
    client.RECONNECT_DELAYS = (0,)
    client._ws = original
    next_socket = iter([replacement])
    client._connect = lambda: next(next_socket)
    client._recv_loop()
    assert original.closed and replacement.closed and client.closed
    assert turns == [("Sí.", "es", 0.9)]
    assert reviews == ["stt_connection_lost", "stt_incomplete_turn", "stt_connection_lost"]
    assert len(errors) == 1


def test_capture_frame_overlapping_playback_is_muted_even_if_playback_ended():
    engine = LiveEngine(lambda *args: None, {}, None)

    def record(size):
        engine._playback_epoch += 2
        engine._stop.set()
        return np.ones(size)

    _, muted = engine._read_frame(SimpleNamespace(record=record))
    assert muted


def test_streaming_capture_substitutes_silence_for_muted_frame():
    engine = LiveEngine(lambda *args: None, {}, None)
    frames, flags = [], []

    def record(size):
        if frames:
            engine._stop.set()
        engine._speaking.set()
        return np.ones(size)

    engine._dg = SimpleNamespace(closed=False, set_capture_muted=flags.append, send_np=frames.append)
    engine._capture_streaming(SimpleNamespace(record=record))
    assert flags == [True]
    assert len(frames) == 1 and np.all(frames[0] == 0)


def test_playback_fifo_and_unmutes_during_synthesis_gap(monkeypatch):
    engine = LiveEngine(lambda *args: None, {}, None)
    played = []
    first, second = np.ones(FRAME), np.full(FRAME, 2)

    def play(audio, sr, device):
        assert engine._speaking.is_set()
        played.append(audio[0])

    def wait():
        if len(played) == 2:
            engine._stop.set()

    def emit(kind, **data):
        if kind == "status" and data["text"].startswith("Escuchando"):
            assert not engine._speaking.is_set()
            if len(played) == 1:
                engine._play_q.put((second, 16000, 2))

    monkeypatch.setattr(engine, "_emit", emit)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(play=play, wait=wait))
    engine._play_q.put((first, 16000, 1))
    engine._tts_play()
    assert played == [1, 2]
    assert not engine._speaking.is_set()


def test_stop_drops_synthesis_that_finishes_after_cancel(monkeypatch):
    engine = LiveEngine(lambda *args: None, {}, None)
    entered = threading.Event()

    def synthesize(*args):
        entered.set()
        assert engine._stop.wait(2)
        return np.ones(4), 16000

    monkeypatch.setitem(sys.modules, "src.tts", SimpleNamespace(synthesize=synthesize))
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(stop=lambda: None))
    engine._tts_q.put(("Sí.", "es", 1))
    worker = threading.Thread(target=engine._tts_synth, daemon=True)
    engine._threads = [worker]
    worker.start()
    assert entered.wait(2)
    engine.stop()
    assert not worker.is_alive()
    assert engine._play_q.empty()


def test_process_worker_continues_after_failed_and_unknown_turns(monkeypatch):
    from src.terminology import TerminologyIndex
    from src.translate import DeepSeekTranslator, Translation

    events = []
    engine = LiveEngine(lambda kind, data: events.append((kind, data)), {}, None)

    def translate(text, lang, *args):
        if text != "Yes.":
            raise RuntimeError("fixture provider error")
        assert lang == "en"
        return Translation(target_lang="es", text="Sí.", confidence=1.0)

    original_get = engine._text_q.get

    def get(timeout):
        try:
            return original_get(block=False)
        except queue.Empty:
            engine._stop.set()
            raise

    monkeypatch.setattr(DeepSeekTranslator, "__init__", lambda self: setattr(self, "model", "fake"))
    monkeypatch.setattr(DeepSeekTranslator, "translate", lambda self, *args: translate(*args))
    monkeypatch.setattr(TerminologyIndex, "load",
                        lambda _: SimpleNamespace(lookup=lambda *args: ["term"] * 5))
    monkeypatch.setattr(engine._text_q, "get", get)
    for text, lang in (("I have chest pain.", "en"), ("No.", None), ("Yes.", "en")):
        engine._on_live_text(text, lang, 1.0)
    engine._process()

    reviews = [data for kind, data in events if kind == "needs_review"]
    assert [data["reason"] for data in reviews] == [
        "translation_failed", "unknown_source_language"
    ]
    assert reviews[0]["emergency_labels"] == ["chest_pain"]
    assert engine._tts_q.get_nowait() == ("Sí.", "es", 3)
    assert engine._tts_q.empty()
