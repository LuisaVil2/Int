import json
import unittest
from types import SimpleNamespace

from src.confidence import ConfidenceEngine
from src.emergency import EmergencyClassifier
from src.live_engine import LiveEngine
from src.memory import ConversationMemory
from src.stt import DeepgramLive


def result(text, start, *, final=True, endpoint=False):
    return json.dumps({
        "type": "Results", "start": start, "duration": 0.5,
        "is_final": final, "speech_final": endpoint,
        "channel": {"alternatives": [{"transcript": text, "confidence": 0.9}]},
    })


class EngineReproductions(unittest.TestCase):
    def test_stream_only_delivers_unique_final_ranges(self):
        turns = []
        client = DeepgramLive("fixture", on_utterance=lambda *turn: turns.append(turn))
        first = result("I did not", 0)
        client._ws = iter([
            result("I did", 0, final=False), first, first,
            result("take it.", 0.5, endpoint=True),
            result("take it.", 0.5, endpoint=True),
            result("No.", 1, endpoint=True),
            result("No.", 1.5, endpoint=True),
        ])
        client._reconnect = lambda *args: False
        client._recv_loop()
        self.assertEqual([turn[0] for turn in turns], ["I did not take it.", "No.", "No."])

    def set_up_turn(self, text, lang, translator):
        events = []
        engine = LiveEngine(lambda kind, data: events.append((kind, data)), {}, None)
        engine._process_text_turn(
            text, lang, 0.99, None, "fixture", None, translator,
            SimpleNamespace(lookup=lambda *args: ["term"] * 5),
            lambda text, fallback=None: fallback,
            ConversationMemory(), EmergencyClassifier(), ConfidenceEngine(), stream_tts=True,
        )
        return engine, events

    def test_invalid_stream_prefix_never_reaches_tts(self):
        def translate(*args, on_text=None):
            if on_text:
                on_text("Tomé quince miligramos. ")
            return SimpleNamespace(text="<UNCLEAR>", target_lang="es",
                                   confidence=0.0, needs_clarification=True)

        engine, events = self.set_up_turn("I took 50 mg.", "en", SimpleNamespace(translate=translate))
        self.assertTrue(engine._tts_q.empty())
        self.assertIn("needs_review", [kind for kind, _ in events])

    def test_unknown_direction_holds_turn(self):
        calls = []

        def translate(*args, **kwargs):
            calls.append(args)
            return SimpleNamespace(text="No", target_lang="es",
                                   confidence=1.0, needs_clarification=False)

        engine, events = self.set_up_turn("No.", None, SimpleNamespace(translate=translate))
        self.assertEqual(calls, [])
        self.assertTrue(engine._tts_q.empty())
        self.assertIn("needs_review", [kind for kind, _ in events])
