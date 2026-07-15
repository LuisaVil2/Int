import numpy as np

from src import tts as tts_module


def test_load_config_reads_fish_defaults(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "tts:\n"
        "  provider: fish\n"
        "  language_auto: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tts_module, "_CONFIG_PATH", cfg)

    config = tts_module.load_tts_config()

    assert config["provider"] == "fish"
    assert config["language_auto"] is True


def test_synthesize_falls_back_to_edge_when_fish_fails(monkeypatch):
    class DummyFishProvider:
        def synthesize(self, text, lang):
            raise RuntimeError("fish speech unavailable")

    class DummyEdgeProvider:
        def synthesize(self, text, lang):
            return np.ones(4, dtype=np.float32), 16000

    monkeypatch.setattr(tts_module, "FishSpeechProvider", DummyFishProvider)
    monkeypatch.setattr(tts_module, "EdgeTTSProvider", DummyEdgeProvider)

    samples, sr = tts_module.synthesize("Hola", "es")

    assert sr == 16000
    assert samples.shape == (4,)
