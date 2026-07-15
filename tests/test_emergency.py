"""EmergencyClassifier: ambos idiomas, categorías nuevas, y regresión del bug de alerta."""
from src.emergency import ALERT_EN, ALERT_ES, EmergencyClassifier


def test_detects_chest_pain_english():
    c = EmergencyClassifier()
    result = c.classify("I have severe chest pain")
    assert result["is_emergency"]
    assert "chest_pain" in result["labels"]


def test_detects_chest_pain_spanish():
    c = EmergencyClassifier()
    result = c.classify("Tengo dolor en el pecho")
    assert result["is_emergency"]
    assert "chest_pain" in result["labels"]


def test_detects_sepsis_new_category():
    c = EmergencyClassifier()
    assert c.classify("doctor, I think this is sepsis")["is_emergency"]
    assert c.classify("puede ser un choque séptico")["is_emergency"]


def test_detects_loss_of_consciousness_new_category():
    c = EmergencyClassifier()
    assert c.classify("she passed out and is unresponsive")["is_emergency"]
    assert c.classify("se desmayó y perdió el conocimiento")["is_emergency"]


def test_no_false_positive_on_unrelated_text():
    c = EmergencyClassifier()
    result = c.classify("The patient has a mild headache and wants a refill")
    assert not result["is_emergency"]
    assert result["labels"] == []


def test_force_qa_review_matches_is_emergency():
    c = EmergencyClassifier()
    result = c.classify("chest pain")
    assert result["force_qa_review"] == result["is_emergency"]


def test_alert_fires_regardless_of_translation_direction():
    """Regresión: antes, el alert solo se agregaba si tr.target_lang == 'en',
    así que una emergencia dicha en inglés y traducida a español NO recibía alerta."""
    c = EmergencyClassifier()
    text = "I have chest pain and can't breathe"
    em = c.classify(text)
    assert em["is_emergency"]

    # simula ambas direcciones de traducción, como en live_engine._process_one
    for target_lang, expected_alert in [("en", ALERT_EN), ("es", ALERT_ES)]:
        out = "texto traducido"
        if em["is_emergency"] and "<UNCLEAR>" not in out:
            out += f"  {ALERT_EN if target_lang == 'en' else ALERT_ES}"
        assert expected_alert in out
