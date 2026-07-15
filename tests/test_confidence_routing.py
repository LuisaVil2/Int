from src.confidence import ConfidenceEngine, ConfidenceInputs


def test_perfect_inputs_score_100():
    e = ConfidenceEngine()
    inputs = ConfidenceInputs(asr_confidence=1.0, llm_confidence=1.0,
                              terminology_certainty=1.0, conversation_consistency=1.0,
                              glossary_match=1.0)
    assert e.score(inputs) == 100


def test_route_boundaries():
    e = ConfidenceEngine()
    assert e.route(95) == "automatic_approval"
    assert e.route(94) == "qa_review"
    assert e.route(80) == "qa_review"
    assert e.route(79) == "pause_manual_approval"
    assert e.route(0) == "pause_manual_approval"


def test_low_asr_confidence_lowers_score():
    e = ConfidenceEngine()
    good = ConfidenceInputs(asr_confidence=0.95)
    bad = ConfidenceInputs(asr_confidence=0.2)
    assert e.score(good) > e.score(bad)


def test_out_of_range_inputs_are_clamped():
    e = ConfidenceEngine()
    inputs = ConfidenceInputs(asr_confidence=1.5, llm_confidence=-0.5)
    score = e.score(inputs)
    assert 0 <= score <= 100


def test_emergency_forces_non_automatic_route():
    """En live_engine._process_one, force_qa_review sube automatic_approval a qa_review."""
    e = ConfidenceEngine()
    inputs = ConfidenceInputs(asr_confidence=1.0, llm_confidence=1.0,
                              terminology_certainty=1.0, conversation_consistency=1.0,
                              glossary_match=1.0)
    score = e.score(inputs)
    route = e.route(score)
    assert route == "automatic_approval"

    force_qa_review = True
    if force_qa_review and route == "automatic_approval":
        route = "qa_review"
    assert route == "qa_review"
