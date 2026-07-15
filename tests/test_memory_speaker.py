from src.memory import ConversationMemory


def test_context_empty():
    m = ConversationMemory()
    assert "sin contexto previo" in m.context()


def test_add_turn_without_speaker_backward_compatible():
    m = ConversationMemory()
    m.add_turn(None, "en", "hello", "hola")
    ctx = m.context()
    assert "(S" not in ctx
    assert "hello -> hola" in ctx


def test_add_turn_with_speaker_id_shown_in_context():
    m = ConversationMemory()
    m.add_turn(0, "en", "hello doctor", "hola doctor")
    m.add_turn(1, "es", "hola", "hello")
    ctx = m.context()
    assert "(S0)" in ctx
    assert "(S1)" in ctx


def test_context_limit_trims_to_last_n_turns():
    m = ConversationMemory()
    for i in range(10):
        m.add_turn(None, "en", f"turn {i}", f"turno {i}")
    limited = m.context(limit=3)
    assert "turn 9" in limited
    assert "turn 0" not in limited
    full = m.context()
    assert "turn 0" in full
