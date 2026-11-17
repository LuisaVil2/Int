"""Explicit opt-in generator. Its only inputs are controlled scenario settings."""
from __future__ import annotations

import json

from ..translate import DeepSeekTranslator
from .schemas import Scenario


def generate_scenario(specialty, difficulty, *, allow_network=False) -> Scenario:
    if not allow_network:
        raise ValueError("External generation requires --allow-network")
    translator = DeepSeekTranslator()
    schema = json.dumps(Scenario.model_json_schema(), ensure_ascii=True)
    response = translator.client.messages.create(
        model=translator.model, temperature=0.6, max_tokens=5000,
        extra_body={"thinking": {"type": "disabled"}},
        system=("Generate fictional bilingual scripts for interpreter QA. Return one JSON object. "
                "Invent every detail and omit names, contact information and other personal identifiers. "
                "Alternate an English clinician and Spanish patient with 4 to 8 utterances. "
                "Write faithful bilingual reference translations. Preserve first person, details and register. "
                "Include clarifications and corrections. Never include identifiable real people. "
                "Use asr_language as the turn language except where ambiguity is deliberately tested. "
                "For intentionally uninterpretable turns use review_expected=true and null reference. "
                "Leave mock_reply null. Required JSON schema: " + schema),
        messages=[{"role": "user", "content": json.dumps({"specialty": specialty, "difficulty": difficulty})}],
    )
    if response.stop_reason != "end_turn":
        raise ValueError("The generator did not return a complete scenario")
    candidate = Scenario.model_validate_json("".join(
        block.text for block in response.content if block.type == "text"))
    if candidate.specialty != specialty or candidate.difficulty != difficulty:
        raise ValueError("The generated scenario does not match the requested settings")
    if any(turn.mock_reply is not None for turn in candidate.turns):
        raise ValueError("Generated scenario drafts must leave mock replies empty")
    return candidate
