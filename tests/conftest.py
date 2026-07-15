from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def terminology_index():
    from src.terminology import TerminologyIndex

    return TerminologyIndex.load(ROOT / "data" / "terminology")
