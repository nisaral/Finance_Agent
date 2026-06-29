import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)


@pytest.fixture
def has_gemini():
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_FALLBACK"))


@pytest.fixture
def has_cartesia():
    return bool(os.getenv("CARTESIA_API_KEY"))


@pytest.fixture
def has_newsapi():
    return bool(os.getenv("NEWSAPI_KEY"))


@pytest.fixture
def all_live_keys(has_gemini, has_cartesia, has_newsapi):
    return has_gemini and has_cartesia and has_newsapi