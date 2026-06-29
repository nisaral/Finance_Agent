import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from cachetools import TTLCache
from textblob import TextBlob

logger = logging.getLogger(__name__)

cache = TTLCache(maxsize=100, ttl=3600)
documents: list[str] = []
metadata: dict = {}

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "news.db"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS news
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            article TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.commit()
    conn.close()


def analyze_sentiment(text: str) -> str:
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "positive"
        if polarity < -0.1:
            return "negative"
        return "neutral"
    except Exception:
        text_lower = text.lower()
        if any(k in text_lower for k in ("good", "great", "rise", "profit")):
            return "positive"
        if any(k in text_lower for k in ("bad", "fall", "loss")):
            return "negative"
        return "neutral"


def save_news(company: str, articles_with_sentiment: list) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        for article in articles_with_sentiment:
            c.execute(
                "INSERT INTO news (company, article, sentiment) VALUES (?, ?, ?)",
                (company, article["article"], article["sentiment"]),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save news: {e}")