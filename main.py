import os
import sqlite3
import json
import base64
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from textblob import TextBlob
import aiohttp
from deepgram import DeepgramClient, SpeakOptions, PrerecordedOptions
from enum import Enum
from dotenv import load_dotenv
from cachetools import TTLCache
import re
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import google.generativeai as genai
import librosa
import tempfile
import soundfile as sf
from pydub import AudioSegment
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import requests
import zipfile
import os

# Function to download and extract GloVe embeddings
def download_glove_embeddings():
    glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    glove_zip_path = "glove.6B.zip"
    glove_file_path = "glove.6B.300d.txt"

    # Check if the file already exists
    if os.path.exists(glove_file_path):
        logger.info("GloVe embeddings file already exists, skipping download.")
        return

    logger.info("Downloading GloVe embeddings...")
    try:
        # Download the zip file
        response = requests.get(glove_url, stream=True)
        response.raise_for_status()
        with open(glove_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Extract the specific file we need
        logger.info("Extracting GloVe embeddings...")
        with zipfile.ZipFile(glove_zip_path, "r") as zip_ref:
            zip_ref.extract(glove_file_path)

        # Clean up the zip file
        os.remove(glove_zip_path)
        logger.info("GloVe embeddings downloaded and extracted successfully.")
    except Exception as e:
        logger.error(f"Failed to download GloVe embeddings: {e}")
        raise ValueError("Could not download GloVe embeddings. Please ensure internet access and try again.")

# Call the download function before loading embeddings
download_glove_embeddings()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Validate environment variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
if not NEWSAPI_KEY:
    raise ValueError("NEWSAPI_KEY not found in environment variables. Please set it in your .env file.")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="Finance Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:5500", "http://localhost:5500", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Deepgram API
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

# Ensure NLTK data for TextBlob is available
try:
    import nltk
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK punkt_tab: {e}. Using fallback sentiment analysis.")

# Load GloVe embeddings
GLOVE_PATH = "glove.6B.300d.txt"
dimension = 300
glove_embeddings = {}
try:
    with open(GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove_embeddings[word] = vector
    logger.info(f"Loaded GloVe embeddings with {len(glove_embeddings)} words")
except Exception as e:
    logger.error(f"Failed to load GloVe embeddings: {e}")
    raise ValueError("GloVe embeddings file not found. Please download glove.6B.300d.txt and place it in the project directory.")

# Function to compute sentence embedding using GloVe
def get_sentence_embedding(text: str) -> np.ndarray:
    try:
        blob = TextBlob(text.lower())
        words = blob.words
    except Exception as e:
        logger.warning(f"TextBlob tokenization failed: {e}, falling back to simple split")
        words = text.lower().split()

    vectors = []
    for word in words:
        vector = glove_embeddings.get(word)
        if vector is not None:
            vectors.append(vector)
    if not vectors:
        return np.zeros(dimension, dtype=np.float32)
    return np.mean(vectors, axis=0)

# Function to analyze sentiment using TextBlob with fallback
def analyze_sentiment(text: str) -> str:
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        logger.warning(f"TextBlob sentiment analysis failed: {e}. Using simple keyword-based sentiment.")
        text_lower = text.lower()
        positive_keywords = ["good", "great", "positive", "rise", "increase", "profit"]
        negative_keywords = ["bad", "poor", "negative", "fall", "decrease", "loss"]
        if any(keyword in text_lower for keyword in positive_keywords):
            return "positive"
        elif any(keyword in text_lower for keyword in negative_keywords):
            return "negative"
        return "neutral"

# Initialize cache
cache = TTLCache(maxsize=100, ttl=3600)

# Document storage for retrieval
documents = []
metadata = {}

# Initialize SQLite database
def init_db():
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'news.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  company TEXT NOT NULL,
                  article TEXT NOT NULL,
                  sentiment TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized successfully")

init_db()

# Pydantic models
class QueryRequest(BaseModel):
    query: Optional[str] = None
    portfolio: str = ""
    user_id: str = "default"

class CompanyRequest(BaseModel):
    company: str

class NewsHistoryRequest(BaseModel):
    company: str
    days: int = 7

class APIDataRequest(BaseModel):
    ticker: str

class State(BaseModel):
    query: str = ""
    portfolio: str = ""
    user_id: str = "default"
    audio: Optional[bytes] = None
    market_data: dict = {}
    news: list = []
    retrieved: list = []
    analysis_result: dict = {}
    narrative: str = ""
    narrative_for_audio: str = ""
    audio_output: str = ""
    audio_transcript: str = ""
    tone_analysis: dict = {}
    error: str = ""
    confidence: float = 0.5
    charts: list = []  # New field to store chart configurations

# Enums for market regions and sectors
class MarketRegion(str, Enum):
    GLOBAL = "global"
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    EMERGING_MARKETS = "emerging_markets"
    LATIN_AMERICA = "latin_america"

class MarketSector(str, Enum):
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    ENERGY = "energy"
    CONSUMER = "consumer"
    INDUSTRIALS = "industrials"
    REAL_ESTATE = "real_estate"
    UTILITIES = "utilities"
    MATERIALS = "materials"
    TELECOMMUNICATIONS = "telecommunications"

# Tone Analysis using librosa
async def analyze_tone(audio_bytes: bytes) -> Dict[str, str]:
    temp_audio_path = None
    try:
        # Save audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        # Convert WAV to a standard format using pydub
        audio_segment = AudioSegment.from_file(temp_audio_path)
        converted_path = temp_audio_path.replace(".wav", "_converted.wav")
        audio_segment.export(converted_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])  # Mono, 16kHz

        # Load audio with soundfile to validate
        y, sr = sf.read(converted_path)
        if len(y) < sr:  # Less than 1 second of audio
            logger.warning("Audio duration too short for tone analysis")
            return {"pitch": "medium", "energy": "high", "tempo": "fast"}  # Fallback tone

        # Load audio with librosa
        y, sr = librosa.load(converted_path, sr=None)
        
        # Analyze pitch, energy, and tempo
        pitch = np.mean(librosa.pitch_tuning(y))
        energy = np.mean(librosa.feature.rms(y=y))
        tempo = librosa.beat.tempo(y=y, sr=sr)[0] if len(y) > sr else 0.0  # Avoid tempo estimation on very short audio
        
        tone = {}
        tone['pitch'] = 'high' if pitch > 0.1 else 'low' if pitch < -0.1 else 'medium'
        tone['energy'] = 'high' if energy > 0.05 else 'low'
        tone['tempo'] = 'fast' if tempo > 120 else 'slow'
        
        logger.info(f"Tone analysis: {tone}")
        return tone
    
    except Exception as e:
        logger.error(f"Tone analysis error: {str(e)}")
        return {"pitch": "medium", "energy": "high", "tempo": "fast"}  # Fallback tone
    
    finally:
        # Clean up temporary files
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if 'converted_path' in locals() and os.path.exists(converted_path):
            os.remove(converted_path)

# API Agent: Fetch market data using yfinance
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_yfinance_data(symbols: List[str]) -> Dict[str, Dict]:
    result = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get("regularMarketPrice", info.get("previousClose", 0))
            volume = info.get("volume", 0)
            if price:
                result[symbol] = {
                    "price": float(price) if price is not None else 100.0,
                    "volume": int(volume) if volume is not None else 100000,
                    "asset_type": "stocks"
                }
            else:
                logger.warning(f"No price data from yfinance for {symbol}")
                result[symbol] = {
                    "price": 100.0,
                    "volume": 100000,
                    "asset_type": "stocks"
                }
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {str(e)}")
            result[symbol] = {
                "price": 100.0,
                "volume": 100000,
                "asset_type": "stocks"
            }
    return result

async def api_node(state: State) -> State:
    state.market_data = {"stocks": {}, "indices": {}, "currencies": {}, "commodities": {}}
    symbols = []
    for s in state.portfolio.split(","):
        s = s.strip()
        if not s:
            continue
        if ":" not in s:
            state.error = f"Invalid portfolio format: '{s}' must include quantity (e.g., 'AAPL:100')"
            logger.error(f"API error: {state.error}")
            return state
        parts = s.split(":")
        if len(parts) != 2:
            state.error = f"Invalid portfolio entry: '{s}' must be in the format 'symbol:quantity'"
            logger.error(f"API error: {state.error}")
            return state
        symbol, quantity = parts
        symbol = symbol.strip()
        quantity = quantity.strip()
        try:
            float(quantity)
        except ValueError:
            state.error = f"Invalid quantity in portfolio: '{quantity}' must be a number"
            logger.error(f"API error: {state.error}")
            return state
        if symbol:
            symbols.append(symbol)

    if not symbols:
        state.error = "No valid symbols provided in portfolio"
        logger.error("API error: No valid symbols provided")
        return state

    try:
        cache_key = ",".join(sorted(symbols))
        if cache_key in cache:
            logger.info(f"Returning cached data for {cache_key}")
            state.market_data = cache[cache_key]
            return state

        yfinance_data = await fetch_yfinance_data(symbols)
        for symbol in symbols:
            if symbol in yfinance_data:
                asset_type = yfinance_data[symbol].get("asset_type", "stocks")
                state.market_data[asset_type][symbol] = {
                    "price": float(yfinance_data[symbol]["price"]),
                    "volume": int(yfinance_data[symbol].get("volume", 0))
                }

        cache[cache_key] = state.market_data
    except Exception as e:
        state.error += f"API error: {str(e)}\n"
        logger.error(f"API error: {str(e)}")
        state.market_data = {"stocks": {s: {"price": 100.0, "volume": 100000} for s in symbols}}
    return state

# Scraping Agent: Fetch news
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_newsapi(symbol: str):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "apiKey": NEWSAPI_KEY,
        "pageSize": 5,
        "language": "en",
        "sortBy": "relevancy"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"NewsAPI error: {resp.status}")
                    return []
                data = await resp.json()
                if data.get("status") != "ok":
                    logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                    return []
                articles = [article["description"] for article in data.get("articles", []) if article.get("description")]
                if not articles:
                    articles = [f"No news available for {symbol}"]
                return articles
    except Exception as e:
        logger.error(f"Error fetching news from NewsAPI: {e}")
        return [f"No news available for {symbol}"]

def save_news(company: str, articles_with_sentiment: list):
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'news.db')
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        for article in articles_with_sentiment:
            text = article["article"]
            sentiment = article["sentiment"]
            c.execute("INSERT INTO news (company, article, sentiment) VALUES (?, ?, ?)",
                     (company, text, sentiment))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(articles_with_sentiment)} news articles for {company}")
    except Exception as e:
        logger.error(f"Failed to save news for {company}: {e}")

async def scraping_node(state: State) -> State:
    state.news = []
    symbols = [s.split(":")[0] for s in state.portfolio.split(",") if s and ":" in s]
    if not symbols:
        return state
    try:
        news = []
        for symbol in symbols:
            cache_key = f"news_{symbol}"
            if cache_key in cache:
                logger.info(f"Returning cached news for {symbol}")
                news.extend(cache[cache_key])
                continue

            news_articles = await fetch_newsapi(symbol)
            articles_with_sentiment = []
            for text in news_articles:
                sentiment = analyze_sentiment(text)
                articles_with_sentiment.append({
                    "article": text,
                    "sentiment": sentiment,
                    "timestamp": datetime.utcnow().isoformat()
                })

            save_news(symbol, articles_with_sentiment)
            cache[cache_key] = articles_with_sentiment
            news.extend(articles_with_sentiment)

        state.news = news
        global documents, metadata
        if news:
            for i, item in enumerate(news):
                doc_id = f"doc_{state.user_id}_{len(documents) + i}"
                metadata[doc_id] = {
                    "content": item["article"],
                    "user_id": state.user_id,
                    "type": "document",
                    "timestamp": datetime.utcnow().isoformat()
                }
                documents.append(item["article"])
    except Exception as e:
        state.error += f"Scraping service error: {str(e)}\n"
        logger.error(f"Scraping error: {str(e)}")
        state.news = []
    return state

# Retriever Agent: Simple keyword-based retrieval using GloVe embeddings
async def retriever_node(state: State) -> State:
    state.retrieved = []
    state.confidence = 0.0
    try:
        global documents, metadata

        if not documents:
            logger.warning("No documents available for retrieval")
            return state

        # Compute query embedding
        query_embedding = get_sentence_embedding(state.query).reshape(1, -1)

        # Store query in metadata
        query_id = f"query_{state.user_id}_{len(documents)}"
        timestamp = datetime.utcnow().isoformat()
        metadata[query_id] = {
            "query": state.query,
            "timestamp": timestamp,
            "user_id": state.user_id,
            "portfolio": state.portfolio,
            "type": "query"
        }
        documents.append(state.query)

        # Compute embeddings for documents
        doc_embeddings = np.array([get_sentence_embedding(doc) for doc in documents])

        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        # Ensure we don't retrieve more indices than available documents
        top_k = min(5, len(documents))
        retrieved_indices = np.argsort(similarities)[-top_k:][::-1]
        scores = similarities[retrieved_indices]

        retrieved_docs = []
        if scores.size > 0:  # Properly check if scores array has elements
            state.confidence = float(max(scores)) if max(scores) is not np.nan else 0.0
            for idx in retrieved_indices:
                # Ensure idx aligns with documents and metadata
                if idx < len(documents):
                    doc_id = f"doc_{state.user_id}_{idx}" if idx < len(state.news) else f"query_{state.user_id}_{idx}"
                    meta = metadata.get(doc_id, {})
                    if meta.get("user_id") == state.user_id and meta.get("type") == "document":
                        retrieved_docs.append(documents[idx])

        state.retrieved = retrieved_docs

    except Exception as e:
        state.error += f"Retriever service error: {str(e)}\n"
        logger.error(f"Retriever error: {str(e)}")
        state.retrieved = []
        state.confidence = 0.0
    return state

# Analysis Agent: Fetch metadata with retry
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_asset_metadata(symbols: List[str]) -> Dict[str, Dict]:
    metadata = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            asset_class = "Stock"
            if "quoteType" in info:
                if info["quoteType"] == "ETF":
                    asset_class = "ETF"
                elif info["quoteType"] == "CRYPTOCURRENCY":
                    asset_class = "Crypto"
                elif info["quoteType"] == "INDEX":
                    asset_class = "Index"
                elif info["quoteType"] == "COMMODITY":
                    asset_class = "Commodity"
            metadata[symbol] = {
                "asset_class": asset_class,
                "sector": info.get("sector", "Unknown"),
                "name": info.get("longName", symbol),
                "region": info.get("country", "Unknown")
            }
            # Reduce sleep time to minimize cancellation risk during shutdown
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.warning(f"yfinance metadata error for {symbol}: {str(e)}")
            metadata[symbol] = {"asset_class": "Unknown", "sector": "Unknown", "region": "Unknown"}
    return metadata

# Analysis Agent: Fetch earnings
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_earnings(symbol: str) -> float:
    try:
        ticker = yf.Ticker(symbol)
        earnings = ticker.earnings_dates
        if earnings is not None and not earnings.empty:
            latest = earnings.iloc[0]
            actual = float(latest.get("Reported EPS", 0))
            estimate = float(latest.get("EPS Estimate", 0))
            return ((actual - estimate) / estimate * 100) if estimate != 0 else 0.0
        return 0.0
    except Exception as e:
        logger.error(f"Earnings fetch error for {symbol}: {str(e)}")
        return 0.0

# Analysis Agent: Analyze portfolio
async def analysis_node(state: State) -> State:
    state.analysis_result = {
        "total_value": 0.0,
        "exposure": {},
        "regional_sector_exposure": {},
        "earnings_surprises": {},
        "asset_class_values": {},
        "regional_sector_values": {}
    }
    portfolio = {}
    for s in state.portfolio.split(","):
        s = s.strip()
        if not s or ":" not in s:
            continue
        try:
            symbol, quantity = s.split(":")
            portfolio[symbol.strip()] = float(quantity.strip())
        except ValueError as e:
            state.error += f"Invalid portfolio entry '{s}': {str(e)}\n"
            logger.error(f"Analysis error: {state.error}")
            return state

    if not portfolio:
        state.error += "No valid portfolio provided\n"
        logger.error("Analysis error: No valid portfolio provided")
        return state
    if not state.market_data:
        state.error += "Market data cannot be empty\n"
        logger.error("Analysis error: Market data cannot be empty")
        return state

    try:
        invalid_symbols = [s for s in portfolio if s not in state.market_data["stocks"]]
        if invalid_symbols:
            state.error += f"Missing market data for: {invalid_symbols}\n"
            logger.error(f"Analysis error: Missing market data for {invalid_symbols}")
            return state

        symbols = list(portfolio.keys())
        try:
            metadata = await fetch_asset_metadata(symbols)
        except asyncio.CancelledError:
            logger.warning("fetch_asset_metadata cancelled during shutdown")
            metadata = {symbol: {"asset_class": "Unknown", "sector": "Unknown", "region": "Unknown"} for symbol in symbols}

        total_value = 0.0
        asset_class_values = {}
        earnings_surprises = {}
        regional_sector_values = {}

        for symbol, shares in portfolio.items():
            market_info = state.market_data["stocks"].get(symbol, {})
            if not market_info:
                logger.warning(f"No price for {symbol}, skipping")
                continue

            price = float(market_info["price"])
            if np.isnan(price) or np.isinf(price):
                price = 100.0  # Fallback price to avoid NaN/Inf
            value = shares * price
            total_value += value

            asset_class = metadata.get(symbol, {}).get("asset_class", "Unknown")
            asset_class_values[asset_class] = asset_class_values.get(asset_class, 0.0) + value

            region = metadata.get(symbol, {}).get("region", "Unknown")
            sector = metadata.get(symbol, {}).get("sector", "Unknown")
            region_sector_key = f"{region}_{sector}"
            regional_sector_values[region_sector_key] = regional_sector_values.get(region_sector_key, 0.0) + value

            if metadata.get(symbol, {}).get("asset_class") == "Stock":
                surprise = await fetch_earnings(symbol)
                if np.isnan(surprise) or np.isinf(surprise):
                    surprise = 0.0
                earnings_surprises[symbol] = surprise

        if total_value == 0:
            state.error += "No valid assets with price data\n"
            logger.error("Analysis error: No valid assets with price data")
            return state

        # Calculate exposure, handle division by zero
        exposure = {}
        for asset_class, value in asset_class_values.items():
            percentage = (value / total_value) * 100 if total_value != 0 else 0.0
            if np.isnan(percentage) or np.isinf(percentage):
                percentage = 0.0
            exposure[asset_class] = float(percentage)

        regional_sector_exposure = {}
        for key, value in regional_sector_values.items():
            percentage = (value / total_value) * 100 if total_value != 0 else 0.0
            if np.isnan(percentage) or np.isinf(percentage):
                percentage = 0.0
            regional_sector_exposure[key] = float(percentage)

        state.analysis_result = {
            "total_value": float(total_value) if not (np.isnan(total_value) or np.isinf(total_value)) else 0.0,
            "exposure": exposure,
            "regional_sector_exposure": regional_sector_exposure,
            "earnings_surprises": earnings_surprises,
            "asset_class_values": {k: float(v) for k, v in asset_class_values.items()},
            "regional_sector_values": {k: float(v) for k, v in regional_sector_values.items()}
        }
    except Exception as e:
        state.error += f"Analysis error: {str(e)}\n"
        logger.error(f"Analysis error: {str(e)}")
    return state

# Language Agent: Fallback to Gemini 1.5 Flash if RAG fails
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_gemini_narrative(query: str, portfolio: str, tone_analysis: Dict[str, str]) -> str:
    tone_summary = f"User's voice tone: Pitch is {tone_analysis.get('pitch', 'unknown')}, energy is {tone_analysis.get('energy', 'unknown')}, tempo is {tone_analysis.get('tempo', 'unknown')}."
    try:
        prompt = f"""
You are a financial analyst. A user has asked: "{query}"
Their portfolio is: {portfolio}
{tone_summary}
Provide a concise market brief (150-200 words) analyzing their portfolio and answering their query.
Adjust your tone to be more empathetic if the user's tone sounds low-energy or slow, and more upbeat if high-energy or fast.
Include:
- Overview of the portfolio
- Key market trends or risks relevant to the query
- Recommendations or outlook
"""
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini narrative fallback failed: {str(e)}")
        return f"Unable to generate analysis for query: {query}. Please try again later."

# Helper to summarize news sentiment
def summarize_news_sentiment(news: List[Dict]) -> str:
    sentiments = [n.get('sentiment', 'neutral') for n in news]
    if not sentiments:
        return "Neutral"
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    return f"Positive: {pos}, Negative: {neg}, Neutral: {len(sentiments) - pos - neg}"

# Helper to determine query focus
def determine_query_focus(query: str) -> str:
    query_lower = query.lower()
    if "visualize" in query_lower:
        return "visualization"
    elif "investment opportunit" in query_lower:
        return "investment_opportunities"
    elif "risk" in query_lower or "exposure" in query_lower:
        return "risk_assessment"
    elif "update" in query_lower or "overview" in query_lower:
        return "portfolio_update"
    else:
        return "portfolio_update"  # Default to a general update

# Function to add stars to section headers for audio narration
def add_audio_section_markers(narrative: str) -> str:
    section_pattern = r"^(.*?):$"
    lines = narrative.split("\n")
    modified_lines = []
    for line in lines:
        if re.match(section_pattern, line.strip()):
            section_name = line.strip().rstrip(":")
            modified_line = f"* * {section_name} * *:"
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)
    return "\n".join(modified_lines)

# Helper to determine tone-based introduction
def get_tone_introduction(tone_analysis: Dict[str, str]) -> str:
    energy = tone_analysis.get("energy", "unknown")
    tempo = tone_analysis.get("tempo", "unknown")
    if energy == "low" or tempo == "slow":
        return "Noting your calm tone, I’ll provide a steady and detailed overview."
    elif energy == "high" or tempo == "fast":
        return "I sense an energetic tone, so I’ll deliver a brisk and upbeat summary!"
    else:
        return "I’ll provide a balanced overview tailored to your needs."

# Language Agent: Generate narrative and charts (if applicable)
async def language_node(state: State) -> State:
    state.narrative = ""
    state.narrative_for_audio = ""
    state.charts = []  # Reset charts
    if not state.query:
        state.error += "No query provided for narrative generation\n"
        return state

    if state.retrieved and state.confidence < 0.3:
        state.narrative = "Could you clarify your query? The retrieved information is insufficient."
        state.narrative_for_audio = add_audio_section_markers(state.narrative)
        return state

    if not state.retrieved and not state.news:
        state.narrative = await fetch_gemini_narrative(state.query, state.portfolio, state.tone_analysis)
        state.narrative_for_audio = add_audio_section_markers(state.narrative)
        return state

    # Determine the focus of the query
    query_focus = determine_query_focus(state.query)

    # Generate a chart if the query focus is visualization
    if query_focus == "visualization":
        sector_breakdown_dict = state.analysis_result.get("exposure", {})
        if sector_breakdown_dict:
            labels = list(sector_breakdown_dict.keys())
            data = list(sector_breakdown_dict.values())
            chart_config = {
                "type": "pie",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Sector Exposure",
                        "data": data,
                        "backgroundColor": [
                            "#FF6F61",  # Coral
                            "#6B5B95",  # Purple
                            "#88B04B",  # Green
                            "#F7CAC9",  # Light Pink
                            "#92A8D1",  # Light Blue
                            "#955251",  # Dark Coral
                            "#B565A7"   # Magenta
                        ],
                        "borderColor": "#FFFFFF",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "legend": {
                            "position": "top",
                            "labels": {
                                "color": "#333333"
                            }
                        },
                        "title": {
                            "display": True,
                            "text": "Sector Exposure Breakdown",
                            "color": "#333333"
                        }
                    }
                }
            }
            state.charts.append(chart_config)

    tone_summary = f"User's voice tone: Pitch is {state.tone_analysis.get('pitch', 'unknown')}, energy is {state.tone_analysis.get('energy', 'unknown')}, tempo is {state.tone_analysis.get('tempo', 'unknown')}."
    tone_intro = get_tone_introduction(state.tone_analysis)

    # Base prompt structure with dynamic sections based on query focus
    prompt = ChatPromptTemplate.from_template(
        """
Good day, sir/madam, as your financial assistant, I’m here with your market brief for {current_date}. {tone_intro}

{tone_summary}
Adjust your tone to be more empathetic if the user's tone sounds low-energy or slow, and more upbeat if high-energy or fast.

Portfolio Summary:
- Total Value: ${total_value:,.2f}
- Key Holdings: {key_holdings}
- Sector Allocation: {sector_breakdown}
- Regional Allocation: {regional_breakdown}

Market Insights:
- Sentiment: {news_sentiment}
- Recent News: {news_summary}
- Earnings: {earnings_analysis}
- Indices: {market_indices}
- FX/Commodities: {fx_commodities}

Analysis for Your Query: "{query}"

""" + (
        # Visualization Focus
        """
Portfolio Snapshot for Visualization:
- Total Value: ${total_value:,.2f}, with {key_holdings} as your main holdings.
- Sector Breakdown: {sector_breakdown}.
- Regional Exposure: {regional_breakdown}.
- Key Insight: Your portfolio is {risk_level} risk, primarily in {focus_sector}. A chart of sector allocations has been generated for you to visualize concentration risks.
"""
        if query_focus == "visualization" else
        # Risk Assessment
        """
Risk Assessment:
- Risk Level: {risk_level}, with a total value of ${total_value:,.2f}.
- Exposure: {exposure_summary}, concentrated in {focus_sector}, {focus_region}.
- Vulnerabilities: {risk_vulnerabilities}.
- Sector Risks: {focus_sector} may face {sector_risks}.
- Regional Risks: {focus_region} could see {regional_risks}.
- Mitigation: {risk_mitigation}. Diversify by {diversification_suggestion} (e.g., VPU for Utilities).
"""
        if query_focus == "risk_assessment" else
        # Investment Opportunities
        """
Investment Opportunities:
- Current Exposure: {exposure_summary}, focused in {focus_sector}, {focus_region}.
- Opportunities: {suggested_sector} (e.g., {suggested_symbols}) looks promising due to {news_summary_insight}.
- Alternative Regions: Consider {alternative_region} for growth.
- Recommendation: Allocate to {investment_suggestion}. Monitor {events_to_monitor}. Confidence: {confidence:.1%}.
"""
        if query_focus == "investment_opportunities" else
        # Portfolio Update (Default)
        """
Portfolio Overview:
- Value: ${total_value:,.2f}, with holdings in {key_holdings}.
- Performance Drivers: {focus_sector} trends are key. {news_summary_insight} Earnings: {earnings_analysis_insight}.
- Market Context: {market_indices_insight}. {fx_commodities_insight}.
- Risk Profile: {risk_level}, with risks from {risk_vulnerabilities}.
- Outlook: {market_outlook} outlook. Recommendation: {recommendation}. Monitor {events_to_monitor}.
"""
    ) + """

GUIDELINES:
- Professional tone, adjusted based on user's voice tone.
- Quantitative with figures.
- Actionable insights with specific recommendations (e.g., suggest stocks/ETFs like VPU, JNJ).
- Risk-aware.
- 150-200 words.
- Confidence levels for assessments.
- Use plain section headers (e.g., "Portfolio Overview:") for written output.
- Avoid using special characters like $ or % in the narrative text for clean audio output.
- If market data is unavailable, use general sector trends or historical insights as a fallback.

SESSION CONTEXT:
- Date: {current_date}
- Trading Environment: {trading_session}
"""
    )

    try:
        current_date = datetime.utcnow().strftime("%B %d, %Y")
        focus_region = MarketRegion.ASIA_PACIFIC if "asia" in state.query.lower() else MarketRegion.GLOBAL
        focus_sector = MarketSector.TECHNOLOGY if "tech" in state.query.lower() else MarketSector.FINANCIALS

        total_value = state.analysis_result.get("total_value", 0.0)
        surprises = state.analysis_result.get("earnings_surprises", {})
        documents = state.retrieved or [item["article"] for item in state.news]
        key_holdings = [s.split(":")[0] for s in state.portfolio.split(",") if s and ":" in s]
        sector_breakdown_dict = state.analysis_result.get("exposure", {})
        regional_sector_exposure = state.analysis_result.get("regional_sector_exposure", {})
        major_indices = {k: v["price"] for k, v in state.market_data.get("indices", {}).items()}
        currency_moves = {k: v["price"] for k, v in state.market_data.get("currencies", {}).items()}
        commodity_prices = {k: v["price"] for k, v in state.market_data.get("commodities", {}).items()}

        # Determine risk level based on concentration
        max_exposure = max(sector_breakdown_dict.values()) if sector_breakdown_dict else 0.0
        risk_level = "High" if max_exposure > 70 else "Moderate" if max_exposure > 40 else "Low"

        exposure_summary = "; ".join([f"{asset}: {weight:.1f} percent" for asset, weight in sector_breakdown_dict.items()]) if sector_breakdown_dict else "N/A"
        earnings_analysis = "; ".join([f"{company}: {surprise:+.1f} percent" for company, surprise in surprises.items()]) if surprises else "No major earnings surprises"
        market_indices = ", ".join([f"{index}: {price:.2f}" for index, price in major_indices.items()]) if major_indices else "Technology sector historically stable"
        fx_commodities = "; ".join([f"{k}: {v:.2f}" for k, v in list(currency_moves.items()) + list(commodity_prices.items())]) if (currency_moves or commodity_prices) else "No significant movements"
        news_summary = " | ".join(documents[:3]) if documents else "No recent news"
        news_sentiment = summarize_news_sentiment(state.news)
        key_holdings_str = ", ".join(key_holdings[:3]) if key_holdings else "N/A"
        sector_breakdown = "; ".join([f"{sector}: {weight:.1f} percent" for sector, weight in sector_breakdown_dict.items()]) if sector_breakdown_dict else "N/A"
        regional_breakdown = "; ".join([f"{rs}: {weight:.1f} percent" for rs, weight in regional_sector_exposure.items()]) if regional_sector_exposure else "N/A"

        # Additional fields for dynamic sections
        news_summary_insight = f"Recent news shows {news_sentiment.lower()}, indicating {'positive momentum' if 'Positive' in news_sentiment else 'potential challenges'}." if news_summary != "No recent news" else "No recent news impacts noted."
        earnings_analysis_insight = f"Earnings show {earnings_analysis.lower()}." if earnings_analysis != "No major earnings surprises" else "No significant earnings impacts."
        market_indices_insight = f"Major indices: {market_indices}" if market_indices != "Technology sector historically stable" else "Technology sector has been stable historically"
        fx_commodities_insight = f"FX/Commodities: {fx_commodities}" if fx_commodities != "No significant movements" else "No notable currency or commodity movements"
        market_outlook = "favorable" if "Positive" in news_sentiment else "cautious"
        risk_vulnerabilities = f"sector-specific downturns in {focus_sector.value.replace('_', ' ').title()}"
        recommendation = "diversify into Utilities (e.g., VPU)" if risk_level == "High" else "maintain allocations, monitor closely"
        events_to_monitor = f"{focus_sector.value.replace('_', ' ').title()} developments and global macro trends"
        sector_risks = "market volatility and regulatory changes"
        regional_risks = "geopolitical tensions or economic slowdown"
        risk_mitigation = "reduce concentration by diversifying into other sectors"
        diversification_suggestion = f"adding exposure to {MarketSector.UTILITIES.value.replace('_', ' ').title()}"
        suggested_sector = MarketSector.HEALTHCARE.value.replace('_', ' ').title()
        suggested_symbols = "JNJ, PFE"
        alternative_region = MarketRegion.EMERGING_MARKETS.value.replace('_', ' ').title()
        investment_suggestion = f"{suggested_sector} stocks like {suggested_symbols}"

        hour = datetime.utcnow().hour
        if 6 <= hour < 12:
            trading_session = "Pre-market / Asian Session"
        elif 12 <= hour < 18:
            trading_session = "European / US Pre-market"
        elif 18 <= hour < 24:
            trading_session = "US Market Hours"
        else:
            trading_session = "After Hours / Asian Pre-market"

        formatted_prompt = prompt.format(
            tone_summary=tone_summary,
            tone_intro=tone_intro,
            current_date=current_date,
            query=state.query,
            total_value=total_value,
            key_holdings=key_holdings_str,
            sector_breakdown=sector_breakdown,
            regional_breakdown=regional_breakdown,
            exposure_summary=exposure_summary,
            earnings_analysis=earnings_analysis,
            market_indices=market_indices,
            fx_commodities=fx_commodities,
            news_summary=news_summary,
            news_sentiment=news_sentiment,
            focus_region=focus_region.value.replace('_', ' ').title(),
            focus_sector=focus_sector.value.replace('_', ' ').title(),
            risk_level=risk_level,
            news_summary_insight=news_summary_insight,
            earnings_analysis_insight=earnings_analysis_insight,
            market_indices_insight=market_indices_insight,
            fx_commodities_insight=fx_commodities_insight,
            market_outlook=market_outlook,
            risk_vulnerabilities=risk_vulnerabilities,
            recommendation=recommendation,
            events_to_monitor=events_to_monitor,
            sector_risks=sector_risks,
            regional_risks=regional_risks,
            risk_mitigation=risk_mitigation,
            diversification_suggestion=diversification_suggestion,
            suggested_sector=suggested_sector,
            suggested_symbols=suggested_symbols,
            alternative_region=alternative_region,
            investment_suggestion=investment_suggestion,
            confidence=state.confidence,
            trading_session=trading_session
        )

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(formatted_prompt)
        state.narrative = response.text.strip()
        state.narrative_for_audio = add_audio_section_markers(state.narrative)

    except Exception as e:
        state.error += f"Language service error: {str(e)}\n"
        logger.error(f"Language error: {str(e)}")
        fallback_narrative = await fetch_gemini_narrative(state.query, state.portfolio, state.tone_analysis)
        state.narrative = fallback_narrative
        state.narrative_for_audio = add_audio_section_markers(fallback_narrative)
    return state

# Voice Agent: Process audio input (STT) and Tone Analysis
async def process_audio_input(state: State) -> State:
    if not state.audio:
        logger.warning("No audio provided for processing")
        return state

    try:
        # Log audio size
        audio_size = len(state.audio)
        logger.info(f"Audio input size: {audio_size} bytes")

        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(state.audio)
            temp_audio_path = temp_audio.name

        # Convert WAV to a standard format using pydub
        audio_segment = AudioSegment.from_file(temp_audio_path)
        converted_path = temp_audio_path.replace(".wav", "_converted.wav")
        audio_segment.export(converted_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])  # Mono, 16kHz

        try:
            # Check audio duration using soundfile
            with sf.SoundFile(converted_path) as f:
                duration = len(f) / f.samplerate
                logger.info(f"Audio duration: {duration:.2f} seconds")
                if duration < 1.0:  # Minimum 1 second of audio
                    logger.warning("Audio duration too short for processing")
                    state.query = "Audio too short to transcribe."
                    state.tone_analysis = {"pitch": "medium", "energy": "high", "tempo": "fast"}  # Fallback tone
                    return state

            # Read the converted audio file for Deepgram
            with open(converted_path, "rb") as f:
                audio_data = f.read()

            # Perform tone analysis
            state.tone_analysis = await analyze_tone(audio_data)

            # Perform STT with Deepgram
            buffer_data = {"buffer": audio_data}
            options = PrerecordedOptions(
                model="nova-2",
                language="en",
                smart_format=True,
                detect_language=True,
                punctuate=True,
                utterances=True
            )

            response = deepgram.listen.prerecorded.v("1").transcribe_file(
                buffer_data,
                options
            )

            # Access the response attributes directly
            if hasattr(response, "results") and hasattr(response.results, "channels"):
                transcript = response.results.channels[0].alternatives[0].transcript.strip()
                state.query = transcript if transcript else "No speech detected in audio."
                logger.info(f"Transcribed audio to query with Deepgram: '{state.query}'")
            else:
                raise Exception("Deepgram response missing expected fields")

        finally:
            # Clean up temporary files
            os.remove(temp_audio_path)
            os.remove(converted_path)

    except Exception as e:
        state.error += f"Audio processing error: {str(e)}\n"
        logger.error(f"Audio processing error: {str(e)}")
        state.query = "Unable to transcribe audio input."
        state.tone_analysis = {"pitch": "medium", "energy": "high", "tempo": "fast"}  # Fallback tone
    return state

# Voice Agent: Generate audio (TTS)
async def generate_audio(state: State) -> State:
    state.audio_output = ""
    state.audio_transcript = state.narrative  # Use the clean narrative for the transcript
    if not state.narrative_for_audio:
        state.error += "No narrative available for audio generation\n"
        return state
    try:
        narrative = state.narrative_for_audio[:2000]
        options = SpeakOptions(
            model="aura-asteria-en",
            encoding="mp3",
        )

        response = deepgram.speak.rest.v("1").stream(
            {"text": narrative},
            options
        )

        audio_content = response.stream.getvalue()
        audio_data = base64.b64encode(audio_content).decode()
        state.audio_output = f"data:audio/mp3;base64,{audio_data}"
        logger.info("Successfully generated audio output with Deepgram")

    except Exception as e:
        state.error += f"Voice generation error: {str(e)}\n"
        logger.error(f"Voice generation error: {str(e)}")
        state.audio_output = ""
    return state

# Main pipeline
async def run_pipeline(query: Optional[str], portfolio: str, user_id: str, audio: Optional[bytes], background_tasks: BackgroundTasks):
    state = State(
        query=query or "",
        portfolio=portfolio,
        user_id=user_id,
        audio=audio
    )

    # Step 1: Process audio input if provided
    if state.audio:
        state = await process_audio_input(state)

    # Validate query, but proceed with a default if empty
    if not state.query:
        state.query = "Provide an update on my portfolio."
        logger.warning(f"No query provided after audio processing, using default query: '{state.query}'")

    # Step 2: Fetch market data
    state = await api_node(state)
    if state.error and "No valid symbols" in state.error:
        return {
            "error": state.error,
            "narrative": "Invalid portfolio format.",
            "audio_file": "",
            "audio_transcript": "",
            "tone_analysis": state.tone_analysis,
            "analysis": {},
            "query": state.query,
            "news": [],
            "charts": []
        }

    # Step 3: Fetch news
    state = await scraping_node(state)

    # Step 4: Retrieve documents
    state = await retriever_node(state)

    # Step 5: Analyze portfolio
    state = await analysis_node(state)

    # Step 6: Generate narrative and charts
    state = await language_node(state)

    # Step 7: Generate audio
    state = await generate_audio(state)

    return {
        "narrative": state.narrative,  # Clean narrative without ** markers
        "audio_file": state.audio_output,
        "audio_transcript": state.audio_transcript,
        "tone_analysis": state.tone_analysis,
        "analysis": state.analysis_result,
        "query": state.query,
        "error": state.error,
        "news": state.news,
        "charts": state.charts  # Include any generated charts
    }

# API Agent endpoint (Live tracking)
@app.post("/api/run")
async def api_run(request: APIDataRequest):
    state = State(portfolio=request.ticker)
    state = await api_node(state)
    if "No valid symbols provided in portfolio" in state.error:
        raise HTTPException(status_code=400, detail=state.error)
    return state.market_data

# Scraping Agent endpoints
@app.post("/scraping/run")
async def scraping_run(request: CompanyRequest):
    try:
        cache_key = f"news_{request.company}"
        if cache_key in cache:
            logger.info(f"Returning cached news for {request.company}")
            return {"news": cache[cache_key]}

        news_articles = await fetch_newsapi(request.company)
        articles_with_sentiment = []
        for text in news_articles:
            sentiment = analyze_sentiment(text)
            articles_with_sentiment.append({
                "article": text,
                "sentiment": sentiment,
                "timestamp": datetime.utcnow().isoformat()
            })

        save_news(request.company, articles_with_sentiment)
        cache[cache_key] = articles_with_sentiment
        return {"news": articles_with_sentiment}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"news": [f"Error fetching news for {request.company}"]}

@app.post("/scraping/history")
async def get_news_history(request: NewsHistoryRequest):
    try:
        db_path = os.path.join(os.path.dirname(__file__), 'data', 'news.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute("""SELECT article, sentiment, timestamp 
                    FROM news 
                    WHERE company = ? 
                    AND timestamp >= datetime('now', ?) 
                    ORDER BY timestamp DESC""",
                 (request.company, f'-{request.days} days'))
        
        results = c.fetchall()
        conn.close()
        
        history = [
            {
                "article": row[0],
                "sentiment": row[1],
                "timestamp": row[2]
            }
            for row in results
        ]
        
        return {"history": history}
    except Exception as e:
        logger.error(f"Error fetching news history: {e}")
        return {"history": []}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Main endpoint for frontend
@app.post("/run")
async def run(
    query: Optional[str] = Form(None),
    portfolio: str = Form(...),
    user_id: str = Form("default"),
    audio: Optional[UploadFile] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        audio_bytes = None
        if audio:
            audio_bytes = await audio.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Empty audio file provided")

        result = await run_pipeline(query, portfolio, user_id, audio_bytes, background_tasks)
        return JSONResponse(
            content=result,
            headers={
                "Access-Control-Allow-Origin": "http://127.0.0.1:5500",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Server error: {str(e)}",
                "narrative": "An error occurred while processing your request.",
                "audio_file": "",
                "audio_transcript": "",
                "tone_analysis": {"pitch": "medium", "energy": "medium", "tempo": "low"},
                "analysis": {},
                "query": query or "",
                "news": [],
                "charts": []
            },
            headers={
                "Access-Control-Allow-Origin": "http://127.0.0.1:5500",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
app.mount("/static", StaticFiles(directory="."), name="static")

# Serve index.html at the root URL
@app.get("/")
async def serve_index():
    return FileResponse("index.html")
