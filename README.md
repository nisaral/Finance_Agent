#  Voice-First Financial Assistant

A voice-powered agent that provides real-time analysis, news, and insights for your stock portfolio. This project was built as a take-home assignment for TimeAI.


---
##  Features

* **Voice & Text Input:** Ask questions using natural language through your microphone or by typing.
* **Live Portfolio Tracking:** Input your holdings (e.g., `AAPL:10, MSFT:20`) to see live market values.
* **AI-Powered Analysis:** Leverages Google's Gemini model to generate concise narratives about portfolio performance, risk, and opportunities.
* **Real-Time News:** Fetches and analyzes the sentiment of the latest news relevant to your holdings.
* **Audio Responses:** Get your market brief read back to you with realistic text-to-speech.
* **Dynamic Visualizations:** Ask the agent to "visualize" your portfolio to generate charts on the fly.

---
##  Architecture

![![WhatsApp Image 2025-08-17 at 14 21 52_c05bebad](https://github.com/user-attachments/assets/ece77a2a-21a6-47f8-994c-f660954c269d)
]

The application is built around an agentic, asynchronous FastAPI backend. The user interacts with a simple HTML/JS frontend, which communicates with the backend via REST APIs. When a query is received, the backend orchestrates a series of agents to fetch data, perform analysis, and generate a multi-modal (text, audio, chart) response. A lightweight `sentence-transformers` model is used for efficient semantic retrieval of news articles.

---
##  Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **AI/ML:**
    * **LLM:** Google Gemini 2.5 Flash
    * **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
    * **Audio Processing:** Librosa, PyDub
* **APIs & Services:**
    * **Speech-to-Text/Text-to-Speech:** Deepgram Aura
    * **Market Data:** yfinance
    * **News:** NewsAPI
* **Frontend:** Vanilla HTML, CSS, JavaScript
* **Database:** SQLite (for news article caching)

---
##  Running Locally

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the root directory and add your API keys:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    NEWSAPI_KEY="YOUR_NEWSAPI_KEY"
    DEEPGRAM_API_KEY="YOUR_DEEPGRAM_API_KEY"
    ```

5.  **Run the application:**
    ```bash
    uvicorn main:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

---
##  Future Improvements

-   [ ] **User Authentication:** To securely store and manage multiple user portfolios.
-   [ ] **Persistent Database:** Move from a temporary SQLite DB to a persistent solution like PostgreSQL.
-   [ ] **Proactive Alerts:** Implement a background worker to monitor for significant news or price changes and alert the user.
