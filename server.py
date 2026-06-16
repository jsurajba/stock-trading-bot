import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables before importing from trading_bot
load_dotenv()

# Import quantitative engines directly
from trading_bot import (
    get_technical_indicators,
    get_latest_market_prices,
    news_client,
    NewsRequest
)

app = FastAPI(title="Antigravity Quant Core Engine API")

# Enable CORS so the browser can communicate with localhost:8000
# NOTE: allow_credentials is omitted since allow_origins=["*"] is set to prevent a startup RuntimeError.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The core watchlist exactly matching the UI layout tiers
WATCHLIST = ["VOO", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "PLTR", "COIN", "SMCI"]

@app.get("/api/telemetry")
async def get_dashboard_telemetry():
    """
    Consolidates advanced live indicators and market news wire 
    into a single unified payload for the frontend terminal.
    """
    try:
        # 1. Fetch data directly from trading_bot pipelines
        live_prices = get_latest_market_prices(WATCHLIST)
        technicals = get_technical_indicators(WATCHLIST)
        
        # 2. Structure asset metrics into the array nodes the UI expects
        compiled_assets = []
        for ticker in WATCHLIST:
            price = live_prices.get(ticker, 0.0)
            tech = technicals.get(ticker, {})
            
            compiled_assets.append({
                "ticker": ticker,
                "price": price,
                "rsi": tech.get("rsi", 50.0),
                "sma50Distance": tech.get("dist_sma", 0.0), # Maps to UI display
                "volRatio": tech.get("vol_ratio", 1.0),
                "realizedVol": tech.get("realized_vol_20d", 30.0)
            })
            
        # 3. Pull recent global news wire headlines via Alpaca
        compiled_news = []
        try:
            req = NewsRequest(limit=10)
            news_response = news_client.get_news(req)
            news_list = news_response.get("news", [])
            
            for index, article in enumerate(news_list):
                # Identify if this article mentions any of our watched tickers
                matched_ticker = "VOO" # default fallback
                for sym in article.get("symbols", []):
                    if sym in WATCHLIST:
                        matched_ticker = sym
                        break
                        
                compiled_news.append({
                    "id": article.get("id", index),
                    "ticker": matched_ticker, # Fixed NameError from camelCase matchedTicker
                    "headline": article.get("headline", "Market Event Flagged"),
                    "summary": article.get("summary", ""),
                    "created_at": article.get("created_at", datetime.utcnow().isoformat())
                })
        except Exception as news_err:
            print(f"News aggregation bypassed: {news_err}")
            
        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "assets": compiled_assets,
            "news_feed": compiled_news
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Launch the local application engine on port 8000
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
