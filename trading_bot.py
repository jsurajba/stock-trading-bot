import os
import json
import time
import sys
import math
from typing import List
from datetime import datetime, timedelta
from collections import Counter
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockLatestTradeRequest, NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
import anthropic
import openai
import pytz

# ==============================================================================
# 1. SETTINGS, CREDENTIALS & INITIALIZATION
# ==============================================================================
# Load environment variables from .env file
load_dotenv()

# LLM Routing Setup
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.getenv("LLM_MODEL")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Validate environment variables dynamically depending on active LLM provider
if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        raise ValueError("CRITICAL CONFIGURATION ERROR: GEMINI_API_KEY is not set but LLM_PROVIDER is configured as 'gemini'.")
elif LLM_PROVIDER == "anthropic":
    if not ANTHROPIC_API_KEY:
        raise ValueError("CRITICAL CONFIGURATION ERROR: ANTHROPIC_API_KEY is not set but LLM_PROVIDER is configured as 'anthropic'.")
elif LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise ValueError("CRITICAL CONFIGURATION ERROR: OPENAI_API_KEY is not set but LLM_PROVIDER is configured as 'openai'.")
else:
    raise ValueError(f"CRITICAL CONFIGURATION ERROR: Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Must be 'gemini', 'anthropic', or 'openai'.")

if not ALPACA_KEY_ID or not ALPACA_SECRET:
    raise ValueError("CRITICAL CONFIGURATION ERROR: ALPACA_KEY_ID and ALPACA_SECRET must be configured in the environment.")

# Initialize AI clients conditionally to prevent errors for missing unused keys
ai_client = None
if LLM_PROVIDER == "gemini":
    ai_client = genai.Client(api_key=GEMINI_API_KEY)

anthropic_client = None
if LLM_PROVIDER == "anthropic":
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

openai_client = None
if LLM_PROVIDER == "openai":
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Always initialize brokerage clients
trading_client = TradingClient(ALPACA_KEY_ID, ALPACA_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALPACA_KEY_ID, ALPACA_SECRET, raw_data=True)
news_client = NewsClient(api_key=ALPACA_KEY_ID, secret_key=ALPACA_SECRET, raw_data=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOURNAL_FILE = os.path.join(SCRIPT_DIR, "trade_journal.json")

# ==============================================================================
# 2. STRUCTURED DATA SCHEMAS
# ==============================================================================
class TickerDecision(BaseModel):
    ticker: str = Field(description="The stock ticker symbol being evaluated.")
    chain_of_thought_analysis: str = Field(description="Detailed internal step-by-step reasoning. Must analyze: 1) RSI overextension, 2) Distance to 50-day SMA, 3) Volume confirmation relative to breaking news, and 4) Volatility / ADX trend strength.")
    analysis: str = Field(description="Final asset thesis incorporating momentum, vol ratio, news, RSI. Max 2 sentences.")
    lessons_applied: str = Field(description="Brief note on how historical journal entries altered today's choice.")
    signal: str = Field(description="Must be exactly 'BUY', 'SELL', or 'HOLD'.")
    confidence_score: int = Field(description="Confidence level from 0 to 100.")
    risk_allocation_percentage: float = Field(description="Target allocation of TOTAL portfolio equity (0.0 to 15.0). Core low-volatility indices (VOO, QQQ, SCHD) can scale up to 10.0 to 15.0% on high confidence. Volatile dynamic stocks must be capped at 2.5% to 5.0%. 0.0 if HOLD/SELL.")

class PortfolioManagerDecision(BaseModel):
    chain_of_thought_reasoning: str = Field(description="Detailed step-by-step macro analysis. Must analyze: 1) Regime Identification (VOO/QQQ distance to SMA, trend strength, global indicators), 2) Asymmetric Risk-Reward conditions, 3) Volume confirmation of catalysts.")
    macro_strategy_outlook: str = Field(description="Executive overview of macro conditions based on the live news feed and regime status.")
    asset_decisions: List[TickerDecision] = Field(description="Array of precise, strategic allocations for every single asset in the dynamic watchlist.")

# ==============================================================================
# 3. UTILITIES & TELEMETRY OUTFLOWS
# ==============================================================================
def send_discord_message(content: str):
    if not DISCORD_WEBHOOK_URL or "YOUR_DISCORD_WEBHOOK" in DISCORD_WEBHOOK_URL:
        return
    try:
        payload = {"content": content[:1950]}
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"!!! Failed to stream telemetry to Discord: {e}")

def load_trade_journal() -> list:
    if not os.path.exists(JOURNAL_FILE): return []
    try:
        with open(JOURNAL_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def log_portfolio_trades(executed_trades: list):
    if not executed_trades: return
    journal = load_trade_journal()
    journal.extend(executed_trades)
    with open(JOURNAL_FILE, "w") as f:
        json.dump(journal[-100:], f, indent=2) 

def build_cooldown_blacklist() -> set:
    journal = load_trade_journal()
    blacklist = set()
    cutoff_date = datetime.now() - timedelta(days=COOLDOWN_DAYS)
    
    for entry in journal:
        if entry.get("action") == "SELL" and "FORCED" in entry.get("thesis", ""):
            entry_time_str = entry.get("timestamp")
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(entry_time_str)
                    if entry_time >= cutoff_date:
                        blacklist.add(entry.get("ticker"))
                except ValueError:
                    pass
    return blacklist

def get_asset_sector(ticker: str) -> str:
    """Returns the macro sector. Assigns a unique isolate ID if the stock is an unmapped dynamic discovery."""
    return SECTOR_MAP.get(ticker, f"Dynamic_Isolate_{ticker}")

# ==============================================================================
# 4. QUANTITATIVE MATHEMATICS & DATA PIPELINES
# ==============================================================================
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1: return 50.0
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ema(prices: List[float], period: int) -> List[float]:
    if not prices: return []
    if len(prices) < period:
        return [sum(prices)/len(prices)] * len(prices)
    ema = []
    multiplier = 2 / (period + 1)
    sma = sum(prices[:period]) / period
    for i in range(len(prices)):
        if i < period - 1:
            ema.append(prices[i])
        elif i == period - 1:
            ema.append(sma)
        else:
            val = prices[i] * multiplier + ema[-1] * (1 - multiplier)
            ema.append(val)
    return ema

def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    if len(closes) < period * 2:
        return 25.0  # Default neutral/choppy fallback
    
    tr = []
    plus_dm = []
    minus_dm = []
    
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        prev_h = highs[i-1]
        prev_l = lows[i-1]
        prev_c = closes[i-1]
        
        tr_val = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr.append(tr_val)
        
        up_move = h - prev_h
        down_move = prev_l - l
        
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)
            
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)
            
    # Wilder's smoothing
    atr = [sum(tr[:period])]
    smoothed_plus_dm = [sum(plus_dm[:period])]
    smoothed_minus_dm = [sum(minus_dm[:period])]
    
    for i in range(period, len(tr)):
        atr.append(atr[-1] - (atr[-1] / period) + tr[i])
        smoothed_plus_dm.append(smoothed_plus_dm[-1] - (smoothed_plus_dm[-1] / period) + plus_dm[i])
        smoothed_minus_dm.append(smoothed_minus_dm[-1] - (smoothed_minus_dm[-1] / period) + minus_dm[i])
        
    dx = []
    for i in range(len(atr)):
        if atr[i] == 0:
            dx.append(0.0)
            continue
        plus_di = (smoothed_plus_dm[i] / atr[i]) * 100
        minus_di = (smoothed_minus_dm[i] / atr[i]) * 100
        diff = abs(plus_di - minus_di)
        denom = plus_di + minus_di
        dx_val = (diff / denom * 100) if denom != 0 else 0.0
        dx.append(dx_val)
        
    if len(dx) < period:
        return 25.0
        
    adx_val = sum(dx[:period]) / period
    for i in range(period, len(dx)):
        adx_val = (adx_val * (period - 1) + dx[i]) / period
        
    return adx_val

def calculate_macd(prices: List[float]) -> tuple:
    if len(prices) < 26:
        return 0.0, 0.0, 0.0
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd_line = [e12 - e26 for e12, e26 in zip(ema_12, ema_26)]
    signal_line = calculate_ema(macd_line, 9)
    macd_hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line[-1], signal_line[-1], macd_hist[-1]

def calculate_bollinger_bands(prices: List[float], period: int = 20) -> tuple:
    if len(prices) < period:
        return (prices[-1] if prices else 0.0), 0.0, 0.0
    prices_subset = prices[-period:]
    sma = sum(prices_subset) / period
    variance = sum((x - sma) ** 2 for x in prices_subset) / period
    std_dev = variance ** 0.5
    upper_band = sma + 2 * std_dev
    lower_band = sma - 2 * std_dev
    bb_width = ((upper_band - lower_band) / sma) * 100 if sma > 0 else 0.0
    return sma, bb_width, std_dev

def calculate_realized_volatility(prices: List[float], period: int = 20) -> float:
    if len(prices) < period + 1:
        return 0.0
    returns = []
    for i in range(len(prices) - period, len(prices)):
        prev = prices[i-1]
        if prev > 0:
            returns.append((prices[i] - prev) / prev)
    if not returns:
        return 0.0
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 0.0
    std_dev = variance ** 0.5
    ann_vol = std_dev * math.sqrt(252) * 100
    return ann_vol

def calculate_vwap_deviation(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], period: int = 20) -> float:
    if len(closes) < period:
        return 0.0
    typical_prices = [(h + l + c) / 3.0 for h, l, c in zip(highs[-period:], lows[-period:], closes[-period:])]
    vols = volumes[-period:]
    sum_pv = sum(tp * v for tp, v in zip(typical_prices, vols))
    sum_v = sum(vols)
    if sum_v == 0:
        return 0.0
    vwap = sum_pv / sum_v
    current_price = closes[-1]
    deviation = ((current_price - vwap) / vwap) * 100
    return deviation

def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    tr = []
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        prev_c = closes[i-1]
        tr_val = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr.append(tr_val)
    
    atr = sum(tr[:period]) / period
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
    return atr

def get_max_asset_exposure(ticker: str, technical_data: dict) -> float:
    """Dynamically scales target allocation caps based on asset type and realized volatility."""
    if ticker in ["VOO", "QQQ", "SCHD"]:
        return 0.15
        
    ticker_data = technical_data.get(ticker, {})
    realized_vol = ticker_data.get("realized_vol_20d", 30.0)
    
    if ticker in CORE_UNIVERSE:
        return 0.10 if realized_vol < 35.0 else 0.05
        
    if realized_vol > 45.0:
        return 0.025
    else:
        return 0.05

def get_asymmetric_guardrails(ticker: str, technical_data: dict, current_price: float) -> tuple:
    """
    Returns asymmetric execution thresholds: (stop_loss_pct, take_profit_pct, trailing_pullback_pct)
    based on asset characteristics and Average True Range (ATR).
    All percentages are positive decimals (e.g., 0.05 for 5%).
    """
    stop_loss = 0.05
    take_profit = 0.08
    trailing_pullback = 0.04
    
    ticker_data = technical_data.get(ticker)
    if not ticker_data:
        return stop_loss, take_profit, trailing_pullback
        
    realized_vol = ticker_data.get("realized_vol_20d", 30.0)
    atr = ticker_data.get("atr", 0.0)
    
    atr_pct = (atr / current_price) if current_price > 0 and atr > 0 else 0.02
    
    if ticker in ["VOO", "QQQ", "SCHD"]:
        stop_loss = max(0.07, 3.0 * atr_pct)
        take_profit = max(0.06, 2.5 * atr_pct)
        trailing_pullback = 0.02 
    elif ticker in CORE_UNIVERSE: 
        stop_loss = max(0.05, 2.5 * atr_pct)
        take_profit = max(0.10, 3.5 * atr_pct)
        trailing_pullback = max(0.03, 1.5 * atr_pct)
    else:
        if realized_vol > 45.0:
            stop_loss = min(0.04, max(0.025, 1.5 * atr_pct)) 
            take_profit = max(0.12, 4.0 * atr_pct)
            trailing_pullback = min(0.03, 1.2 * atr_pct)
        else:
            stop_loss = min(0.05, max(0.03, 2.0 * atr_pct))
            take_profit = max(0.10, 3.0 * atr_pct)
            trailing_pullback = min(0.04, 1.5 * atr_pct)
            
    return stop_loss, take_profit, trailing_pullback

def get_technical_indicators(tickers: List[str]) -> dict:
    indicators = {}
    if not tickers: return indicators
    try:
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=150)
        )
        bars_dict = data_client.get_stock_bars(req)
        if not bars_dict: return {}
        
        for ticker, data_points in bars_dict.items():
            close_prices = [float(day["c"]) for day in data_points]
            highs = [float(day["h"]) for day in data_points]
            lows = [float(day["l"]) for day in data_points]
            volumes = [float(day["v"]) for day in data_points]
            
            if len(close_prices) == 0: continue
            
            current_price = close_prices[-1]
            current_vol = volumes[-1]
            
            sma_50 = sum(close_prices[-50:]) / min(len(close_prices), 50)
            rsi_14 = calculate_rsi(close_prices)
            
            lookback_20 = min(20, len(close_prices)-1)
            lookback_50 = min(50, len(close_prices)-1)
            
            ret_20d = ((current_price - close_prices[-(lookback_20+1)]) / close_prices[-(lookback_20+1)]) * 100 if lookback_20 > 0 else 0.0
            ret_50d = ((current_price - close_prices[-(lookback_50+1)]) / close_prices[-(lookback_50+1)]) * 100 if lookback_50 > 0 else 0.0
            
            dist_sma = ((current_price - sma_50) / sma_50) * 100 if sma_50 > 0 else 0.0
            
            avg_vol_20d = sum(volumes[-20:]) / min(len(volumes), 20) if len(volumes) > 0 else 1
            vol_ratio = current_vol / avg_vol_20d if avg_vol_20d > 0 else 1.0

            # Advanced Telemetry Calculations
            _, bb_width, std_dev = calculate_bollinger_bands(close_prices, 20)
            real_vol_20d = calculate_realized_volatility(close_prices, 20)
            adx_14 = calculate_adx(highs, lows, close_prices, 14)
            _, _, macd_hist = calculate_macd(close_prices)
            vwap_dev = calculate_vwap_deviation(highs, lows, close_prices, volumes, 20)
            atr_14 = calculate_atr(highs, lows, close_prices, 14)

            indicators[ticker] = {
                "rsi": round(rsi_14, 2), 
                "sma_50": round(sma_50, 2),
                "dist_sma": round(dist_sma, 2),
                "ret_20d": round(ret_20d, 2),
                "ret_50d": round(ret_50d, 2),
                "vol_ratio": round(vol_ratio, 2),
                "bb_width": round(bb_width, 2),
                "realized_vol_20d": round(real_vol_20d, 2),
                "adx": round(adx_14, 2),
                "macd_hist": round(macd_hist, 2),
                "vwap_deviation": round(vwap_dev, 2),
                "atr": round(atr_14, 2)
            }
    except Exception as e:
        print(f"!!! Error calculating advanced technicals: {e}")
    return indicators

def get_latest_market_prices(tickers: List[str]) -> dict:
    if not tickers: return {}
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=tickers)
        trades_dict = data_client.get_stock_latest_trade(req)
        if not trades_dict: return {}
        return {ticker: float(trade["p"]) for ticker, trade in trades_dict.items()}
    except Exception as e:
        print(f"!!! Error fetching live data: {e}")
        return {}

def is_market_open() -> bool:
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception as e:
        return True

# ==============================================================================
# 5. MULTI-LLM PROVIDER ROUTER (ADAPTER LAYER)
# ==============================================================================
def generate_portfolio_decision(prompt: str, system_instruction: str) -> PortfolioManagerDecision:
    """
    Standardizes structured generation between Gemini, Anthropic (Claude), and OpenAI APIs.
    Routes to the active provider and maps raw response schemas directly to
    the Pydantic 'PortfolioManagerDecision' model, preventing downstream KeyErrors.
    """
    if LLM_PROVIDER == "gemini":
        model_name = LLM_MODEL if LLM_MODEL else "gemini-3.5-flash"
        print(f"Routing to Google GenAI. Target Model: {model_name}")
        
        response = ai_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=PortfolioManagerDecision,
                max_output_tokens=65536,
                temperature=0.2
            )
        )
        return PortfolioManagerDecision.model_validate_json(response.text)
        
    elif LLM_PROVIDER == "anthropic":
        model_name = LLM_MODEL if LLM_MODEL else "claude-4.7-sonnet"
        print(f"Routing to Anthropic Console. Target Model: {model_name}")
        
        # Build JSON Schema mapping from Pydantic
        schema = PortfolioManagerDecision.model_json_schema()
        
        # Define the submission tool for forced tool calling
        tools = [
            {
                "name": "submit_portfolio_decision",
                "description": "Submit structured allocations and macro briefings from the Portfolio Manager.",
                "input_schema": schema
            }
        ]
        
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=4000,
            system=system_instruction,
            messages=[
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            tool_choice={"type": "tool", "name": "submit_portfolio_decision"},
            temperature=0.2
        )
        
        # Retrieve the structured tool call from Claude response blocks
        tool_use_block = None
        for block in response.content:
            if hasattr(block, 'type') and block.type == 'tool_use':
                tool_use_block = block
                break
                
        if not tool_use_block:
            raise ValueError("ValidationError: Anthropic model failed to issue structured tool call 'submit_portfolio_decision'.")
            
        # Pydantic validates input dictionary, resolving schema keys
        input_data = tool_use_block.input
        return PortfolioManagerDecision.model_validate(input_data)
        
    elif LLM_PROVIDER == "openai":
        model_name = LLM_MODEL if LLM_MODEL else "gpt-5.5"
        print(f"Routing to OpenAI. Target Model: {model_name}")
        
        response = openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format=PortfolioManagerDecision,
            temperature=0.2
        )
        return response.choices[0].message.parsed
        
    else:
        raise ValueError(f"CRITICAL ERROR: Unsupported LLM_PROVIDER '{LLM_PROVIDER}' configured.")

# ==============================================================================
# 6. CORE AUTONOMOUS EXECUTION ENGINE
# ==============================================================================
def run_market_manager_cycle():
    print(f"\n========================================================")
    current_time = datetime.now()
    print(f"CYCLE TRIGGERED AT: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"========================================================")
    
    print("Executing Real-Time Account Balance Sheet Audit...")
    positions = trading_client.get_all_positions()
    executed_logs = []
    active_positions_dict = {}
    sector_exposure_dict = {}
    
    held_tickers = [pos.symbol for pos in positions]
    held_bars = {}
    held_technicals = {}
    if held_tickers:
        try:
            held_bars_req = StockBarsRequest(symbol_or_symbols=held_tickers, timeframe=TimeFrame.Day, start=current_time - timedelta(days=30))
            held_bars = data_client.get_stock_bars(held_bars_req)
        except Exception: pass
        held_technicals = get_technical_indicators(held_tickers)

    # ----------------------------------------------------------------------
    # PHASE 0: DETERMINISTIC RISK ENGINE (Trailing Stops & Hard Cuts)
    # ----------------------------------------------------------------------
    print("--- Phase 0: Executing Deterministic Python Risk Engine ---")
    for pos in positions:
        ticker = pos.symbol
        qty = int(pos.qty)
        avg_entry = float(pos.avg_entry_price)
        current_market_value = float(pos.market_value)
        current_price = current_market_value / qty
        pnl_pct = (current_price - avg_entry) / avg_entry
        
        sector = get_asset_sector(ticker)
        sector_exposure_dict[sector] = sector_exposure_dict.get(sector, 0.0) + current_market_value
        
        stop_loss_pct, take_profit_pct, trailing_pullback_pct = get_asymmetric_guardrails(ticker, held_technicals, current_price)
        
        if pnl_pct <= -stop_loss_pct:
            alert_msg = f"⚠️ **RISK BREACH:** {ticker} dropped {pnl_pct*105:.2f}% (Stop Loss threshold: {stop_loss_pct*100:.2f}%). Executing hard stop liquidation."
            print(f"!!! {alert_msg}")
            send_discord_message(alert_msg)
            trading_client.submit_order(order_data=MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC))
            executed_logs.append({"timestamp": current_time.isoformat(), "ticker": ticker, "action": "SELL", "shares": qty, "thesis": f"FORCED HARD STOP-LOSS AT -{stop_loss_pct*100:.1f}%."})
            sector_exposure_dict[sector] -= current_market_value 
            continue 
            
        elif pnl_pct >= take_profit_pct:
            try:
                highs = [float(day["h"]) for day in held_bars.get(ticker, [])]
                if highs:
                    recent_peak = max(highs)
                    drawdown_from_peak = (current_price - recent_peak) / recent_peak
                    current_rsi = held_technicals.get(ticker, {}).get("rsi", 50)
                    
                    if drawdown_from_peak <= -trailing_pullback_pct:
                        alert_msg = f"💰 **TRAILING STOP TRIGGERED:** {ticker} pulled back {abs(drawdown_from_peak)*100:.2f}% from peak (limit: {trailing_pullback_pct*100:.2f}%) with degrading momentum (RSI {current_rsi}). Locking in {pnl_pct*100:.2f}% profit."
                        print(f"$$$ {alert_msg}")
                        send_discord_message(alert_msg)
                        trading_client.submit_order(order_data=MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC))
                        executed_logs.append({"timestamp": current_time.isoformat(), "ticker": ticker, "action": "SELL", "shares": qty, "thesis": f"FORCED TRAILING TAKE-PROFIT ({trailing_pullback_pct*100:.1f}% trailing)."})
                        sector_exposure_dict[sector] -= current_market_value 
                        continue 
            except Exception:
                pass 
            
        active_positions_dict[ticker] = {
            "qty": qty, "market_value": current_market_value, "unrealized_pnl": float(pos.unrealized_pl)
        }

    cooldown_blacklist = build_cooldown_blacklist()
    if cooldown_blacklist:
        print(f"System Cooldown Active for recently liquidated assets: {cooldown_blacklist}")

    # ----------------------------------------------------------------------
    # PHASE 1: FREQUENCY-WEIGHTED DISCOVERY & HYBRID MERGE
    # ----------------------------------------------------------------------
    print("--- Phase 1: Scanning Global Market News Wire ---")
    try:
        req = NewsRequest(limit=40)
        news_response = news_client.get_news(req)
        news_list = news_response.get("news", [])
    except Exception as e:
        news_list = []

    raw_discovered_symbols = []
    live_news_feed = "=== GLOBAL MARKET NEWS FIREHOSE ===\n"
    
    for article in news_list:
        headline = article.get("headline", "")
        summary = article.get("summary", "").replace('\n', ' ') if article.get("summary") else ""
        live_news_feed += f"- {headline}\n  {summary[:180]}...\n"
        
        article_symbols = article.get("symbols", [])
        for sym in article_symbols:
            if sym.isalpha() and len(sym) <= 5:
                raw_discovered_symbols.append(sym)
                
    symbol_frequencies = Counter(raw_discovered_symbols)
    trending_tickers = [ticker for ticker, count in symbol_frequencies.most_common(15)]
    print(f"Top Trending Assets Found in News Flow: {trending_tickers}")

    dynamic_watchlist = list(set(CORE_UNIVERSE + trending_tickers + list(active_positions_dict.keys())))

    # ----------------------------------------------------------------------
    # PHASE 2: ADVANCED TELEMETRY INGESTION
    # ----------------------------------------------------------------------
    print("--- Phase 2: Pulling Advanced Telemetry for Dynamic Watchlist ---")
    market_prices = get_latest_market_prices(dynamic_watchlist)
    technical_data = get_technical_indicators(dynamic_watchlist)
    
    account = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    
    state_summary = f"=== PORTFOLIO BALANCE SHEET ===\nTotal Equity: ${portfolio_value:,.2f} | Liquid Cash: ${float(account.cash):,.2f}\n\n=== ACTIVE POSITIONS ===\n"
    if not active_positions_dict: state_summary += "No active positions.\n"
    for ticker, data in active_positions_dict.items():
        state_summary += f"- {ticker} ({get_asset_sector(ticker)}): {data['qty']} shares | Value: ${data['market_value']:,.2f} | PnL: ${data['unrealized_pnl']:,.2f}\n"

    watchlist_context = "=== DYNAMIC TELEMETRY (PRICE, RSI, SMA, DISTANCE TO SMA, VOLATILITY, ADX, MACD, VWAP, BB WIDTH, VOL RATIO) ===\n"
    for ticker in dynamic_watchlist:
        price = market_prices.get(ticker, 0.0)
        ind = technical_data.get(ticker, {})
        rsi = ind.get("rsi", "N/A")
        dist_sma = ind.get("dist_sma", "N/A")
        r20 = ind.get("ret_20d", "N/A")
        r50 = ind.get("ret_50d", "N/A")
        v_rat = ind.get("vol_ratio", "N/A")
        bb_w = ind.get("bb_width", "N/A")
        r_vol = ind.get("realized_vol_20d", "N/A")
        adx = ind.get("adx", "N/A")
        macd = ind.get("macd_hist", "N/A")
        vwap = ind.get("vwap_deviation", "N/A")
        atr = ind.get("atr", "N/A")
        
        watchlist_context += (
            f"- {ticker} ({get_asset_sector(ticker)}): Price ${price:,.2f} | RSI: {rsi} | Dist_SMA: {dist_sma}% | "
            f"20d Ret: {r20}% | 50d Ret: {r50}% | Vol Ratio: {v_rat}x | BB Width: {bb_w}% | 20d Real Vol: {r_vol}% | "
            f"ADX: {adx} | MACD Hist: {macd} | VWAP Dev: {vwap}% | ATR: {atr}\n"
        )

    # Broad Market Regime Filter
    voo_ind = technical_data.get("VOO", {})
    qqq_ind = technical_data.get("QQQ", {})
    voo_dist_sma = voo_ind.get("dist_sma", 0.0)
    qqq_dist_sma = qqq_ind.get("dist_sma", 0.0)
    
    is_defensive = False
    if voo_dist_sma < -2.0 or qqq_dist_sma < -2.0:
        is_defensive = True
        
    if is_defensive:
        regime_instruction = (
            "🚨 DEFENSIVE CAP-PRESERVATION MODE TRIGGERED 🚨\n"
            "The broad market (VOO/QQQ) is trading deeply below its 50-day SMA (VOO Dist: {voo_dist_sma}%, QQQ Dist: {qqq_dist_sma}%).\n"
            "As Chief Investment Officer, you MUST prioritize CAPITAL PRESERVATION:\n"
            "- Do NOT issue BUY signals unless you have extreme conviction (confidence_score >= 90) on core stable indexes (VOO, SCHD) or outstanding breakout candidates.\n"
            "- Favor HOLDing or SELLing existing volatile dynamic positions.\n"
            "- Be highly conservative and assign very small risk allocation percentages (e.g. 0.0 - 2.5%) if buying."
        ).format(voo_dist_sma=voo_dist_sma, qqq_dist_sma=qqq_dist_sma)
    else:
        regime_instruction = (
            "✅ NORMAL RISK-ON MODE ACTIVE ✅\n"
            "The broad market is in an uptrend or trading near its 50-day SMA (VOO Dist: {voo_dist_sma}%, QQQ Dist: {qqq_dist_sma}%).\n"
            "You may selectively seek high-conviction momentum and mean-reversion trades across core and trending stocks."
        ).format(voo_dist_sma=voo_dist_sma, qqq_dist_sma=qqq_dist_sma)

    prompt = f"{state_summary}\n\n{watchlist_context}\n\n{live_news_feed}"
    
    system_instruction = (
        "You are the Chief Investment Officer managing a hybrid AI hedge fund utilizing momentum, volume distribution, mean-reversion (Distance to SMA), and macro catalysts.\n"
        "Your workspace contains structural core stable index assets, current active inventory, and trending breaking news targets.\n\n"
        f"{regime_instruction}\n\n"
        "REQUIRED REASONING SEQUENCE (Chain-of-Thought):\n"
        "You must follow this exact analysis sequence inside the 'chain_of_thought_reasoning' and 'chain_of_thought_analysis' fields:\n"
        "1. REGIME IDENTIFICATION: Explicitly analyze broad market conditions (VOO/QQQ SMA distance, volatility, ADX trend strength) and classify the regime (e.g., Bullish Momentum, Volatile Mean-Reversion, or Bearish Distribution).\n"
        "2. ASYMMETRIC RISK-REWARD: Check for entry setups. Buy pullbacks near the 50-day SMA, but avoid overextended assets. Never BUY assets with RSI > 70.\n"
        "3. VOLUME CONFIRMATION: Verify breaking news breakouts with Volume Ratios > 1.2x. Do not trade breakouts on thin volume.\n\n"
        "CRITICAL ALLOCATION & EXECUTION RULES:\n"
        "1. INVENTORY AWARENESS: You may ONLY issue a 'SELL' signal if the ticker is explicitly listed under 'ACTIVE POSITIONS'.\n"
        "2. DIVERSIFICATION MANDATE: Balance allocations across sectors (Technology, Financials, Healthcare, Consumer, Broad Market).\n"
        "3. DYNAMIC CONVICTION SCALING:\n"
        "   - Core low-volatility indices (VOO, QQQ, SCHD) can scale up to 10.0% to 15.0% for maximum confidence.\n"
        "   - Core individual stocks (AAPL, MSFT, NVDA) can scale up to 5.0% to 10.0%.\n"
        "   - Volatile, trending, or speculative stocks MUST be capped at 2.5% to 5.0%.\n"
        "4. COMPACT RESPONSE: Keep analysis concise and direct."
    )

    # ==============================================================================
    # PHASE 3: PORTFOLIO ALLOCATION (ROUTED DYNAMICALLY)
    # ==============================================================================
    print("--- Phase 3: Forwarding Data To Chief Investment Officer ---")
    try:
        portfolio_decision = generate_portfolio_decision(prompt, system_instruction)
    except Exception as e:
        print(f"!!! Orchestration Engine Error or JSON validation failure: {e}")
        log_portfolio_trades(executed_logs)
        return

    discord_macro_payload = (
        f"📊 **[CIO MACRO BRIEFING]**\n"
        f"```\n{portfolio_decision.macro_strategy_outlook}\n```\n"
        f"**Portfolio Equity:** ${portfolio_value:,.2f} | **Cash Available:** ${float(account.cash):,.2f}"
    )
    print(f"\n================ EXECUTIVE MACRO SUMMARY ================\n{portfolio_decision.macro_strategy_outlook}\n========================================================")
    send_discord_message(discord_macro_payload)
    
    # ----------------------------------------------------------------------
    # PHASE 4: TRANSACTION PROCESSING (SELLS THEN BUYS)
    # ----------------------------------------------------------------------
    print("\n--- Phase 4: Processing AI Liquidations (SELL Orders) ---")
    for decision in portfolio_decision.asset_decisions:
        if decision.signal == "SELL" and decision.ticker in active_positions_dict:
            owned_qty = active_positions_dict[decision.ticker]["qty"]
            log_text = f"🛑 **AI Discretionary SELL:** Liquidating {owned_qty} units of **{decision.ticker}**\n*Thesis:* {decision.analysis}"
            print(log_text)
            send_discord_message(log_text)
            trading_client.submit_order(order_data=MarketOrderRequest(symbol=decision.ticker, qty=owned_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC))
            executed_logs.append({"timestamp": current_time.isoformat(), "ticker": decision.ticker, "action": "SELL", "shares": owned_qty, "thesis": decision.analysis, "lessons": decision.lessons_applied})
            
            asset_sector = get_asset_sector(decision.ticker)
            sector_exposure_dict[asset_sector] -= active_positions_dict[decision.ticker]["market_value"]

    print("\n--- Phase 5: Processing Strategic Capital Deployment (BUY Orders) ---")
    available_liquid_cash = float(account.cash)

    for decision in portfolio_decision.asset_decisions:
        if decision.signal == "BUY" and decision.confidence_score >= 75:
            
            if decision.ticker in cooldown_blacklist:
                print(f"⚠️ Guardrail Active: Blocking {decision.ticker} BUY signal due to recent forced liquidation.")
                continue

            spot_price = market_prices.get(decision.ticker)
            if not spot_price: continue
            
            asset_sector = get_asset_sector(decision.ticker)
            current_sector_dollars = sector_exposure_dict.get(asset_sector, 0.0)
            max_sector_dollars = portfolio_value * MAX_SECTOR_EXPOSURE
            
            max_exposure = get_max_asset_exposure(decision.ticker, technical_data)
            
            safe_allocation_pct = min(decision.risk_allocation_percentage, max_exposure * 100) / 100.0
            target_dollar_allocation = portfolio_value * safe_allocation_pct
            
            if (current_sector_dollars + target_dollar_allocation) > max_sector_dollars:
                print(f"🛡️ Sector Guardrail Active: Blocking {decision.ticker}. {asset_sector} exceeds {MAX_SECTOR_EXPOSURE*100}% allocation limit.")
                continue

            if target_dollar_allocation > available_liquid_cash:
                print(f"⚠️ Insufficient Cash Guardrail: Skipping {decision.ticker}. Needed: ${target_dollar_allocation:,.2f}, Available: ${available_liquid_cash:,.2f}")
                continue 
                
            calculated_qty = int(target_dollar_allocation // spot_price)
            current_owned = active_positions_dict.get(decision.ticker, {}).get("qty", 0)
            
            if calculated_qty > current_owned:
                order_qty = calculated_qty - current_owned
                order_cost = order_qty * spot_price
                
                if order_cost > available_liquid_cash:
                    continue
                    
                log_text = (
                    f"🚀 **AI Dynamic BUY:** **{decision.ticker}** ({asset_sector}) | Conviction: {decision.confidence_score}%\n"
                    f"Order: Purchasing {order_qty} units (~${order_cost:,.2f})\n"
                    f"*Thesis:* {decision.analysis}"
                )
                print(log_text)
                send_discord_message(log_text)
                
                try:
                    trading_client.submit_order(order_data=MarketOrderRequest(symbol=decision.ticker, qty=order_qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                    executed_logs.append({"timestamp": current_time.isoformat(), "ticker": decision.ticker, "action": "BUY", "shares": order_qty, "thesis": decision.analysis, "lessons": decision.lessons_applied})
                    
                    available_liquid_cash -= order_cost
                    sector_exposure_dict[asset_sector] = current_sector_dollars + order_cost
                    
                except Exception as e:
                    print(f"!!! Brokerage rejected order for {decision.ticker}: {e}")
                
    log_portfolio_trades(executed_logs)
    print("\n--- Portfolio Management Cycle Complete ---")

# ==============================================================================
# 7. APPLICATION LIFECYCLE CONTROLLER
# ==============================================================================
# Structural Core Layer
CORE_UNIVERSE = ["VOO", "QQQ", "SCHD", "AAPL", "MSFT", "NVDA"]

# Execution Tuning
LOOP_INTERVAL = 1170 
COOLDOWN_DAYS = 3 

# Portfolio Risk Constraints
MAX_SECTOR_EXPOSURE = 0.25 # Maximum 25% of total portfolio equity in a single sector

# Sector Mapping Dictionary
SECTOR_MAP = {
    # Tech / Semi
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AVGO": "Technology", "AMD": "Technology",
    "QCOM": "Technology", "PANW": "Technology", "PLTR": "Technology", "INTC": "Technology", "TXN": "Technology",
    "ORCL": "Technology", "CRM": "Technology", "CSCO": "Technology", "ASML": "Technology", "SMCI": "Technology",
    "QQQ": "Technology", "XLK": "Technology",
    # Communications
    "GOOG": "Communications", "META": "Communications", "NFLX": "Communications", "DIS": "Communications", "XLC": "Communications",
    # Financials
    "JPM": "Financials", "V": "Financials", "GS": "Financials", "BAC": "Financials", "MS": "Financials", "MA": "Financials", "XLF": "Financials",
    # Healthcare
    "LLY": "Healthcare", "UNH": "Healthcare", "VRTX": "Healthcare", "JNJ": "Healthcare", "MRK": "Healthcare", "XLV": "Healthcare",
    # Consumer Discretionary
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer", "MCD": "Consumer", "XLY": "Consumer",
    # Industrials & Energy
    "CAT": "Industrials", "GE": "Industrials", "XOM": "Energy", "CVX": "Energy", "XLI": "Industrials", "XLE": "Energy",
    # Broad Market / Other
    "VOO": "Broad Market", "SCHD": "Broad Market", "TLT": "Fixed Income", "GLD": "Commodities"
}

def get_next_market_open() -> datetime:
    """
    Calculates the exact datetime of the next market open (9:30 AM US/Eastern).
    If today is a weekday and it is before 9:30 AM Eastern, the next open is today at 9:30 AM.
    Otherwise, it is 9:30 AM Eastern on the next weekday.
    """
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    
    # Today at 9:30 AM Eastern
    today_open = now_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If today is a weekday (Monday=0 to Friday=4) and we are before 9:30 AM Eastern today
    if now_eastern.weekday() < 5 and now_eastern < today_open:
        return today_open
        
    # Otherwise, loop to find the next weekday
    next_day = now_eastern + timedelta(days=1)
    while True:
        if next_day.weekday() < 5:
            return next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        next_day += timedelta(days=1)

def sleep_until_next_cycle():
    """
    Cadence controller that sleeps for standard LOOP_INTERVAL during market hours,
    or calculates the exact seconds remaining until 9:30 AM Eastern of the next
    trading day when the market is closed, preventing timer overlap/drift.
    """
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    
    hour = now_eastern.hour
    minute = now_eastern.minute
    is_weekday = now_eastern.weekday() < 5
    
    is_during_market_hours = False
    if is_weekday and (9 <= hour < 16):
        if hour == 9:
            is_during_market_hours = (minute >= 30)
        else:
            is_during_market_hours = True
            
    if is_during_market_hours:
        print(f"Market is active. Sleeping for standard interval: {LOOP_INTERVAL} seconds.")
        time.sleep(LOOP_INTERVAL)
    else:
        next_open = get_next_market_open()
        time_to_sleep = (next_open - now_eastern).total_seconds()
        
        # Add a 15-second safety buffer to ensure endpoints are updated and open
        time_to_sleep = max(10, time_to_sleep + 15)
        
        wake_time_str = next_open.strftime('%Y-%m-%d %H:%M:%S %Z')
        print(f"Market is closed. Sleeping for {time_to_sleep:.1f} seconds until next market open: {wake_time_str}\n")
        time.sleep(time_to_sleep)

def main():
    print("Initializing Autonomous Live Engine...")
    print(f"Operational Config: Ingestion cadence anchored to {LOOP_INTERVAL} second intervals.")
    print(f"Active Provider: {LLM_PROVIDER.upper()}")
    send_discord_message(f"🟢 **System Online:** Autonomous V6.1 Production Engine initialized successfully. Provider: {LLM_PROVIDER.upper()}")
    
    while True:
        try:
            if is_market_open():
                run_market_manager_cycle()
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Equities market is closed. Suspending AI allocation engine.")
            
            sleep_until_next_cycle()
            
        except KeyboardInterrupt:
            send_discord_message("🔴 **System Offline:** Manual termination sequence captured.")
            sys.exit(0)
        except Exception as e:
            err_msg = f"🚨 **Runtime Exception in Container Loop:** {e}"
            print(f"!!! {err_msg}")
            send_discord_message(err_msg)
            time.sleep(60)

if __name__ == "__main__":
    main()
