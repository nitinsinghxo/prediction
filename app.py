# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import uvicorn

app = FastAPI(title="Smart Stock Recommender")

# Model placeholder - replace with real model later
def simple_model(stock_data, expert_opinion=None):
    # Dummy rule: If latest close > open -> BUY, else SELL
    last_row = stock_data.iloc[-1]
    recommendation = "BUY" if last_row['Close'] > last_row['Open'] else "SELL"
    # Adjust recommendation based on expert opinion
    if expert_opinion:
        if expert_opinion.lower() in ["buy", "hold", "strong buy"]:
            recommendation = "BUY"
        elif expert_opinion.lower() in ["sell", "strong sell"]:
            recommendation = "SELL"
    return recommendation

# Request schema for expert input
class ExpertInput(BaseModel):
    ticker: str
    opinion: str

# Response schema for recommendations
class StockRecommendation(BaseModel):
    ticker: str
    industry: str
    recommendation: str
    recent_close: float
    recent_open: float

# Cache stocks list and industries
stocks_cache: Dict[str, str] = {}

def fetch_sp500_stocks():
    # Scrape Wikipedia for S&P 500 tickers and industries
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    df = pd.read_html(str(table))[0]
    return df[['Symbol', 'GICS Sector']]

def initialize_stocks_cache():
    global stocks_cache
    df = fetch_sp500_stocks()
    stocks_cache = dict(zip(df['Symbol'], df['GICS Sector']))

@app.on_event("startup")
async def startup_event():
    initialize_stocks_cache()

@app.get("/stocks", response_model=List[str])
async def get_all_stocks():
    return list(stocks_cache.keys())

@app.get("/industries", response_model=List[str])
async def get_all_industries():
    return sorted(list(set(stocks_cache.values())))

@app.get("/stocks/by_industry/{industry}", response_model=List[str])
async def get_stocks_by_industry(industry: str):
    return [ticker for ticker, ind in stocks_cache.items() if ind.lower() == industry.lower()]

@app.get("/recommendations", response_model=List[StockRecommendation])
async def get_recommendations(industry: Optional[str] = None):
    # Analyze all stocks or by industry
    filtered_stocks = {k: v for k, v in stocks_cache.items() if (industry is None or v.lower() == industry.lower())}
    results = []
    for ticker, sector in filtered_stocks.items():
        try:
            df = yf.download(ticker, period="1mo", interval="1d", progress=False)
            if df.empty or len(df) < 2:
                continue
            rec = simple_model(df)
            results.append(StockRecommendation(
                ticker=ticker,
                industry=sector,
                recommendation=rec,
                recent_close=round(df.iloc[-1]['Close'], 2),
                recent_open=round(df.iloc[-1]['Open'], 2)
            ))
        except Exception:
            continue
    return results

# Expert advice storage (in-memory for demo)
expert_opinions = {}

@app.post("/expert_advice", response_model=StockRecommendation)
async def post_expert_advice(data: ExpertInput):
    ticker = data.ticker.upper()
    if ticker not in stocks_cache:
        raise HTTPException(status_code=404, detail="Ticker not found")
    expert_opinions[ticker] = data.opinion
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if df.empty or len(df) < 2:
            raise HTTPException(status_code=404, detail="No sufficient stock data")
        rec = simple_model(df, expert_opinion=data.opinion)
        return StockRecommendation(
            ticker=ticker,
            industry=stocks_cache[ticker],
            recommendation=rec,
            recent_close=round(df.iloc[-1]['Close'], 2),
            recent_open=round(df.iloc[-1]['Open'], 2)
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Error analyzing stock data")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
