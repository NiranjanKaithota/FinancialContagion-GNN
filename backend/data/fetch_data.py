"""
Layer 1: Data Ingestion
Fetches NIFTY 50 stock data from Yahoo Finance.
Saves: prices.csv, returns.csv, metadata.json
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# NIFTY 50 tickers (Yahoo Finance format uses .NS suffix for NSE)
NIFTY50_TICKERS = {
    # Banking & Finance
    "HDFCBANK.NS":  {"name": "HDFC Bank",        "sector": "Banking"},
    "ICICIBANK.NS": {"name": "ICICI Bank",        "sector": "Banking"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra",    "sector": "Banking"},
    "AXISBANK.NS":  {"name": "Axis Bank",         "sector": "Banking"},
    "SBIN.NS":      {"name": "State Bank",        "sector": "Banking"},
    "BAJFINANCE.NS":{"name": "Bajaj Finance",     "sector": "Finance"},
    "BAJAJFINSV.NS":{"name": "Bajaj Finserv",     "sector": "Finance"},

    # IT
    "TCS.NS":       {"name": "TCS",               "sector": "IT"},
    "INFY.NS":      {"name": "Infosys",           "sector": "IT"},
    "WIPRO.NS":     {"name": "Wipro",             "sector": "IT"},
    "HCLTECH.NS":   {"name": "HCL Tech",          "sector": "IT"},
    "TECHM.NS":     {"name": "Tech Mahindra",     "sector": "IT"},

    # Energy & Oil
    "RELIANCE.NS":  {"name": "Reliance",          "sector": "Energy"},
    "ONGC.NS":      {"name": "ONGC",              "sector": "Energy"},
    "POWERGRID.NS": {"name": "Power Grid",        "sector": "Energy"},
    "NTPC.NS":      {"name": "NTPC",              "sector": "Energy"},
    "COALINDIA.NS": {"name": "Coal India",        "sector": "Energy"},

    # Consumer & FMCG
    "HINDUNILVR.NS":{"name": "HUL",               "sector": "FMCG"},
    "ITC.NS":       {"name": "ITC",               "sector": "FMCG"},
    "NESTLEIND.NS": {"name": "Nestle India",      "sector": "FMCG"},
    "BRITANNIA.NS": {"name": "Britannia",         "sector": "FMCG"},

    # Pharma
    "SUNPHARMA.NS": {"name": "Sun Pharma",        "sector": "Pharma"},
    "DRREDDY.NS":   {"name": "Dr Reddy's",        "sector": "Pharma"},
    "CIPLA.NS":     {"name": "Cipla",             "sector": "Pharma"},
    "DIVISLAB.NS":  {"name": "Divi's Labs",       "sector": "Pharma"},

    # Automobiles
    "MARUTI.NS":    {"name": "Maruti Suzuki",     "sector": "Auto"},
    "TATAMOTORS.NS":{"name": "Tata Motors",       "sector": "Auto"},
    "M&M.NS":       {"name": "Mahindra",          "sector": "Auto"},
    "BAJAJ-AUTO.NS":{"name": "Bajaj Auto",        "sector": "Auto"},

    # Infrastructure & Metals
    "ADANIPORTS.NS":{"name": "Adani Ports",       "sector": "Infra"},
    "ULTRACEMCO.NS":{"name": "UltraTech Cement",  "sector": "Infra"},
    "TATASTEEL.NS": {"name": "Tata Steel",        "sector": "Metals"},
    "JSWSTEEL.NS":  {"name": "JSW Steel",         "sector": "Metals"},
    "HINDALCO.NS":  {"name": "Hindalco",          "sector": "Metals"},

    # Telecom & Others
    "BHARTIARTL.NS":{"name": "Bharti Airtel",     "sector": "Telecom"},
    "INDUSINDBK.NS":{"name": "IndusInd Bank",     "sector": "Banking"},
    "ASIANPAINT.NS":{"name": "Asian Paints",      "sector": "Consumer"},
    "TITAN.NS":     {"name": "Titan",             "sector": "Consumer"},
    "LTIM.NS":      {"name": "LTIMindtree",       "sector": "IT"},
}

SECTOR_COLORS = {
    "Banking":  "#E24B4A",
    "Finance":  "#D4537E",
    "IT":       "#378ADD",
    "Energy":   "#EF9F27",
    "FMCG":     "#1D9E75",
    "Pharma":   "#5DCAA5",
    "Auto":     "#7F77DD",
    "Infra":    "#D85A30",
    "Metals":   "#888780",
    "Telecom":  "#639922",
    "Consumer": "#BA7517",
}

def fetch_prices(start="2018-01-01", end=None):
    """Download adjusted close prices for all NIFTY 50 stocks."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    tickers = list(NIFTY50_TICKERS.keys())
    print(f"Fetching {len(tickers)} tickers from {start} to {end}...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]

    # Drop columns with too many NaNs (>20% missing)
    threshold = 0.2 * len(prices)
    prices = prices.dropna(axis=1, thresh=int(len(prices) - threshold))

    # Forward fill then drop remaining NaN rows
    prices = prices.ffill().dropna()

    print(f"  Retained {prices.shape[1]} tickers, {prices.shape[0]} trading days")
    return prices


def compute_returns(prices):
    """Daily log returns."""
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def compute_rolling_stats(returns, window=252):
    """Per-stock: annualized vol, beta vs NIFTY 50 index, mean return."""
    stats = {}
    # Use equal-weight index as market proxy
    market = returns.mean(axis=1)

    for col in returns.columns:
        r = returns[col]
        vol = r.rolling(window).std().iloc[-1] * np.sqrt(252)
        cov = r.rolling(window).cov(market).iloc[-1]
        var = market.rolling(window).var().iloc[-1]
        beta = cov / var if var > 0 else 1.0
        mu = r.mean() * 252  # annualised
        stats[col] = {
            "vol": round(float(vol), 4),
            "beta": round(float(beta), 4),
            "mu": round(float(mu), 4),
        }
    return stats


def build_metadata(prices, rolling_stats):
    """Combine ticker info with computed stats."""
    metadata = {}
    for ticker in prices.columns:
        info = NIFTY50_TICKERS.get(ticker, {"name": ticker, "sector": "Unknown"})
        stats = rolling_stats.get(ticker, {})
        metadata[ticker] = {
            "ticker": ticker,
            "name": info["name"],
            "sector": info["sector"],
            "color": SECTOR_COLORS.get(info["sector"], "#888780"),
            **stats,
        }
    return metadata


def save_outputs(prices, returns, metadata, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prices.to_csv(f"{out_dir}/prices.csv")
    returns.to_csv(f"{out_dir}/returns.csv")
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved prices.csv, returns.csv, metadata.json → {out_dir}")


if __name__ == "__main__":
    OUT_DIR = os.path.join(os.path.dirname(__file__), "../../outputs")

    prices = fetch_prices(start="2018-01-01")
    returns = compute_returns(prices)
    rolling_stats = compute_rolling_stats(returns)
    metadata = build_metadata(prices, rolling_stats)

    save_outputs(prices, returns, metadata, OUT_DIR)

    print("\nSample metadata:")
    for t, m in list(metadata.items())[:3]:
        print(f"  {t}: {m}")

    print(f"\nPrice data shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
