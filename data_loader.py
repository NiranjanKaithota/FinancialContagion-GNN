"""
data_loader.py
==============
Real-data pipeline for the Financial Contagion GNN.

Pulls 50-stock daily OHLCV from Yahoo Finance (2015-2024), engineers a rich
node-feature matrix X and a rolling correlation-based adjacency tensor A, and
generates *real* binary crash labels derived from max-drawdown in a 90-day
forward window.  Every snapshot is saved so the walk-forward backtest can
replay the full history without re-downloading.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. UNIVERSE & PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
STOCK_UNIVERSE = {
    "Tech":       ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL",
                   "ADBE", "CRM",  "AMD",  "CSCO", "INTC"],
    "Finance":    ["JPM",  "BAC",  "WFC",  "MS",   "GS",
                   "C",    "BLK",  "SPGI", "AXP",  "V"],
    "Energy":     ["XOM",  "CVX",  "COP",  "EOG",  "SLB",
                   "MPC",  "HAL",  "VLO",  "OXY",  "PSX"],
    "Healthcare": ["UNH",  "JNJ",  "LLY",  "MRK",  "ABBV",
                   "PFE",  "TMO",  "DHR",  "ABT",  "BMY"],
    "Consumer":   ["AMZN", "WMT",  "PG",   "HD",   "COST",
                   "KO",   "PEP",  "MCD",  "NKE",  "SBUX"],
}

TICKERS      = [t for v in STOCK_UNIVERSE.values() for t in v]
SECTOR_MAP   = {t: s for s, ts in STOCK_UNIVERSE.items() for t in ts}

DATA_START   = "2015-01-01"
DATA_END     = "2024-12-31"

# Graph construction
CORR_WINDOW      = 252      # 1-year rolling correlation window (trading days)
CORR_THRESHOLD   = 0.45     # minimum |corr| to draw an edge

# Feature engineering windows
VOL_WINDOW   = 21           # 1-month rolling volatility
MOM_SHORT    = 21
MOM_LONG     = 63           # 3-month momentum
BETA_WINDOW  = 63           # market-beta regression window

# Label generation
LABEL_HORIZON    = 63       # ~3 months forward look
CRASH_THRESHOLD  = -0.20    # max-drawdown < -20 % → crash label = 1

CACHE_DIR    = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. DOWNLOAD & CACHE PRICE DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_prices() -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "prices.parquet")
    if os.path.exists(cache_path):
        print(f"[data_loader] Loading prices from cache: {cache_path}")
        prices = pd.read_parquet(cache_path)
    else:
        print("[data_loader] Downloading prices from Yahoo Finance …")
        raw = yf.download(
            TICKERS,
            start=DATA_START,
            end=DATA_END,
            auto_adjust=True,
            threads=True,
            progress=True,
        )["Close"]
        prices = raw.ffill().bfill()
        prices.to_parquet(cache_path)
        print(f"[data_loader] Saved to {cache_path}. Shape: {prices.shape}")
    return prices


# ──────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING  (node features X)
# ──────────────────────────────────────────────────────────────────────────────
def build_features(prices: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    """
    Returns a MultiIndex DataFrame: rows = (date, ticker), columns = features.
    Features per node:
      - ret_1d, ret_5d, ret_21d       : raw returns at multiple horizons
      - vol_21d                        : 21-day realised volatility (ann.)
      - mom_short, mom_long            : price momentum (relative)
      - rsi_14                         : RSI oscillator
      - beta_63d                       : rolling market beta vs SPY
      - sector_one_hot (5 cols)        : sector membership
    """
    ret  = prices.pct_change()
    logr = np.log(prices / prices.shift(1))

    # Multi-horizon returns
    ret_1d  = ret
    ret_5d  = prices.pct_change(5)
    ret_21d = prices.pct_change(21)

    # Annualised volatility
    vol_21d = ret.rolling(VOL_WINDOW).std() * np.sqrt(252)

    # Momentum: (price / SMA) - 1
    mom_short = prices / prices.rolling(MOM_SHORT).mean() - 1
    mom_long  = prices / prices.rolling(MOM_LONG).mean() - 1

    # RSI-14
    delta = ret.copy()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    rsi   = 100 - (100 / (1 + rs))

    # Rolling market beta (OLS slope vs SPY daily return)
    spy_ret = spy.pct_change().reindex(ret.index)
    beta = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for col in ret.columns:
        cov_ = ret[col].rolling(BETA_WINDOW).cov(spy_ret)
        var_ = spy_ret.rolling(BETA_WINDOW).var()
        beta[col] = cov_ / (var_ + 1e-9)

    # Sector one-hot
    sectors  = sorted(STOCK_UNIVERSE.keys())
    sec_dummies = {f"sec_{s}": [1.0 if SECTOR_MAP[t] == s else 0.0
                                for t in prices.columns]
                   for s in sectors}

    # Stack into long format  ──  index: (date, ticker)
    frames = []
    for ticker in tqdm(prices.columns, desc="Building node features", leave=False):
        df = pd.DataFrame({
            "ret_1d":    ret_1d[ticker],
            "ret_5d":    ret_5d[ticker],
            "ret_21d":   ret_21d[ticker],
            "vol_21d":   vol_21d[ticker],
            "mom_short": mom_short[ticker],
            "mom_long":  mom_long[ticker],
            "rsi_14":    rsi[ticker],
            "beta_63d":  beta[ticker],
        })
        for s in sectors:
            df[f"sec_{s}"] = 1.0 if SECTOR_MAP[ticker] == s else 0.0
        df["ticker"] = ticker
        df["sector"] = SECTOR_MAP[ticker]
        frames.append(df)

    features_long = pd.concat(frames)
    features_long.index.name = "date"
    features_long = features_long.reset_index()
    features_long["date"] = pd.to_datetime(features_long["date"])
    features_long = features_long.dropna()
    return features_long


# ──────────────────────────────────────────────────────────────────────────────
# 4. ADJACENCY MATRIX  (edge matrix A per snapshot date)
# ──────────────────────────────────────────────────────────────────────────────
def build_rolling_adjacency(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns dict  {date_str: adjacency_DataFrame}  for every snapshot date
    (monthly, end-of-month).  Each DataFrame is (50×50) with entries = corr
    magnitude where |corr| >= CORR_THRESHOLD, else 0.
    Edges weighted by absolute correlation so the GNN can use them directly.
    """
    ret = prices.pct_change().dropna()
    snapshot_dates = ret.resample("ME").last().index   # month-end dates

    adjacencies = {}
    for snap_date in tqdm(snapshot_dates, desc="Building adjacency snapshots"):
        window = ret.loc[:snap_date].tail(CORR_WINDOW)
        if len(window) < CORR_WINDOW // 2:
            continue
        corr = window.corr().abs()
        A = corr.where(corr >= CORR_THRESHOLD, other=0.0).copy()
        A_arr = A.to_numpy(copy=True)
        np.fill_diagonal(A_arr, 0.0)             # no self-loops
        A = pd.DataFrame(A_arr, index=A.index, columns=A.columns)
        adjacencies[snap_date.strftime("%Y-%m-%d")] = A
    return adjacencies


# ──────────────────────────────────────────────────────────────────────────────
# 5. REAL CRASH LABELS  (forward max-drawdown)
# ──────────────────────────────────────────────────────────────────────────────
def build_crash_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For every (ticker, date) compute max-drawdown over next LABEL_HORIZON days.
    Binary label:  1 if max-drawdown < CRASH_THRESHOLD, else 0.
    Also stores the raw drawdown value for regression tasks.
    """
    records = []
    dates   = prices.index

    for i, date in enumerate(tqdm(dates, desc="Computing crash labels", leave=False)):
        end_idx = min(i + LABEL_HORIZON, len(dates) - 1)
        if end_idx == i:
            continue
        future = prices.iloc[i:end_idx + 1]
        rolling_max = future.cummax()
        drawdowns = (future - rolling_max) / rolling_max
        max_dd = drawdowns.min()         # most negative value per ticker

        for ticker in prices.columns:
            records.append({
                "date":       date,
                "ticker":     ticker,
                "max_dd":     float(max_dd[ticker]),
                "crash_label": int(max_dd[ticker] <= CRASH_THRESHOLD),
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6. MASTER BUILD  (called by all downstream modules)
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset(force_rebuild: bool = False):
    """
    Build or load the full dataset.
    Returns:
        prices          : DataFrame (date × ticker) of adjusted close prices
        features_long   : long-format node features (date, ticker, feat…)
        adjacencies     : dict{date_str → 50×50 adjacency DataFrame}
        crash_labels    : DataFrame (date, ticker, max_dd, crash_label)
        spy             : SPY close-price Series (benchmark)
    """
    feat_cache   = os.path.join(CACHE_DIR, "features_long.parquet")
    label_cache  = os.path.join(CACHE_DIR, "crash_labels.parquet")
    adj_cache    = os.path.join(CACHE_DIR, "adjacencies.parquet")

    # Download prices + SPY
    prices = load_prices()
    spy_raw = yf.download("SPY", start=DATA_START, end=DATA_END,
                          auto_adjust=True, progress=False)["Close"]
    spy = spy_raw.squeeze().reindex(prices.index).ffill().bfill()

    if not force_rebuild and all(os.path.exists(p)
                                  for p in [feat_cache, label_cache, adj_cache]):
        print("[data_loader] Loading processed features/labels from cache …")
        features_long = pd.read_parquet(feat_cache)
        crash_labels  = pd.read_parquet(label_cache)
        adj_flat      = pd.read_parquet(adj_cache)
        # Reconstruct adjacency dict
        adjacencies = {}
        for snap_date, grp in adj_flat.groupby("snap_date"):
            A = grp.drop(columns="snap_date").set_index("ticker_row")
            A.columns.name = None
            adjacencies[snap_date] = A
    else:
        print("[data_loader] Rebuilding processed data (this takes ~2-3 min) …")
        features_long = build_features(prices, spy)
        crash_labels  = build_crash_labels(prices)
        adjacencies   = build_rolling_adjacency(prices)

        # Cache adjacency as a flat parquet
        adj_rows = []
        for snap_date, A in adjacencies.items():
            flat = A.copy()
            flat.index.name = "ticker_row"
            flat = flat.reset_index()
            flat["snap_date"] = snap_date
            adj_rows.append(flat)
        adj_flat = pd.concat(adj_rows, ignore_index=True)

        features_long.to_parquet(feat_cache)
        crash_labels.to_parquet(label_cache)
        adj_flat.to_parquet(adj_cache)

        # Also save human-readable CSVs (backward compatibility)
        features_long.to_csv("node_features_X.csv", index=False)
        crash_labels.to_csv("crash_labels.csv", index=False)
        prices.pct_change().dropna().to_csv("daily_returns.csv")

        print("[data_loader] Dataset built and cached.")

    return prices, features_long, adjacencies, crash_labels, spy


# ──────────────────────────────────────────────────────────────────────────────
# 7. SNAPSHOT UTILITY  (used by backtest & GNN training)
# ──────────────────────────────────────────────────────────────────────────────
NUMERIC_FEATS = [
    "ret_1d", "ret_5d", "ret_21d",
    "vol_21d", "mom_short", "mom_long",
    "rsi_14", "beta_63d",
    "sec_Consumer", "sec_Energy", "sec_Finance",
    "sec_Healthcare", "sec_Tech",
]

def get_snapshot(features_long: pd.DataFrame,
                 adjacencies: dict,
                 crash_labels: pd.DataFrame,
                 as_of_date: str,
                 scaler: StandardScaler | None = None):
    """
    Extracts a single graph snapshot for `as_of_date`.

    Returns:
        X_norm    : (N, F) numpy array — normalised node features
        A_df      : (N, N) adjacency DataFrame for this snapshot
        y         : (N,) numpy array — crash labels
        tickers   : list of N ticker strings (consistent ordering)
        scaler    : fitted scaler (pass back in for test-time consistency)
    """
    # Nearest adjacency snapshot at or before as_of_date
    snap_dates = sorted(adjacencies.keys())
    snap_ts = pd.Timestamp(as_of_date)
    valid_snaps = [d for d in snap_dates if pd.Timestamp(d) <= snap_ts]
    if not valid_snaps:
        raise ValueError(f"No adjacency snapshot available before {as_of_date}")
    snap_key = valid_snaps[-1]
    A_df = adjacencies[snap_key]
    tickers = list(A_df.columns)

    feat_on_date = features_long.copy()
    feat_on_date["date"] = pd.to_datetime(feat_on_date["date"])
    snap_ts = pd.Timestamp(as_of_date)
    feat_on_date = feat_on_date[feat_on_date["date"] <= snap_ts]
    feat_latest  = feat_on_date.sort_values("date").groupby("ticker").last()
    feat_latest  = feat_latest.reindex(tickers)  # align to adjacency order

    X_raw = feat_latest[NUMERIC_FEATS].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    if scaler is None:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_raw).astype(np.float32)
    else:
        X_norm = scaler.transform(X_raw).astype(np.float32)

    # Labels: latest label on or before as_of_date per ticker
    lab_on_date = crash_labels.copy()
    lab_on_date["date"] = pd.to_datetime(lab_on_date["date"])
    lab_on_date = lab_on_date[lab_on_date["date"] <= snap_ts]
    lab_latest  = lab_on_date.sort_values("date").groupby("ticker").last()
    lab_latest  = lab_latest.reindex(tickers)
    y = lab_latest["crash_label"].fillna(0).values.astype(np.float32)

    return X_norm, A_df, y, tickers, scaler


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prices, features_long, adjacencies, crash_labels, spy = build_dataset()
    print(f"\n✓ Prices:        {prices.shape}")
    print(f"✓ Features:      {features_long.shape}")
    print(f"✓ Adj snapshots: {len(adjacencies)}")
    print(f"✓ Crash labels:  {crash_labels.shape}")
    crash_rate = crash_labels["crash_label"].mean()
    print(f"✓ Overall crash rate: {crash_rate:.1%}")
    print("\nTop 5 crash-prone tickers:")
    print(crash_labels.groupby("ticker")["crash_label"].mean()
          .sort_values(ascending=False).head())
