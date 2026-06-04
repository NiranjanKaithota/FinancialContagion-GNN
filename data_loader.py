import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1. DEFINE THE EXPANDED UNIVERSE (50 STOCKS)
# =========================================================
stock_universe = {
    'Tech': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'CSCO', 'INTC'],
    'Finance': ['JPM', 'BAC', 'WFC', 'MS', 'GS', 'C', 'BLK', 'SPGI', 'AXP', 'V'],
    # FIX: Swapped delisted PXD for HAL (Halliburton)
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'HAL', 'VLO', 'OXY', 'PSX'], 
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'MRK', 'ABBV', 'PFE', 'TMO', 'DHR', 'ABT', 'BMY'],
    'Consumer': ['AMZN', 'WMT', 'PG', 'HD', 'COST', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX']
}

tickers = [ticker for sector, symbols in stock_universe.items() for ticker in symbols]

# Flatten the dictionary to get just the list of tickers
tickers = [ticker for sector, symbols in stock_universe.items() for ticker in symbols]

# =========================================================
# 2. PULL HISTORICAL DATA
# =========================================================
print("Downloading historical data...")

# We explicitly set auto_adjust=True and pull 'Close'. 
# We also set threads=False to force a stable, sequential download.
data = yf.download(
    tickers, 
    start="2018-01-01", 
    end="2024-01-01", 
    auto_adjust=True, 
    threads=False
)['Close']

# Clean the data: Forward fill (carry previous day's price forward) 
# for days the market was closed or data is missing, then backward fill.
data = data.ffill().bfill()
print(f"Data downloaded. Shape: {data.shape}")

# =========================================================
# 3. GENERATE NODE FEATURES MATRIX (X)
# =========================================================
print("\nGenerating Node Features Matrix (X)...")

# A. Calculate Daily Returns
daily_returns = data.pct_change().dropna()

# B. Calculate Rolling Volatility (30-day window)
rolling_volatility = daily_returns.rolling(window=30).std().dropna()

# C. Calculate Momentum (Current price / 200-day moving average)
ma_200 = data.rolling(window=200).mean().dropna()
aligned_data = data.loc[ma_200.index]
momentum = aligned_data / ma_200

# D. Extract the most recent data point for each feature 
latest_returns = daily_returns.iloc[-1]
latest_volatility = rolling_volatility.iloc[-1]
latest_momentum = momentum.iloc[-1]

# Combine into a single DataFrame
node_features_df = pd.DataFrame({
    'Return': latest_returns,
    'Volatility': latest_volatility,
    'Momentum': latest_momentum
})

# E. Normalize the features (Crucial for Neural Networks)
scaler = StandardScaler()
normalized_features = scaler.fit_transform(node_features_df)

# Convert back to a readable DataFrame
X_matrix = pd.DataFrame(
    normalized_features, 
    index=node_features_df.index, 
    columns=node_features_df.columns
)

# Add the categorical Sector column back in
sector_mapping = {ticker: sector for sector, tickers in stock_universe.items() for ticker in tickers}
X_matrix['Sector'] = X_matrix.index.map(sector_mapping)

print("Node Matrix [X] complete. Preview:")
print(X_matrix.head())

# =========================================================
# 4. GENERATE EDGE ADJACENCY MATRIX (A)
# =========================================================
print("\nGenerating Edge Adjacency Matrix (A)...")

# A. Calculate the Pearson Correlation matrix of daily returns
correlation_matrix = daily_returns.corr()

# B. Apply the Threshold Filter
THRESHOLD = 0.60

# Create a copy to act as our Adjacency Matrix
A_matrix = correlation_matrix.copy()

# If the correlation is below the threshold, set the connection to 0 (No edge)
A_matrix[A_matrix < THRESHOLD] = 0.0

# Set the diagonal to 0 (A stock cannot infect itself in graph theory)
# Set the diagonal to 0 (A stock cannot infect itself in graph theory)
# We safely update the values by replacing the dataframe with a fresh, writable numpy array
A_array = A_matrix.to_numpy(copy=True)
np.fill_diagonal(A_array, 0.0)
A_matrix.iloc[:, :] = A_array

print(f"Edge Matrix [A] complete using threshold {THRESHOLD}. Preview:")
print(A_matrix.head())

# Optional: Count how many active edges survived the threshold
total_possible_edges = (len(tickers) * (len(tickers) - 1))
active_edges = (A_matrix > 0).sum().sum()
print(f"\nNetwork Density: {active_edges} active connections out of {total_possible_edges} possible.")

# =========================================================
# 4.5 GENERATE REAL-WORLD TARGETS (y)
# =========================================================
print("\nCalculating Real-World Target Variables (Max Drawdown)...")

# Define our timeline
# Let's pretend today is Jan 1, 2020. We want to predict the COVID crash (Feb-March 2020)
feature_end_date = "2020-01-01"
target_end_date = "2020-03-30" # 90 days into the future to capture the crash

# 1. Isolate the future price data
future_data = data.loc[feature_end_date:target_end_date]

# 2. Calculate Maximum Drawdown for every stock in that future window
rolling_max = future_data.cummax()
drawdowns = (future_data - rolling_max) / rolling_max
max_drawdowns = drawdowns.min() # The absolute lowest point (e.g., -0.35)

# 3. Apply the threshold to create binary labels (Crash = 1, Safe = 0)
CRASH_THRESHOLD = -0.20 # A 20% drop is a bear market/crash
y_labels = (max_drawdowns <= CRASH_THRESHOLD).astype(float)

# Save this to our node features dataframe
X_matrix['Target_Crash'] = y_labels.values

print("Real targets calculated! Preview of future crashes:")
print(X_matrix[['Sector', 'Target_Crash']].head(10))

# =========================================================
# 5. EXPORT FOR TEAM HAND-OFF
# =========================================================
print("\nExporting data to CSV files...")
X_matrix.to_csv('node_features_X.csv')
A_matrix.to_csv('edge_matrix_A.csv')
daily_returns.to_csv('daily_returns.csv')
print("Data exported successfully! You are ready for the hand-off.")