import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pypfopt import risk_models, expected_returns, BlackLittermanModel, EfficientFrontier

# =========================================================
# 1. SYNTHETIC CRISIS DATA GENERATOR
# =========================================================
def generate_synthetic_crisis_data(start_date="2020-01-01", days=756):
    """
    Generates 3 years of daily returns for 50 assets across 5 sectors.
    Injects a geopolitical systemic shock between months 18 and 24.
    """
    print("--- Generating Synthetic Market Data & Crisis Regimes ---")
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=days, freq="B")
    
    sectors = {
        'Tech': [f'TECH_{i}' for i in range(10)],
        'Finance': [f'FIN_{i}' for i in range(10)],
        'Energy': [f'ENG_{i}' for i in range(10)],
        'Healthcare': [f'HLTH_{i}' for i in range(10)],
        'Consumer': [f'CONS_{i}' for i in range(10)]
    }
    
    tickers = [t for sec in sectors.values() for t in sec]
    sector_map = {t: sec for sec, tick_list in sectors.items() for t in tick_list}
    
    # Baseline: Low correlation (~0.20), Normal Volatility
    base_returns = np.random.normal(0.0005, 0.015, size=(days, 50))
    df_returns = pd.DataFrame(base_returns, index=dates, columns=tickers)
    
    # Crisis Window (Approx. Month 18 to 24)
    crisis_start_idx = 18 * 21
    crisis_end_idx = 24 * 21
    
    # Systemic Shock Profile
    market_shock_factor = np.random.normal(-0.02, 0.04, size=(crisis_end_idx - crisis_start_idx, 1))
    
    for i, ticker in enumerate(tickers):
        sector = sector_map[ticker]
        if sector in ['Tech', 'Finance']:
            # High integration, severe drawdown, 3x-4x volatility
            df_returns.iloc[crisis_start_idx:crisis_end_idx, i] += market_shock_factor[:, 0] * 1.5 
            df_returns.iloc[crisis_start_idx:crisis_end_idx, i] += np.random.normal(0, 0.06, size=(crisis_end_idx - crisis_start_idx))
        else:
            # Defensive safe-havens: Flat/Positive drift, decoupled
            df_returns.iloc[crisis_start_idx:crisis_end_idx, i] += np.random.normal(0.001, 0.02, size=(crisis_end_idx - crisis_start_idx))
            
    # Convert returns to price history
    df_prices = (1 + df_returns).cumprod() * 100
    return df_prices, sectors

# =========================================================
# 2. DATA PIPELINE & GRAPH BUILDER
# =========================================================
def build_graph_features(price_slice, sector_dict):
    """
    Constructs normalized features (X), Adjacency matrix (A), and Target labels (y).
    """
    returns = price_slice.pct_change().dropna()
    
    # Features: Mean Return, Volatility, 20-day Momentum
    mean_ret = returns.mean()
    volatility = returns.std()
    momentum = (price_slice.iloc[-1] / price_slice.iloc[-20]) - 1 if len(price_slice) > 20 else mean_ret * 0
    
    features_df = pd.DataFrame({'Return': mean_ret, 'Volatility': volatility, 'Momentum': momentum})
    features_normalized = (features_df - features_df.mean()) / (features_df.std() + 1e-8)
    x_tensor = torch.tensor(features_normalized.values, dtype=torch.float)
    
    # Adjacency (Correlation > 0.55)
    corr_matrix = returns.corr().fillna(0)
    A_matrix = corr_matrix.values.copy()
    np.fill_diagonal(A_matrix, 0)
    A_matrix[A_matrix < 0.55] = 0.0
    
    A_tensor = torch.tensor(A_matrix, dtype=torch.float)
    edge_index = (A_tensor > 0).nonzero(as_tuple=False).t()
    edge_weight = A_tensor[edge_index[0], edge_index[1]]
    
    # Target: Localized proxy drawdown over last 60 days
    recent_prices = price_slice.tail(60)
    drawdowns = (recent_prices.min() - recent_prices.max()) / recent_prices.max()
    y_labels = (drawdowns < -0.15).astype(float) # 1.0 if crashed
    y_tensor = torch.tensor(y_labels.values, dtype=torch.float).view(-1, 1)
    
    # Sector indices
    tickers = price_slice.columns
    sector_map = {t: sec for sec, tick_list in sector_dict.items() for t in tick_list}
    unique_sectors = list(sector_dict.keys())
    sector_to_idx = {sec: idx for idx, sec in enumerate(unique_sectors)}
    sector_idx_tensor = torch.tensor([sector_to_idx[sector_map[t]] for t in tickers], dtype=torch.long)
    
    graph_data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_weight, y=y_tensor)
    return graph_data, sector_idx_tensor, len(unique_sectors), tickers

# =========================================================
# 3. STATE-AWARE GNN ARCHITECTURE
# =========================================================
class HierarchicalContagionGNN(nn.Module):
    def __init__(self, num_features=3, hidden=16):
        super(HierarchicalContagionGNN, self).__init__()
        self.micro_conv = GCNConv(num_features, hidden)
        self.macro_combine = nn.Linear(hidden * 2, hidden)
        self.predictor = nn.Linear(hidden, 1)

    def forward(self, data, sector_idx, total_sectors):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # Micro Level
        firm_embeds = F.relu(self.micro_conv(x, edge_index, edge_weight=edge_weight))
        
        # Macro Level (Sector Pooling)
        sector_embeds_list = []
        for s in range(total_sectors):
            mask = (sector_idx == s)
            sector_mean = firm_embeds[mask].mean(dim=0) if mask.any() else torch.zeros(firm_embeds.size(1))
            sector_embeds_list.append(sector_mean)
        
        sector_embeds = torch.stack(sector_embeds_list)
        broadcasted_macro = sector_embeds[sector_idx]
        
        # Combine & Predict
        combined = torch.cat([firm_embeds, broadcasted_macro], dim=1)
        hidden_out = F.relu(self.macro_combine(combined))
        return torch.sigmoid(self.predictor(hidden_out))

# =========================================================
# 4. WFFT TRAINING ENGINE
# =========================================================
def wfft_train_and_predict(graph_data, sector_idx, num_sectors, model_path="active_agent.pt"):
    model = HierarchicalContagionGNN()
    
    # WFFT Checkpoint Logic
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        lr, epochs = 0.001, 15  # Warm-start fine-tuning
        print("   -> State Found: Warm-starting WFFT (LR=0.001, Epochs=15)")
    else:
        lr, epochs = 0.01, 100  # Cold-start baseline
        print("   -> No State: Cold-starting optimization (LR=0.01, Epochs=100)")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data, sector_idx, num_sectors)
        loss = criterion(out, graph_data.y)
        loss.backward()
        optimizer.step()
        
    # Persist the mutated state
    torch.save(model.state_dict(), model_path)
    
    model.eval()
    with torch.no_grad():
        predictions = model(graph_data, sector_idx, num_sectors).numpy().flatten()
    return predictions

# =========================================================
# 5. BLACK-LITTERMAN PORTFOLIO AGENT
# =========================================================
def optimize_portfolio_bl(price_slice, contagion_probs, tickers):
    """
    Transforms GNN contagion probabilities into Black-Litterman allocations.
    """
    # Baseline Covariance and Priors
    S = risk_models.sample_cov(price_slice)
    # Equal-weighted proxy market cap
    mcaps = {t: 1.0 for t in tickers} 
    
    delta = BlackLittermanModel.market_implied_risk_aversion(price_slice)
    market_prior = BlackLittermanModel.market_implied_prior_returns(mcaps, delta, S)
    
    # Absolute Views Translation: Expected Return View = 0.12 - (P * 0.40)
    views_dict = {ticker: 0.12 - (prob * 0.40) for ticker, prob in zip(tickers, contagion_probs)}
    
    bl = BlackLittermanModel(S, pi=market_prior, absolute_views=views_dict)
    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()
    
    # Optimization Constraints
    ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=(0, 0.35))
    ef.add_objective(objective_functions.L2_reg, gamma=0.1) # Stabilization
    
    try:
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        return np.array([cleaned_weights[t] for t in tickers])
    except:
        # Fallback to equal weight if optimizer diverges during extreme crisis
        return np.ones(len(tickers)) / len(tickers)

# Helper for L2 Regularization
from pypfopt import objective_functions

# =========================================================
# 6. WALK-FORWARD EXECUTION LOOP
# =========================================================
def run_wfft_backtest():
    prices_df, sector_dict = generate_synthetic_crisis_data()
    tickers = prices_df.columns
    
    # Clean slate for WFFT
    if os.path.exists("active_agent.pt"):
        os.remove("active_agent.pt")
        
    start_date = prices_df.index[0] + relativedelta(months=6)
    end_date = prices_df.index[-2]
    current_date = start_date
    
    portfolio_value = 100000.0
    benchmark_value = 100000.0
    
    history_dates = []
    history_port = []
    history_bench = []
    
    print("\n=======================================================")
    print("INITIATING WALK-FORWARD FINE-TUNING (WFFT) BACKTEST")
    print("=======================================================\n")
    
    while current_date < end_date:
        print(f"[{current_date.strftime('%Y-%m')}] Processing Node...")
        
        # 1. Historical Slice (Strict zero look-ahead)
        hist_slice = prices_df.loc[:current_date]
        next_month = current_date + relativedelta(months=1)
        out_of_sample_slice = prices_df.loc[current_date:next_month]
        
        if len(out_of_sample_slice) < 2:
            break
            
        # 2. Build Graph & Targets
        graph_data, sector_idx, num_sec, _ = build_graph_features(hist_slice, sector_dict)
        
        # 3. WFFT GNN Inference
        contagion_probs = wfft_train_and_predict(graph_data, sector_idx, num_sec)
        
        # 4. Agent Reallocation
        weights = optimize_portfolio_bl(hist_slice, contagion_probs, tickers)
        
        # 5. Calculate Realized Out-of-Sample Returns
        period_returns = out_of_sample_slice.iloc[-1] / out_of_sample_slice.iloc[0] - 1
        
        # Evaluate performance
        port_return = np.dot(weights, period_returns.values)
        bench_return = np.dot(np.ones(50)/50, period_returns.values)
        
        portfolio_value *= (1 + port_return)
        benchmark_value *= (1 + bench_return)
        
        history_dates.append(current_date)
        history_port.append(portfolio_value)
        history_bench.append(benchmark_value)
        
        current_date = next_month
        
    print("\n=======================================================")
    print("BACKTEST COMPLETE")
    print(f"Final Agent Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Final Benchmark Value (Eq Wgt): ${benchmark_value:,.2f}")
    print("=======================================================")

if __name__ == "__main__":
    run_wfft_backtest()