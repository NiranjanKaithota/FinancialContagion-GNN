# Financial Contagion Graph Neural Network (GNN) & Autonomous Portfolio Agent

## 📌 Project Overview

This project presents a dual-system quantitative finance architecture designed to predict systemic market crashes and automatically reallocate capital toward defensive safe-haven assets.

The framework bridges **Deep Graph Learning** and **Modern Portfolio Theory** by combining:

* A **Hierarchical Graph Neural Network (GNN)** that models financial contagion across interconnected firms and sectors.
* An **Autonomous Portfolio Agent** powered by **Black-Litterman optimization** that dynamically adjusts portfolio allocations based on AI-generated systemic risk signals.

---

# ⚙️ Dual-System Architecture

## 1. The Contagion Predictor (Hierarchical GNN)

Traditional financial models often analyze assets independently. This model instead treats the market as an interconnected network where financial distress can propagate through highly correlated assets.

### Network Structure

#### Nodes (Firms)

* 50 highly liquid equities
* Distributed across 5 macro sectors:

  * Technology
  * Finance
  * Energy
  * Healthcare
  * Consumer

#### Node Features ($X$)

Each firm is represented by **13 standardized features**:

* Immediate returns momentum: **1-day**, **5-day**, and **21-day** returns
* Stress indicator: **21-day annualized rolling volatility**
* SMA momentum: short momentum (**21-day SMA relative difference**) and long momentum (**63-day SMA relative difference**)
* Technical indicator: **14-day Relative Strength Index (RSI)**
* Market sensitivity: **63-day rolling systematic market beta vs. SPY**
* Sector membership: **5 sector one-hot columns**

#### Edges ($A$)

Connections between firms are created using:

* **1-Year Rolling Pearson Correlations** (252 trading days)
* **Correlation Threshold $\geq$ 0.45** (absolute correlation value)

Only statistically significant relationships are retained within the adjacency matrix.

### Hierarchical Message Passing

The PyTorch Geometric architecture follows a **Micro → Macro → Micro** information flow:

1. Firm-level embeddings exchange information through graph message passing.
2. Firm representations are pooled into sector-level "Super Nodes."
3. Sector health embeddings are learned and propagated back to individual firms.
4. The network outputs a final:

**Contagion Risk Probability**

[
P_risk \in [0,1]
]

where:

* 0.00 = Structurally healthy
* 1.00 = High systemic contagion risk

---

## 2. The Autonomous Financial Agent (Black-Litterman Optimizer)

The GNN generates contagion probabilities, but portfolio execution requires optimal allocation weights. The autonomous agent translates risk metrics into real portfolio weights:

### 2A. GNN-BL Expected Return Views
We map the GNN contagion score $s_i \in [0, 1]$ into expected annual returns using a calibrated linear translation function:
$$E[R_i] = 12\% - 38\% \times s_i$$
Expected views are bounded between $[-25\%, +20\%]$. The uncertainty matrix $\Omega$ is scaled dynamically: assets with higher contagion risk are given tighter uncertainty bounds, representing the model's higher confidence in its defensive view of crash-bound firms.

### 2B. Network Centrality Risk Penalty
Assets that are highly central within the correlation network (shock amplifiers) receive an additional penalty. We compute **betweenness centrality** on the active correlation network:
$$\text{Composite Risk}_i = s_i + 0.10 \times \text{Centrality}_i$$
$$E[R_i]_{\text{adjusted}} = E[R_i] - 0.05 \times \text{Composite Risk}_i$$

### 2C. Asymmetric Systemic Circuit Breaker (Cash Rotation)
To protect capital during generalized market crashes, the agent runs an emergency circuit breaker:
- If the **average GNN contagion score across all assets > 0.45**, the portfolio rotates **100% of capital into Cash**.
- Cash holdings grow at the daily risk-free rate of **4% annualized** ($0.04 / 252$ daily).

### 2D. Dynamic Transaction Cost Modeling
We incorporate realistic rebalancing friction by charging a transaction fee of **10 bps (0.10%)** of rebalanced turnover:
$$\text{Turnover}_t = \sum_{i} |w_{i, t} - w'_{i, t-1}|$$
where $w'_{i, t-1}$ is the asset weight adjusted for price drift at the end of the prior month. Dynamic transaction costs are applied to both the GNN Agent and the Equal Weight baseline portfolios.

### 2E. Defensive Optimization
Blended Black-Litterman expected returns and a Ledoit-Wolf shrinkage covariance matrix are processed via an Efficient Frontier optimizer targeting **Maximum Sharpe Ratio** with long-only constraints ($w_i \in [0\%, 20\%]$). If optimization fails during high stress, it falls back to a Min-Volatility optimizer, and finally to inverse-risk weighting.

---

# 🛠️ Technology Stack

## Data Engineering

* yfinance
* pandas
* numpy
* scikit-learn

## Deep Learning

* PyTorch
* PyTorch Geometric (PyG)

## Quantitative Finance

* PyPortfolioOpt
* CVXPY

---

# System Architecture Breakdown

## 📂 `data_loader.py` — The Pipeline & Graph Builder

This script downloads, cleans, and restructures financial market data into a mathematical graph format that an AI can understand.

### The Universe
Tracks **50 stocks** split evenly across **5 macro-sectors**:

- Technology
- Finance
- Energy
- Healthcare
- Consumer

### Node Features Matrix ($X$)

For every stock, it extracts **13 normalized metrics**:
- Returns momentum at 1-day, 5-day, and 21-day windows
- 21-day annualized rolling volatility
- Short-term (21-day SMA) and long-term (63-day SMA) relative momentum
- 14-day Relative Strength Index (RSI)
- 63-day rolling market beta vs. SPY
- 5 sector membership one-hot columns

### Edge Adjacency Matrix ($A$)

It calculates a monthly Pearson correlation matrix of daily returns. If the absolute correlation between two stocks is:
$$|\rho_{ij}| \geq 0.45$$
a structural link (**edge**) is created between them, with the edge weight equal to the correlation magnitude, mapping how shocks propagate through the market network.

### Target Creation ($y$)

The pipeline looks ahead into historical data and assigns labels:
- **1.0** (Crash) → Stock experienced a maximum drawdown of **$\leq -20\%$** in the next **63 trading days (approx. 3 months)**.
- **0.0** (Safe) → Stock remained relatively safe (drawdown $> -20\%$).

---

## 📂 `gnn_model.py` — The Contagion Predictor

This is the core intelligence layer of the system.

Unlike traditional machine learning models that treat stocks independently, this module uses a **Hierarchical Temporal Contagion Graph Attention Network (HTC-GNN)** to model interconnected market risk.

### Micro Level (Graph Attention Layers)
Uses two Multi-head Graph Attention Network (GAT) layers. In GAT, the model dynamically learns attention weights $\alpha_{ij}$ representing "how much firm $i$ should listen to firm $j$". This mapping represents contagion channels.

### Macro Level (Sector Super-node Pooling)
Firms are pooled by sector to form "sector super-nodes". A 2-layer sector MLP refines this macro representation to evaluate:
- Technology Sector Health
- Finance Sector Health
- Energy Sector Health
- Healthcare Sector Health
- Consumer Sector Health

### Cross-Level Attention & Temporal GRU Cell
- **Cross-Level Attention**: Individual firms query their sector's super-node, learning how much sector-level distress impacts them.
- **GRU Cell**: A Gated Recurrent Unit cell processes the fused embedding sequentially month-to-month, allowing historical contagion memory to accumulate across snapshots.

### The Signal
The model outputs a calibrated **Contagion Risk Probability** for each stock:
$$\text{Contagion Risk} \in [0,1]$$
where:
- **0.0** = Low contagion vulnerability ( defensive stock )
- **1.0** = High likelihood of propagating or being pulled into systemic collapse.

---

## 📂 `portfolio_agent.py` — The Defensive Risk Manager

This module acts as the institutional portfolio manager.

Its purpose is to convert AI-generated contagion risk scores into actual portfolio allocations.

### Market Prior

Establishes baseline expected returns using:

- Market capitalization weights
- Investor risk aversion assumptions

### Translating AI Signals into Finance

Expected returns are adjusted according to GNN-predicted contagion risk:

$$
\text{Expected Return}
=
10\% - (\text{Contagion Risk} \times 30\%)
$$

Examples:

| Contagion Risk | Adjusted Expected Return |
|---------------|-------------------------|
| 0.0 | 10% |
| 0.5 | -5% |
| 1.0 | -20% |

### Black–Litterman Optimization

Combines:

- Historical market equilibrium returns (priors)
- Forward-looking GNN risk forecasts (views)

to produce more robust expected return estimates.

### Efficient Frontier Optimization

Runs a **Maximum Sharpe Ratio** optimization to determine the mathematically optimal portfolio weights.

---

## 📂 `backtest_loop.py` — The Simulation Sandbox

This module provides the historical simulation framework used to evaluate the strategy.

### Walk-Forward Execution

The simulation advances strictly:

- Month-by-month
- In chronological order
- Without future information leakage

### Look-Ahead Bias Prevention

At each simulated date, the following modules only receive information available up to that point in time:

- `data_loader.py`
- `gnn_model.py`
- `portfolio_agent.py`

This ensures all performance results are realistic and free from look-ahead bias.

---

# Operational Workflow

When all modules are combined into a production pipeline, the system executes the following monthly cycle:

```text
[1. data_loader]
        │
        └── Extracts market prices and builds graph structure
            (Nodes + Correlation Edges)
        │
        ▼
[2. gnn_model]
        │
        └── Analyzes graph and estimates contagion risk
        │
        ▼
[3. portfolio_agent]
        │
        └── Converts risk scores into optimized portfolio weights
        │
        ▼
[4. backtest_loop]
        │
        └── Simulates portfolio performance
            Advances time by one month
            Repeats the entire process
```

## Monthly Decision Flywheel

```text
Market Data
     │
     ▼
Graph Construction
     │
     ▼
GNN Contagion Prediction
     │
     ▼
Black-Litterman Portfolio Optimization
     │
     ▼
Portfolio Allocation
     │
     ▼
Performance Evaluation
     │
     ▼
Advance Time (+1 Month)
     │
     └── Repeat
```

This creates a fully end-to-end framework that:

1. Learns structural market relationships through graph networks.
2. Predicts crash contagion risk before market stress events.
3. Adjusts expected returns based on AI-derived risk forecasts.
4. Constructs optimal portfolios using Black–Litterman optimization.
5. Validates performance through realistic walk-forward backtesting.

---

# 🚀 Execution Pipeline

Run the full simulation using a Python 3.10/3.11 virtual environment.
### 0. Environment Setup (Python 3.11)
Before running the pipeline, you must create and activate an isolated virtual environment to ensure the C++ and CUDA background libraries compile correctly for PyTorch Geometric.

**For Windows:**
```bash
# Create the environment using Python 3.11
py -3.11 -m venv gnn_env

# Activate the environment
gnn_env\Scripts\activate

pip install -r requirements.txt
```

## Step 1 — Data Engineering

```bash
python data_loader.py
```

Responsibilities:

* Download historical market data
* Generate maximum drawdown labels (`y`)
* Construct feature matrix (`X`)
* Construct adjacency matrix (`A`)
* Export processed datasets

---

## Step 2 — Train Hierarchical GNN

```bash
python gnn_model.py
```

Responsibilities:

* Load graph datasets
* Train the Hierarchical GNN
* Predict systemic contagion probabilities
* Save trained model weights

---

## Step 3 — Portfolio Optimization Backtest

```bash
python backtest_loop.py
```

or

```bash
python portfolio_agent.py
```

Responsibilities:

* Load pretrained GNN
* Walk forward through time
* Generate dynamic Black-Litterman views
* Compute optimal portfolio allocations
* Simulate monthly rebalancing

---

## Step 4 — Network Visualization

```bash
python visualize_network.py
```
(Need to change to better visualization)
* Generate publication-quality network topology graphs
* Visualize contagion pathways
* Color-code firms by systemic risk score

---

# 📊 Core Research Objective

The primary objective is to answer:

> Can a Hierarchical Graph Neural Network identify structural financial contagion early enough to improve portfolio allocation decisions and reduce downside risk during market crises?

The framework seeks to quantify whether graph-based systemic risk signals can generate superior risk-adjusted performance compared to traditional portfolio construction methods.

---

# 🔮 Future Enhancements

## 1. Dynamic Transaction Cost Modeling

Integrate realistic execution frictions into the walk-forward backtest:

* Institutional brokerage fees
* Slippage estimates
* Portfolio turnover penalties

Target:

* 5–10 bps transaction cost modeling

This will help evaluate whether generated alpha remains significant after implementation costs.

---

## 2. Asymmetric Systemic Circuit Breaker

Introduce a portfolio-level emergency defense mechanism.

### Proposed Logic

If:

```text
Average GNN Contagion Score > 0.75
```

Then:

```text
100% Equity Exposure → 0%
100% Capital → Cash / Short-Term Treasuries
```

This creates an AI-driven risk-off regime during periods of systemic panic.

---

## 3. Formalized Performance Metrics Export

Automate benchmark comparison and reporting.

### Metrics

* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Annualized Return
* Volatility
* Calmar Ratio

### Benchmark

* S&P 500 Buy-and-Hold

### Export Format

```json
{
  "sharpe": 1.42,
  "sortino": 2.11,
  "max_drawdown": -0.13,
  "benchmark_sharpe": 0.91
}
```

This enables direct integration with academic papers, dashboards, and research reports.

---

# 📈 Expected Contributions

This project demonstrates how:

* Graph Neural Networks can model systemic financial contagion.
* Hierarchical sector structures improve market risk representation.
* Black-Litterman optimization can convert AI signals into actionable portfolio allocations.
* Autonomous agents can dynamically defend capital during periods of elevated systemic stress.

The resulting framework serves as a foundation for next-generation AI-driven portfolio management and systemic risk forecasting systems.
