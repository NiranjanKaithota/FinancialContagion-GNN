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

Each firm is represented by:

* Normalized Volatility
* Daily Returns
* 200-Day Price Momentum

#### Edges ($A$)

Connections between firms are created using:

* 1-Year Rolling Pearson Correlations
* Correlation Threshold > 0.60

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

The GNN generates risk probabilities, but investment decisions require portfolio weights.

The autonomous agent converts risk predictions into actionable capital allocation decisions.

### Signal Translation

The agent transforms contagion probabilities into investor return expectations:

| GNN Risk Score | Investor View        |
| -------------- | -------------------- |
| 0.00           | +10% Expected Return |
| 0.50           | Neutral              |
| 0.99           | -20% Expected Return |

These views become inputs to the Black-Litterman framework.

### Bayesian Updating

Using **PyPortfolioOpt**, the model combines:

* Historical market covariance structure
* AI-generated absolute views

to produce posterior expected returns.

### Defensive Optimization

The optimized portfolio is generated using:

* Markowitz Efficient Frontier
* Minimum Variance Optimization
* Maximum Sharpe Ratio Optimization

The system naturally:

* Reduces exposure to highly infected assets
* Allocates capital toward defensive sectors
* Concentrates capital in structurally isolated safe havens

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

For every stock, it extracts three normalized metrics:

- Daily Return
- 30-day Volatility
- 200-day Momentum

### Edge Adjacency Matrix ($A$)

It calculates a Pearson correlation matrix of daily returns.

If the correlation between two stocks is:

$$
\rho_{ij} \geq 0.60
$$

a structural link (**edge**) is created between them, mapping how shocks can propagate through the market network.

### Target Creation ($y$)

The pipeline looks ahead into historical data (with emphasis on the 2020 COVID market crash) and assigns labels:

- **1.0** → Stock crashed by **≥ 20%**
- **0.0** → Stock remained relatively safe

---

## 📂 `gnn_model.py` — The Contagion Predictor

This is the core intelligence layer of the system.

Unlike traditional machine learning models that treat stocks independently, this module uses a **Hierarchical Graph Convolutional Network (GCN)** to model interconnected market risk.

### Micro Level

Simulates how distress in one stock (e.g., Apple) propagates through correlation-based connections to neighboring stocks.

### Macro Level

Aggregates company-level signals into sector-level representations to estimate:

- Technology Sector Health
- Finance Sector Health
- Energy Sector Health
- Healthcare Sector Health
- Consumer Sector Health

### The Signal

The model outputs a **Contagion Risk Probability** for every stock:

$$
\text{Contagion Risk} \in [0,1]
$$

where:

- **0.0** = Low contagion vulnerability
- **1.0** = High likelihood of being pulled into a market-wide collapse

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
