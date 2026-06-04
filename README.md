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
