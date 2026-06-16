"""
Layer 4: Black-Litterman Portfolio Engine
Uses GNN risk scores as "views" to adjust equilibrium return expectations.
High risk score → negative view on that asset → BL tilts portfolio away.
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize


# ── BL parameters ──────────────────────────────────────────────────────────────
TAU      = 0.05    # Uncertainty scaling of the prior (standard BL choice)
RISK_AVE = 2.5     # Market risk aversion coefficient (λ)


def compute_equilibrium_returns(cov_matrix, weights, risk_aversion=RISK_AVE):
    """
    Π = λ * Σ * w_mkt
    Reverse-engineer implied returns from market-cap weights.
    """
    return risk_aversion * cov_matrix @ weights


def gnn_scores_to_views(risk_scores, node_order, metadata, base_mu):
    """
    Convert GNN risk scores to BL return views.

    Formula: view_i = base_mu_i * (1 - 2 * risk_i)
    - risk = 0.0 → view = +base_mu  (no change)
    - risk = 0.5 → view = 0         (neutral)
    - risk = 1.0 → view = -base_mu  (full reversal)

    Returns:
      P  [K x N]: view selection matrix (one view per asset)
      Q  [K]:     view return vector
      Ω  [K x K]: view uncertainty (diagonal)
    """
    N = len(node_order)
    P = np.eye(N)          # Absolute views on each asset
    Q = np.zeros(N)
    omega_diag = np.zeros(N)

    for i, nid in enumerate(node_order):
        risk = float(risk_scores.get(nid, 0.5))
        mu   = float(base_mu[i])
        # View: how much we expect this asset to deviate from equilibrium
        Q[i] = mu * (1 - 2 * risk)
        # Uncertainty: high-risk assets get higher uncertainty in their view
        omega_diag[i] = TAU * (0.1 + risk * 0.4)

    Omega = np.diag(omega_diag)
    return P, Q, Omega


def black_litterman_posterior(pi, cov, P, Q, Omega, tau=TAU):
    """
    Standard Black-Litterman formula.
    
    Posterior mean:
      μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1 * Π + P'Ω^-1 * Q]
    
    Posterior covariance:
      M = [(τΣ)^-1 + P'Ω^-1P]^-1
    """
    tauSigma    = tau * cov
    tauSigma_inv = np.linalg.inv(tauSigma)
    Omega_inv   = np.linalg.inv(Omega)

    # Precision matrices
    A = tauSigma_inv + P.T @ Omega_inv @ P
    b = tauSigma_inv @ pi + P.T @ Omega_inv @ Q

    M       = np.linalg.inv(A)
    mu_bl   = M @ b

    return mu_bl, M


def mean_variance_optimize(mu, cov, risk_aversion=RISK_AVE, long_only=True):
    """
    Max Sharpe via mean-variance optimization.
    min  w'Σw - (1/λ) * w'μ
    s.t. Σw = 1, w ≥ 0
    """
    N = len(mu)
    w0 = np.ones(N) / N

    def neg_sharpe(w):
        port_ret = w @ mu
        port_var = w @ cov @ w
        return -(port_ret / (np.sqrt(port_var) + 1e-8))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 0.25)] * N if long_only else [(-0.1, 0.25)] * N  # max 25% per asset

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-9})

    return result.x if result.success else w0


def compute_portfolio_metrics(weights, returns_df, node_order):
    """Sharpe, max drawdown, annualized return, vol for a given weight vector."""
    port_ret = returns_df[node_order] @ weights
    ann_ret  = float(port_ret.mean() * 252)
    ann_vol  = float(port_ret.std() * np.sqrt(252))
    sharpe   = ann_ret / (ann_vol + 1e-8)

    cumulative = (1 + port_ret.fillna(0)).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd   = float(drawdown.min())

    return {
        "ann_return": round(ann_ret, 4),
        "ann_vol":    round(ann_vol, 4),
        "sharpe":     round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "cumulative_returns": cumulative.tolist(),
        "dates": [str(d.date()) for d in returns_df.index],
    }


def run_bl_optimization(risk_scores_dict, node_order, returns_df, metadata):
    """
    Full BL pipeline for a given set of GNN risk scores.
    
    Returns dict with:
      - equal_weight:   baseline allocation
      - bl_weights:     Black-Litterman allocation
      - metrics_equal:  performance of equal weight
      - metrics_bl:     performance of BL portfolio
      - views:          what the GNN is saying about each asset
    """
    # Filter to tickers present in both returns and node_order
    valid = [t for t in node_order if t in returns_df.columns]
    ret   = returns_df[valid].dropna()
    N     = len(valid)

    # Covariance matrix (annualized)
    cov = ret.cov().values * 252

    # Equal-weight baseline (market proxy)
    w_equal = np.ones(N) / N

    # Equilibrium returns
    pi = compute_equilibrium_returns(cov, w_equal)

    # GNN views
    base_mu = pi.copy()
    P, Q, Omega = gnn_scores_to_views(risk_scores_dict, valid, metadata, base_mu)

    # Black-Litterman posterior
    mu_bl, cov_bl = black_litterman_posterior(pi, cov, P, Q, Omega)

    # Optimize weights
    w_bl = mean_variance_optimize(mu_bl, cov + cov_bl)

    # Metrics
    m_equal = compute_portfolio_metrics(w_equal, ret, valid)
    m_bl    = compute_portfolio_metrics(w_bl,    ret, valid)

    # Build output: per-asset view summary
    views = []
    for i, nid in enumerate(valid):
        meta = metadata.get(nid, {})
        views.append({
            "ticker":      nid,
            "name":        meta.get("name", nid),
            "sector":      meta.get("sector", "Unknown"),
            "color":       meta.get("color", "#888780"),
            "risk_score":  round(float(risk_scores_dict.get(nid, 0.5)), 3),
            "eq_weight":   round(float(w_equal[i]), 4),
            "bl_weight":   round(float(w_bl[i]),    4),
            "pi_return":   round(float(pi[i]),      4),
            "bl_return":   round(float(mu_bl[i]),   4),
        })

    # Sector-level aggregation
    sector_weights = {}
    for v in views:
        s = v["sector"]
        sector_weights[s] = {
            "eq":   round(sector_weights.get(s, {}).get("eq", 0) + v["eq_weight"], 4),
            "bl":   round(sector_weights.get(s, {}).get("bl", 0) + v["bl_weight"], 4),
            "color": v["color"],
        }

    return {
        "views":            views,
        "sector_weights":   sector_weights,
        "metrics_equal":    m_equal,
        "metrics_bl":       m_bl,
        "improvement": {
            "sharpe_delta":  round(m_bl["sharpe"] - m_equal["sharpe"], 4),
            "drawdown_delta": round(m_bl["max_drawdown"] - m_equal["max_drawdown"], 4),
        }
    }


def run_backtests(snapshots, risk_scores_all, node_order, returns_df, metadata,
                  crisis_periods=None):
    """
    Rolling backtest: rebalance portfolio every snapshot using BL+GNN weights.
    Compare vs equal-weight buy-and-hold.
    """
    if crisis_periods is None:
        crisis_periods = {
            "2020_covid":    ("2020-01-01", "2020-12-31"),
            "2022_rate_hike":("2022-01-01", "2022-12-31"),
        }

    valid = [t for t in node_order if t in returns_df.columns]

    results = {}
    for period_name, (start, end) in crisis_periods.items():
        period_ret = returns_df[valid].loc[start:end].fillna(0)
        if len(period_ret) < 20:
            continue

        # Find risk scores closest to start of period
        dates_in_scores = [s["date"] for s in snapshots]
        start_idx = next((i for i, d in enumerate(dates_in_scores) if d >= start), 0)
        scores_at_start = risk_scores_all[min(start_idx, len(risk_scores_all)-1)]
        risk_dict = {nid: scores_at_start[i]
                     for i, nid in enumerate(node_order) if nid in valid}

        # Compute BL weights at period start
        cov_pre = returns_df[valid].loc[:start].tail(252).cov().values * 252
        w_eq = np.ones(len(valid)) / len(valid)
        pi   = compute_equilibrium_returns(cov_pre, w_eq)
        base_mu = pi.copy()
        P, Q, Om = gnn_scores_to_views(risk_dict, valid, metadata, base_mu)
        mu_bl, cov_bl = black_litterman_posterior(pi, cov_pre, P, Q, Om)
        w_bl = mean_variance_optimize(mu_bl, cov_pre + cov_bl)

        # Cumulative returns
        cum_eq = (1 + period_ret @ w_eq).cumprod()
        cum_bl = (1 + period_ret @ w_bl).cumprod()

        max_dd_eq = float(((cum_eq - cum_eq.cummax()) / cum_eq.cummax()).min())
        max_dd_bl = float(((cum_bl - cum_bl.cummax()) / cum_bl.cummax()).min())

        results[period_name] = {
            "dates":       [str(d.date()) for d in period_ret.index],
            "cum_eq":      [round(v, 4) for v in cum_eq.tolist()],
            "cum_bl":      [round(v, 4) for v in cum_bl.tolist()],
            "max_dd_eq":   round(max_dd_eq, 4),
            "max_dd_bl":   round(max_dd_bl, 4),
            "sharpe_eq":   round(float((period_ret @ w_eq).mean() / (period_ret @ w_eq).std() * np.sqrt(252)), 3),
            "sharpe_bl":   round(float((period_ret @ w_bl).mean() / (period_ret @ w_bl).std() * np.sqrt(252)), 3),
        }

    return results


if __name__ == "__main__":
    OUT_DIR = os.path.join(os.path.dirname(__file__), "../../outputs")

    print("Loading data...")
    returns = pd.read_csv(f"{OUT_DIR}/returns.csv", index_col=0, parse_dates=True)
    with open(f"{OUT_DIR}/metadata.json") as f:
        metadata = json.load(f)
    with open(f"{OUT_DIR}/risk_scores.json") as f:
        risk_data = json.load(f)
    with open(f"{OUT_DIR}/shock_scenarios.json") as f:
        scenarios = json.load(f)
    with open(f"{OUT_DIR}/graph_snapshots.json") as f:
        snapshots = json.load(f)

    node_order   = risk_data["node_order"]
    latest_scores = risk_data["scores"][-1]
    risk_dict    = {nid: latest_scores[i] for i, nid in enumerate(node_order)}

    # Base portfolio result (current market state)
    print("\nRunning Black-Litterman optimization (current state)...")
    bl_result = run_bl_optimization(risk_dict, node_order, returns, metadata)

    with open(f"{OUT_DIR}/bl_result.json", "w") as f:
        json.dump(bl_result, f, indent=2)
    print(f"BL result saved → bl_result.json")

    # BL for each crisis scenario
    scenario_bl = {}
    for sname, sdata in scenarios.items():
        print(f"  Optimizing for scenario: {sname}")
        sr = run_bl_optimization(sdata["node_risk"], node_order, returns, metadata)
        scenario_bl[sname] = sr

    with open(f"{OUT_DIR}/scenario_bl.json", "w") as f:
        json.dump(scenario_bl, f, indent=2)
    print(f"Scenario BL results saved → scenario_bl.json")

    # Rolling backtests
    print("\nRunning backtests...")
    backtest = run_backtests(
        snapshots, risk_data["scores"], node_order, returns, metadata
    )
    with open(f"{OUT_DIR}/backtest.json", "w") as f:
        json.dump(backtest, f, indent=2)
    print(f"Backtest saved → backtest.json")

    # Summary
    print(f"\n=== Portfolio Comparison (Current State) ===")
    print(f"  Equal weight:  Sharpe={bl_result['metrics_equal']['sharpe']:.3f}  "
          f"MaxDD={bl_result['metrics_equal']['max_drawdown']:.2%}")
    print(f"  BL + GNN:      Sharpe={bl_result['metrics_bl']['sharpe']:.3f}  "
          f"MaxDD={bl_result['metrics_bl']['max_drawdown']:.2%}")
    print(f"  Sharpe gain:   {bl_result['improvement']['sharpe_delta']:+.3f}")
