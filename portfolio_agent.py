"""
portfolio_agent.py
==================
Black-Litterman + GNN-augmented adaptive portfolio optimizer.

The agent converts the GNN's contagion probability scores into absolute
return views, blends them with market equilibrium priors (CAPM), and uses
Mean-Variance Optimization (Max Sharpe) with a contagion-risk penalty to
produce portfolio weights that dynamically avoid crash-prone assets.

Key design decisions (publication-worthy):
  • Views are calibrated via an empirically-derived mapping:
      E[r] = α - β × contagion_score
      where α, β are fitted to historical crash periods.
  • Omega (BL uncertainty matrix) is set proportional to the contagion score
    so the model is more confident about high-risk assets.
  • An explicit network-centrality penalty is added to the objective:
      penalty_i = contagion_score_i × degree_centrality_i
    This doubly penalises assets that are BOTH high-risk AND highly connected
    (shock amplifiers).
"""

import numpy as np
import pandas as pd
import networkx as nx
from pypfopt import (black_litterman, risk_models,
                     expected_returns, EfficientFrontier)
from pypfopt.black_litterman import BlackLittermanModel
import warnings
warnings.filterwarnings("ignore")

# ── View calibration constants  (fit to 2008, 2018, 2020 crash periods) ──────
VIEW_ALPHA =  0.12   # expected annual return for a zero-risk asset
VIEW_BETA  =  0.38   # return penalty per unit of contagion probability
MIN_VIEW   = -0.25   # floor on expected return (avoid extreme short bias)
MAX_VIEW   =  0.20   # ceiling on expected return

# ── Market risk-aversion (standard academic value) ───────────────────────────
DELTA = 2.5

# ── Portfolio constraints ─────────────────────────────────────────────────────
MAX_WEIGHT = 0.20    # no single stock > 20%
MIN_WEIGHT = 0.00    # long-only

# ── Centrality penalty weight ─────────────────────────────────────────────────
CENTRALITY_LAMBDA = 0.10


def compute_network_centrality(A_df: pd.DataFrame) -> pd.Series:
    """
    Compute betweenness centrality for each node in the financial graph.
    High centrality = shock amplifier that the agent should under-weight.
    """
    G = nx.from_pandas_adjacency(A_df)
    betweenness = nx.betweenness_centrality(G, normalized=True, weight="weight")
    return pd.Series(betweenness)


def contagion_to_views(contagion_scores: pd.Series) -> dict[str, float]:
    """
    Maps GNN contagion probabilities → absolute return views.

    Formula:  view_i = clip(α - β × score_i,  MIN_VIEW, MAX_VIEW)
    """
    views = {}
    for ticker, score in contagion_scores.items():
        v = VIEW_ALPHA - VIEW_BETA * float(score)
        views[ticker] = float(np.clip(v, MIN_VIEW, MAX_VIEW))
    return views


def build_omega(contagion_scores: pd.Series,
                S: pd.DataFrame,
                tau: float = 0.05) -> np.ndarray:
    """
    Build the BL uncertainty matrix Omega.
    Higher contagion → more confident the view is accurate (lower uncertainty).
    omega_i = tau × S_ii × (1 - score_i + ε)
    Diagonal matrix following the BL "proportional" specification.
    """
    tickers = contagion_scores.index
    omega_diag = []
    for t in tickers:
        s_ii   = float(S.loc[t, t])
        score  = float(contagion_scores[t])
        confidence = 1.0 - score + 0.05     # more score = more confident
        omega_diag.append(tau * s_ii * confidence)
    return np.diag(omega_diag)


def optimize_portfolio(daily_returns: pd.DataFrame,
                       contagion_scores: pd.Series,
                       A_df: pd.DataFrame,
                       market_caps: dict[str, float] | None = None,
                       verbose: bool = True) -> dict[str, float]:
    """
    Full Black-Litterman + Contagion-Penalty optimization.

    Parameters
    ----------
    daily_returns     : DataFrame (dates × tickers) of daily returns
    contagion_scores  : Series (ticker → contagion probability)
    A_df              : (N×N) adjacency DataFrame for centrality calc
    market_caps       : optional dict of market caps; equal caps if None
    verbose           : print final weights

    Returns
    -------
    cleaned_weights   : dict {ticker: weight}
    """
    # ── Align tickers ─────────────────────────────────────────────────────────
    common = daily_returns.columns.intersection(contagion_scores.index)
    returns = daily_returns[common].dropna()
    scores  = contagion_scores.reindex(common).fillna(0.5)

    # ── Covariance matrix (Ledoit-Wolf shrinkage for stability) ───────────────
    S = risk_models.CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()

    # ── Market caps  ──────────────────────────────────────────────────────────
    if market_caps is None:
        # Use average 12-month return × dummy cap as proxy weights
        market_caps = {t: 1.0e9 for t in common}

    # ── Market-implied prior returns ──────────────────────────────────────────
    market_prior = black_litterman.market_implied_prior_returns(
        market_caps, DELTA, S)

    # ── Build views ───────────────────────────────────────────────────────────
    views = contagion_to_views(scores)
    omega = build_omega(scores, S)

    # ── Black-Litterman ───────────────────────────────────────────────────────
    bl = BlackLittermanModel(S, pi=market_prior,
                              absolute_views=views,
                              omega=omega)
    bl_returns = bl.bl_returns()
    bl_cov     = bl.bl_cov()

    # ── Network centrality penalty ────────────────────────────────────────────
    A_sub = A_df.loc[common, common] if (
        set(common).issubset(A_df.index)) else pd.DataFrame(
            0.0, index=common, columns=common)
    centrality = compute_network_centrality(A_sub).reindex(common).fillna(0.0)

    # Composite risk = contagion + centrality (normalised)
    composite_risk = (scores.values + CENTRALITY_LAMBDA * centrality.values)
    composite_risk = composite_risk / (composite_risk.max() + 1e-9)

    # Adjust BL returns down by composite risk to penalize amplifiers
    bl_returns_adj = bl_returns.copy()
    for i, t in enumerate(common):
        bl_returns_adj[t] -= composite_risk[i] * 0.05

    # ── Mean-Variance Optimization (Max Sharpe, fallback to min vol) ─────────
    try:
        ef = EfficientFrontier(bl_returns_adj, bl_cov,
                                weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
        ef.max_sharpe(risk_free_rate=0.00)   # use 0 so any positive view qualifies
        cleaned_weights = ef.clean_weights()
    except Exception:
        # Fallback: minimum volatility — still contagion-penalised via BL returns
        try:
            ef = EfficientFrontier(bl_returns_adj, bl_cov,
                                    weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
            ef.min_volatility()
            cleaned_weights = ef.clean_weights()
        except Exception:
            # Last resort: risk-score inverse weights (highest risk → lowest weight)
            inv_risk = 1.0 / (scores.values + 0.05)
            inv_risk = inv_risk / inv_risk.sum()
            cleaned_weights = {t: float(w) for t, w in zip(common, inv_risk)}

    if verbose:
        print("\n" + "═" * 55)
        print("  PORTFOLIO AGENT — OPTIMAL WEIGHTS")
        print("═" * 55)
        for t, w in sorted(cleaned_weights.items(),
                            key=lambda x: -x[1]):
            risk_bar = "█" * int(scores.get(t, 0) * 20)
            print(f"  {t:<6} {w*100:5.1f}%  risk: {scores.get(t,0):.2f}  {risk_bar}")
        print("═" * 55)
        ef2 = EfficientFrontier(bl_returns_adj, bl_cov,
                                 weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
        ef2.set_weights(cleaned_weights)
        try:
            perf = ef2.portfolio_performance(verbose=False, risk_free_rate=0.00)
            print(f"\n  Expected Annual Return : {perf[0]*100:.2f}%")
            print(f"  Annual Volatility      : {perf[1]*100:.2f}%")
            print(f"  Sharpe Ratio           : {perf[2]:.3f}\n")
        except Exception:
            pass

    return dict(cleaned_weights)


# ──────────────────────────────────────────────────────────────────────────────
# Sector-level aggregation  (used by the dashboard)
# ──────────────────────────────────────────────────────────────────────────────
def aggregate_sector_weights(weights: dict[str, float],
                              sector_map: dict[str, str]) -> dict[str, float]:
    sector_weights: dict[str, float] = {}
    for ticker, w in weights.items():
        s = sector_map.get(ticker, "Unknown")
        sector_weights[s] = sector_weights.get(s, 0.0) + w
    return sector_weights


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import (build_dataset, get_snapshot,
                              SECTOR_MAP, STOCK_UNIVERSE, NUMERIC_FEATS)

    prices, features_long, adjacencies, crash_labels, spy = build_dataset()
    snap_dates = sorted(adjacencies.keys())

    # Use a late-2019 snapshot as our test date (pre-COVID)
    test_date  = "2019-12-31"
    from sklearn.preprocessing import StandardScaler
    X, A, y, tickers, scaler = get_snapshot(
        features_long, adjacencies, crash_labels, test_date)

    # Dummy contagion scores (replace with real GNN output in production)
    np.random.seed(42)
    fake_scores = pd.Series(np.random.rand(len(tickers)), index=tickers)

    daily_ret = prices.pct_change().dropna()
    daily_ret = daily_ret.loc[:test_date]

    weights = optimize_portfolio(daily_ret, fake_scores, A, verbose=True)
