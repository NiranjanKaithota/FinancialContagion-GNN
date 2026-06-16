"""
Layer 2: Dynamic Graph Construction
Builds weekly graph snapshots from return correlations + Granger causality.
Output: graph_snapshots.json  (list of {date, nodes, edges} objects)
"""

import pandas as pd
import numpy as np
import json
import os
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")


WINDOW = 60        # trading days per snapshot (~3 months)
STEP   = 20        # slide forward 20 days (~1 month) per snapshot
CORR_THRESHOLD  = 0.35   # |ρ| above this → add correlation edge
GRANGER_PVAL    = 0.05   # p-value below this → add Granger edge
GRANGER_MAXLAG  = 2      # max lag for Granger test


def build_node_features(returns_window, metadata):
    """
    Node feature vector per stock in this window:
      [mean_return, volatility, skewness, kurtosis, beta_to_portfolio]
    """
    market = returns_window.mean(axis=1)
    nodes = []
    for ticker in returns_window.columns:
        r = returns_window[ticker]
        vol = float(r.std() * np.sqrt(252))
        mu  = float(r.mean() * 252)
        skew = float(r.skew())
        kurt = float(r.kurtosis())
        cov  = float(r.cov(market))
        var  = float(market.var())
        beta = cov / var if var > 1e-10 else 1.0
        meta = metadata.get(ticker, {})
        nodes.append({
            "id":     ticker,
            "name":   meta.get("name", ticker),
            "sector": meta.get("sector", "Unknown"),
            "color":  meta.get("color", "#888780"),
            "features": {
                "mu":   round(mu, 4),
                "vol":  round(vol, 4),
                "skew": round(skew, 4),
                "kurt": round(kurt, 4),
                "beta": round(beta, 4),
            }
        })
    return nodes


def build_correlation_edges(returns_window, threshold=CORR_THRESHOLD):
    """Add an edge between i and j if |corr(i,j)| > threshold."""
    corr = returns_window.corr()
    tickers = list(returns_window.columns)
    edges = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = corr.iloc[i, j]
            if abs(c) >= threshold:
                edges.append({
                    "source":   tickers[i],
                    "target":   tickers[j],
                    "weight":   round(float(abs(c)), 4),
                    "type":     "correlation",
                    "directed": False,
                })
    return edges


def build_granger_edges(returns_window, tickers_subset=None,
                        maxlag=GRANGER_MAXLAG, pval_thresh=GRANGER_PVAL):
    """
    Directed edges: i → j if returns of i Granger-cause returns of j.
    Run only on a subset for speed (top-20 by vol).
    """
    if tickers_subset is None:
        vols = returns_window.std()
        tickers_subset = list(vols.nlargest(20).index)

    edges = []
    subset = returns_window[tickers_subset].dropna()

    for i, t1 in enumerate(tickers_subset):
        for t2 in tickers_subset:
            if t1 == t2:
                continue
            try:
                data = subset[[t2, t1]].dropna()
                if len(data) < 30:
                    continue
                result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                # Use minimum p-value across lags
                min_pval = min(
                    result[lag][0]["ssr_ftest"][1]
                    for lag in range(1, maxlag + 1)
                )
                if min_pval < pval_thresh:
                    edges.append({
                        "source":   t1,
                        "target":   t2,
                        "weight":   round(float(1 - min_pval), 4),
                        "type":     "granger",
                        "directed": True,
                    })
            except Exception:
                continue
    return edges


def build_snapshots(returns, metadata, window=WINDOW, step=STEP):
    """Slide window over full return history and build graph per snapshot."""
    tickers = list(returns.columns)
    dates   = returns.index
    snapshots = []

    indices = list(range(window, len(dates), step))
    print(f"Building {len(indices)} graph snapshots...")

    for idx_num, end_idx in enumerate(indices):
        start_idx = end_idx - window
        window_ret = returns.iloc[start_idx:end_idx]
        snap_date  = str(dates[end_idx - 1].date())

        nodes = build_node_features(window_ret, metadata)
        corr_edges = build_correlation_edges(window_ret)
        gran_edges = build_granger_edges(window_ret)

        # Merge & deduplicate edges
        all_edges = corr_edges + gran_edges

        snapshots.append({
            "date":       snap_date,
            "node_count": len(nodes),
            "edge_count": len(all_edges),
            "nodes":      nodes,
            "edges":      all_edges,
        })

        if (idx_num + 1) % 5 == 0 or idx_num == 0:
            print(f"  [{idx_num+1}/{len(indices)}] {snap_date}: "
                  f"{len(nodes)} nodes, {len(all_edges)} edges "
                  f"({len(corr_edges)} corr + {len(gran_edges)} granger)")

    return snapshots


def compute_sector_adjacency(snapshots):
    """
    Aggregate edges to sector level for macro contagion view.
    Returns list of {date, sector_edges} per snapshot.
    """
    sector_snapshots = []
    for snap in snapshots:
        node_sector = {n["id"]: n["sector"] for n in snap["nodes"]}
        sector_weights = {}
        for e in snap["edges"]:
            s1 = node_sector.get(e["source"], "Unknown")
            s2 = node_sector.get(e["target"], "Unknown")
            if s1 == s2:
                continue
            key = tuple(sorted([s1, s2]))
            sector_weights[key] = sector_weights.get(key, 0) + e["weight"]
        sector_edges = [
            {"source": k[0], "target": k[1], "weight": round(v, 4)}
            for k, v in sector_weights.items()
        ]
        sector_snapshots.append({
            "date": snap["date"],
            "sector_edges": sector_edges,
        })
    return sector_snapshots


if __name__ == "__main__":
    OUT_DIR = os.path.join(os.path.dirname(__file__), "../../outputs")

    print("Loading data...")
    returns  = pd.read_csv(f"{OUT_DIR}/returns.csv", index_col=0, parse_dates=True)
    with open(f"{OUT_DIR}/metadata.json") as f:
        metadata = json.load(f)

    snapshots = build_snapshots(returns, metadata)
    sector_adj = compute_sector_adjacency(snapshots)

    with open(f"{OUT_DIR}/graph_snapshots.json", "w") as f:
        json.dump(snapshots, f)

    with open(f"{OUT_DIR}/sector_adjacency.json", "w") as f:
        json.dump(sector_adj, f)

    print(f"\nSaved {len(snapshots)} snapshots → graph_snapshots.json")
    print(f"Saved {len(sector_adj)} sector graphs → sector_adjacency.json")
    print(f"\nExample snapshot [{snapshots[-1]['date']}]:")
    print(f"  {snapshots[-1]['node_count']} nodes, {snapshots[-1]['edge_count']} edges")
