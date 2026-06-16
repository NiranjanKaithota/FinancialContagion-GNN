"""
Master pipeline runner.
Run this once to generate all outputs before starting the API + frontend.

Steps:
  1. Fetch NIFTY 50 data from Yahoo Finance
  2. Build dynamic graph snapshots
  3. Train GNN + generate risk scores
  4. Run Black-Litterman optimization
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

OUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def step(n, title):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'='*60}")


def main():
    total_start = time.time()

    # ── Step 1: Data ────────────────────────────────────────────────────────
    step(1, "Fetching NIFTY 50 data from Yahoo Finance")
    from data.fetch_data import fetch_prices, compute_returns, compute_rolling_stats, build_metadata, save_outputs
    prices = fetch_prices(start="2018-01-01")
    returns = compute_returns(prices)
    rolling_stats = compute_rolling_stats(returns)
    metadata = build_metadata(prices, rolling_stats)
    save_outputs(prices, returns, metadata, OUT_DIR)

    # ── Step 2: Graph ────────────────────────────────────────────────────────
    step(2, "Building dynamic graph snapshots")
    import pandas as pd, json
    from data.build_graph import build_snapshots, compute_sector_adjacency

    returns_df = pd.read_csv(f"{OUT_DIR}/returns.csv", index_col=0, parse_dates=True)
    with open(f"{OUT_DIR}/metadata.json") as f:
        metadata = json.load(f)

    snapshots = build_snapshots(returns_df, metadata)
    sector_adj = compute_sector_adjacency(snapshots)

    with open(f"{OUT_DIR}/graph_snapshots.json", "w") as f:
        json.dump(snapshots, f)
    with open(f"{OUT_DIR}/sector_adjacency.json", "w") as f:
        json.dump(sector_adj, f)
    print(f"  Saved {len(snapshots)} snapshots")

    # ── Step 3: GNN ─────────────────────────────────────────────────────────
    step(3, "Training GNN + generating risk scores")
    import torch
    from models.gnn_model import (train_model, generate_risk_scores,
                                   generate_shock_scenarios)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, node_order, pyg_snaps = train_model(
        snapshots, returns_df, epochs=40, seq_len=8, device=device
    )
    torch.save({"model_state": model.state_dict(), "node_order": node_order},
               f"{OUT_DIR}/gnn_model.pt")

    risk_scores = generate_risk_scores(model, pyg_snaps, node_order, seq_len=8)
    snap_dates  = [s["date"] for s in snapshots[8:]]
    with open(f"{OUT_DIR}/risk_scores.json", "w") as f:
        json.dump({"node_order": node_order, "dates": snap_dates,
                   "scores": risk_scores}, f)

    scenarios = generate_shock_scenarios(model, pyg_snaps, node_order)
    with open(f"{OUT_DIR}/shock_scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved risk scores for {len(risk_scores)} snapshots")

    # ── Step 4: Black-Litterman ──────────────────────────────────────────────
    step(4, "Running Black-Litterman optimization")
    from engine.black_litterman import run_bl_optimization, run_backtests

    latest_scores = risk_scores[-1]
    risk_dict = {nid: latest_scores[i] for i, nid in enumerate(node_order)}

    bl_result = run_bl_optimization(risk_dict, node_order, returns_df, metadata)
    with open(f"{OUT_DIR}/bl_result.json", "w") as f:
        json.dump(bl_result, f, indent=2)

    scenario_bl = {}
    for sname, sdata in scenarios.items():
        sr = run_bl_optimization(sdata["node_risk"], node_order, returns_df, metadata)
        scenario_bl[sname] = sr
    with open(f"{OUT_DIR}/scenario_bl.json", "w") as f:
        json.dump(scenario_bl, f, indent=2)

    with open(f"{OUT_DIR}/risk_scores.json") as f:
        risk_data = json.load(f)
    backtest = run_backtests(snapshots, risk_data["scores"], node_order, returns_df, metadata)
    with open(f"{OUT_DIR}/backtest.json", "w") as f:
        json.dump(backtest, f, indent=2)

    # ── Done ────────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    print(f"  Output files in: {OUT_DIR}/")
    print(f"    prices.csv, returns.csv, metadata.json")
    print(f"    graph_snapshots.json, sector_adjacency.json")
    print(f"    gnn_model.pt, risk_scores.json, shock_scenarios.json")
    print(f"    bl_result.json, scenario_bl.json, backtest.json")
    print(f"\n  Next: cd backend && uvicorn api.main:app --reload --port 8000")
    print(f"        cd frontend && npm install && npm run dev")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
