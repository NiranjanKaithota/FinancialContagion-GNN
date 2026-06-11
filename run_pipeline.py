"""
run_pipeline.py
===============
Master script — runs the full pipeline end-to-end in order:

  Step 1: Build dataset (real Yahoo Finance data, cache it)
  Step 2: Train HTC-GNN (walk-forward, saves best model)
  Step 3: Evaluate on test set (AUROC, AP, F1, confusion matrix)
  Step 4: Run walk-forward backtest (equity curves, Sharpe, etc.)
  Step 5: Generate publication figures (network heatmaps, sector plots)
  Step 6: Launch interactive dashboard

Usage:
  python run_pipeline.py               # full pipeline + dashboard
  python run_pipeline.py --skip-train  # load existing model, run backtest + dashboard
  python run_pipeline.py --dashboard   # only launch the dashboard
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="FinContagion-GNN Pipeline")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip data download and GNN training")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip the walk-forward backtest")
    parser.add_argument("--dashboard",  action="store_true",
                        help="Only launch the dashboard (skip all computation)")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force re-download and re-build all cached data")
    args = parser.parse_args()

    print("\n" + "═" * 65)
    print("  FinContagion-GNN  —  Full Research Pipeline")
    print("═" * 65 + "\n")

    if not args.dashboard:

        # ── STEP 1: DATA ─────────────────────────────────────────────────────
        if not args.skip_train:
            print("▶ STEP 1: Building dataset from Yahoo Finance …")
            from data_loader import build_dataset
            prices, features_long, adjacencies, crash_labels, spy = \
                build_dataset(force_rebuild=args.force_rebuild)
            print(f"  ✓ Prices: {prices.shape}   "
                  f"Snapshots: {len(adjacencies)}   "
                  f"Labels: {crash_labels.shape}\n")

        # ── STEP 2 & 3: GNN TRAINING + EVAL ──────────────────────────────────
        if not args.skip_train:
            print("▶ STEP 2: Training HTC-GNN …")
            from data_loader import (build_dataset, get_snapshot,
                                      SECTOR_MAP, STOCK_UNIVERSE, NUMERIC_FEATS)
            from gnn_model import (HTCGNNModel, build_pyg_data,
                                    sector_idx_tensor, train_model,
                                    evaluate_model, NUM_FEATURES)

            prices, features_long, adjacencies, crash_labels, spy = \
                build_dataset()

            UNIQUE_SECTORS = sorted(STOCK_UNIVERSE.keys())
            snap_dates     = sorted(adjacencies.keys())
            train_dates    = [d for d in snap_dates if d <= "2018-12-31"]
            val_dates      = [d for d in snap_dates if "2018-12-31" < d <= "2019-12-31"]
            test_dates     = [d for d in snap_dates if d > "2019-12-31"]

            def make_snaps(dates):
                snaps, _sc = [], None
                for d in dates:
                    try:
                        X, A, y, tks, _sc = get_snapshot(
                            features_long, adjacencies, crash_labels, d, _sc)
                        si = sector_idx_tensor(tks, SECTOR_MAP, UNIQUE_SECTORS)
                        snaps.append({
                            "pyg_data":    build_pyg_data(X, A, y),
                            "sector_idx":  si,
                            "num_sectors": len(UNIQUE_SECTORS),
                            "tickers":     tks,
                            "date":        d,
                        })
                    except Exception as e:
                        print(f"  [warn] Skip {d}: {e}")
                return snaps, _sc

            tr_snaps, _  = make_snaps(train_dates)
            va_snaps, _  = make_snaps(val_dates)
            te_snaps, _  = make_snaps(test_dates)

            model = train_model(tr_snaps, va_snaps,
                                num_features=len(NUMERIC_FEATS))

            print("\n▶ STEP 3: Test-set evaluation …")
            summary = evaluate_model(model, te_snaps, list(SECTOR_MAP.keys()))
            summary.to_csv("gnn_predictions.csv")
            print(f"\n  ✓ Predictions saved → gnn_predictions.csv\n")

        # ── STEP 4: BACKTEST ─────────────────────────────────────────────────
        if not args.skip_backtest:
            print("▶ STEP 4: Running walk-forward backtest …")
            from backtest_loop import run_backtest
            results, metrics, monthly = run_backtest()
            print("  ✓ Backtest complete.\n")

        # ── STEP 5: STATIC FIGURES ────────────────────────────────────────────
        print("▶ STEP 5: Generating publication figures …")
        try:
            from visualize_network import (load_data, build_graph,
                                            plot_risk_heatmap, plot_sector_graph,
                                            plot_sector_risk_matrix)
            import networkx as nx
            preds, A_df, cl = load_data()
            if A_df is not None:
                G, tks = build_graph(A_df, preds)
                pos = nx.spring_layout(G, k=1.4, seed=42, iterations=100)
                plot_risk_heatmap(G, pos, preds)
                plot_sector_graph(G, preds)
                plot_sector_risk_matrix(preds)
                print("  ✓ Figures saved.\n")
            else:
                print("  [warn] edge_matrix_A.csv not found — skipping figures\n")
        except Exception as e:
            print(f"  [warn] Figure generation failed: {e}\n")

    # ── STEP 6: DASHBOARD ────────────────────────────────────────────────────
    print("▶ STEP 6: Launching dashboard …")
    print("  Open your browser at http://127.0.0.1:8050\n")
    from generate_dashboard import app
    app.run(debug=False, port=8050, host="0.0.0.0")


if __name__ == "__main__":
    main()
