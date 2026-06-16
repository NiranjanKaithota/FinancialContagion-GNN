"""
backtest_loop.py
================
Walk-forward backtest of the HTC-GNN + BL portfolio agent.

Methodology (zero look-ahead bias):
────────────────────────────────────
For each month t in the test window [2020-01 … 2024-12]:
  1. Build graph snapshot using only data available at t
  2. Run the trained GNN to get contagion scores
  3. Feed scores into the BL portfolio agent → weights w_t
  4. Compute realised portfolio return over [t, t+1]
  5. Record metrics: return, Sharpe, max-drawdown, turnover

Three strategies compared:
  • GNN-BL Agent      : our model
  • Equal Weight (EW) : 1/N benchmark
  • SPY Buy-and-Hold  : market index

Output artefacts:
  backtest_results.csv   — daily portfolio values
  backtest_metrics.csv   — period-level risk/return statistics
  backtest_chart.png     — publication-quality comparison chart
"""

import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from data_loader import (build_dataset, get_snapshot,
                          SECTOR_MAP, STOCK_UNIVERSE, NUMERIC_FEATS)
from gnn_model import (HTCGNNModel, build_pyg_data, sector_idx_tensor,
                        DEVICE, NUM_FEATURES, HIDDEN_DIM, FocalLoss)
from portfolio_agent import optimize_portfolio, aggregate_sector_weights

UNIQUE_SECTORS = sorted(STOCK_UNIVERSE.keys())
MODEL_PATH     = os.path.join("cache", "best_model.pt")

# ── Backtest period ───────────────────────────────────────────────────────────
BACKTEST_START = "2020-01-01"
BACKTEST_END   = "2024-11-30"
INITIAL_CAPITAL = 1_000_000.0

# ── Risk-free rate (annualised, for Sharpe) ───────────────────────────────────
RISK_FREE_ANNUAL = 0.04
RISK_FREE_DAILY  = RISK_FREE_ANNUAL / 252

# ── Enhancements ──────────────────────────────────────────────────────────────
TRANSACTION_COST_BPS      = 0.0010  # 10 bps dynamic transaction cost rebalancing drag
CIRCUIT_BREAKER_THRESHOLD = 0.45    # Rotate to 100% Cash if average contagion > 0.45


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def sharpe(returns: np.ndarray) -> float:
    excess = returns - RISK_FREE_DAILY
    if excess.std() < 1e-9:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / (running_max + 1e-9)
    return float(dd.min())


def sortino(returns: np.ndarray) -> float:
    excess      = returns - RISK_FREE_DAILY
    downside    = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / (downside.std() + 1e-9))


def calmar(returns: np.ndarray, equity: np.ndarray) -> float:
    ann_ret = (1 + returns.mean()) ** 252 - 1
    mdd     = abs(max_drawdown(equity))
    return float(ann_ret / mdd) if mdd > 1e-6 else 0.0


def portfolio_return(weights: dict, daily_ret_slice: pd.DataFrame) -> pd.Series:
    """
    Compute daily portfolio returns given a constant weight dict
    over daily_ret_slice (a sub-DataFrame of daily returns).
    """
    tickers = [t for t in weights if t in daily_ret_slice.columns]
    w_arr   = np.array([weights[t] for t in tickers])
    w_arr   = w_arr / (w_arr.sum() + 1e-9)
    ret_mat = daily_ret_slice[tickers].values
    port_ret = ret_mat @ w_arr
    return pd.Series(port_ret, index=daily_ret_slice.index)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[backtest] Loading dataset ...")
    prices, features_long, adjacencies, crash_labels, spy = build_dataset()
    daily_ret = prices.pct_change().dropna()

    # SPY returns for benchmark
    spy_ret   = spy.pct_change().dropna()
    spy_ret.name = "SPY"

    # ── 2. Load trained GNN ───────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print("[backtest] No trained model found. Running training first ...")
        from gnn_model import train_model
        snap_dates    = sorted(adjacencies.keys())
        train_dates   = [d for d in snap_dates if d <= "2018-12-31"]
        val_dates     = [d for d in snap_dates if "2018-12-31" < d <= "2019-12-31"]

        def make_snaps(dates):
            snaps, scaler_ = [], None
            for d in dates:
                try:
                    X, A, y, tks, scaler_ = get_snapshot(
                        features_long, adjacencies, crash_labels, d, scaler_)
                    si = sector_idx_tensor(tks, SECTOR_MAP, UNIQUE_SECTORS)
                    snaps.append({
                        "pyg_data":    build_pyg_data(X, A, y),
                        "sector_idx":  si,
                        "num_sectors": len(UNIQUE_SECTORS),
                        "tickers":     tks, "date": d,
                    })
                except Exception as e:
                    print(f"  [warn] Skip {d}: {e}")
            return snaps, scaler_

        tr_snaps, _  = make_snaps(train_dates)
        va_snaps, _  = make_snaps(val_dates)
        train_model(tr_snaps, va_snaps, num_features=len(NUMERIC_FEATS))

    model = HTCGNNModel(num_features=len(NUMERIC_FEATS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                      weights_only=True))
    model.eval()
    print(f"[backtest] Model loaded from {MODEL_PATH}")

    # ── Initialize Online Fine-Tuning (WFFT) & Enhancements ───────────────────
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_ft = FocalLoss(alpha=0.75, gamma=2.0)
    
    snap_history = []
    gnn_prev_drift = None
    ew_prev_drift  = None

    # ── 3. Walk-forward loop (monthly rebalancing) ────────────────────────────
    current_date = pd.Timestamp(BACKTEST_START)
    end_date     = pd.Timestamp(BACKTEST_END)

    gnn_records = []
    ew_records  = []
    spy_records = []
    monthly_log = []

    current_weights = None
    scaler          = None

    # equal-weight baseline
    all_tickers = [t for v in STOCK_UNIVERSE.values() for t in v]
    ew_weights  = {t: 1.0 / len(all_tickers) for t in all_tickers}

    print(f"\n[backtest] Walk-forward: {BACKTEST_START} -> {BACKTEST_END}")
    month_iter = []
    d = current_date
    while d < end_date:
        month_iter.append(d)
        d = d + relativedelta(months=1)

    for snap_date in tqdm(month_iter, desc="Monthly rebalancing"):
        snap_str  = snap_date.strftime("%Y-%m-%d")
        next_date = snap_date + relativedelta(months=1)

        # ── Get graph snapshot ────────────────────────────────────────────────
        try:
            X, A, y, tickers, scaler = get_snapshot(
                features_long, adjacencies, crash_labels, snap_str, scaler)
        except Exception as e:
            tqdm.write(f"  [warn] Snapshot {snap_str} failed: {e}  — skipping")
            continue

        si = sector_idx_tensor(tickers, SECTOR_MAP, UNIQUE_SECTORS).to(DEVICE)
        data = build_pyg_data(X, A, y).to(DEVICE)

        H = len(snap_history)

        # ── Walk-Forward Fine-Tuning (WFFT) Online Update ─────────────────────
        # If we have a snapshot from 3 months ago (since labels look 63 trading days forward),
        # its targets are now fully realized and can be used to update GNN weights online.
        if H >= 3:
            model.train()
            snap_prev = snap_history[H-3]
            h_prev_ft = snap_history[H-4]["h_state"] if (H-4 >= 0) else None
            
            for ft_epoch in range(5):
                optimizer_ft.zero_grad()
                logits_ft, _, _ = model(
                    snap_prev["pyg_data"],
                    snap_prev["sector_idx"],
                    len(UNIQUE_SECTORS),
                    h_prev=h_prev_ft.detach() if h_prev_ft is not None else None
                )
                loss_ft = criterion_ft(logits_ft, snap_prev["pyg_data"].y)
                loss_ft.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_ft.step()

        # ── Run GNN Inference on Current Snapshot ─────────────────────────────
        model.eval()
        h_prev_curr = snap_history[H-1]["h_state"] if (H-1 >= 0) else None
        with torch.no_grad():
            logits, h_state_new, _ = model(
                data,
                si,
                len(UNIQUE_SECTORS),
                h_prev=h_prev_curr
            )
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        contagion_scores = pd.Series(probs, index=tickers)

        # Save snapshot and state in history for future fine-tuning
        snap_history.append({
            "date": snap_str,
            "pyg_data": data,
            "sector_idx": si,
            "tickers": tickers,
            "h_state": h_state_new.detach() if h_state_new is not None else None,
        })

        avg_contagion = float(contagion_scores.mean())

        # ── Systemic Circuit Breaker ──────────────────────────────────────────
        # If systemic contagion risk is extremely high, rotate 100% of capital to Cash
        if avg_contagion > CIRCUIT_BREAKER_THRESHOLD:
            cb_active = True
            current_weights = {t: 0.0 for t in tickers}
        else:
            cb_active = False
            # ── Optimize portfolio ────────────────────────────────────────────
            hist_ret = daily_ret.loc[:snap_str]
            try:
                current_weights = optimize_portfolio(
                    hist_ret, contagion_scores, A, verbose=False)
            except Exception as e:
                tqdm.write(f"  [warn] Optimization failed {snap_str}: {e}")
                if current_weights is None:
                    current_weights = ew_weights.copy()

        # ── Realised returns for this month ───────────────────────────────────
        mask = (daily_ret.index > snap_date) & (daily_ret.index <= next_date)
        month_ret = daily_ret.loc[mask]
        if len(month_ret) == 0:
            continue

        if cb_active:
            # Portfolio return is daily risk-free rate (Cash proxy)
            gnn_ret = pd.Series(RISK_FREE_DAILY, index=month_ret.index)
        else:
            gnn_ret = portfolio_return(current_weights, month_ret)

        ew_ret  = portfolio_return(ew_weights,      month_ret)
        spy_day = spy_ret.reindex(month_ret.index).fillna(0.0)

        # ── Drift-adjusted weights calculation for previous month-end ─────────
        # For GNN Agent:
        asset_cum_ret = (1 + month_ret).prod()
        w_drift_old = {}
        for t in tickers:
            w_drift_old[t] = current_weights.get(t, 0.0) * asset_cum_ret.get(t, 1.0)
        sum_drift = sum(w_drift_old.values())
        if sum_drift > 1e-9:
            w_drift_old = {t: w / sum_drift for t, w in w_drift_old.items()}
        else:
            w_drift_old = {t: 0.0 for t in tickers}

        # For Equal Weight Agent:
        ew_drift_old = {}
        for t in tickers:
            ew_drift_old[t] = ew_weights.get(t, 0.0) * asset_cum_ret.get(t, 1.0)
        sum_ew_drift = sum(ew_drift_old.values())
        if sum_ew_drift > 1e-9:
            ew_drift_old = {t: w / sum_ew_drift for t, w in ew_drift_old.items()}
        else:
            ew_drift_old = {t: 0.0 for t in tickers}

        # ── Compute turnover and transaction costs ────────────────────────────
        # For GNN Agent:
        if len(gnn_records) == 0 or gnn_prev_drift is None:
            # First month: turnover is 1.0 (allocating from cash/capital)
            turnover = 1.0
        else:
            turnover = sum(abs(current_weights.get(t, 0.0) - gnn_prev_drift.get(t, 0.0)) for t in tickers)

        # For Equal Weight Agent:
        if len(ew_records) == 0 or ew_prev_drift is None:
            ew_turnover = 1.0
        else:
            ew_turnover = sum(abs(ew_weights.get(t, 0.0) - ew_prev_drift.get(t, 0.0)) for t in tickers)

        gnn_prev_drift = w_drift_old
        ew_prev_drift  = ew_drift_old

        # Adjust returns for transaction costs (apply drag on the first trading day of the month)
        gnn_ret_adj = gnn_ret.copy()
        gnn_ret_adj.iloc[0] -= TRANSACTION_COST_BPS * turnover

        ew_ret_adj = ew_ret.copy()
        ew_ret_adj.iloc[0] -= TRANSACTION_COST_BPS * ew_turnover

        gnn_records.append(gnn_ret_adj)
        ew_records.append(ew_ret_adj)
        spy_records.append(spy_day)

        # Sector allocation this month
        sector_w = aggregate_sector_weights(current_weights, SECTOR_MAP)
        if cb_active:
            sector_w = {}
            w_cash_val = 1.0
        else:
            w_cash_val = 0.0

        monthly_log.append({
            "date": snap_str,
            "gnn_monthly_ret": float(gnn_ret_adj.sum()),
            "ew_monthly_ret":  float(ew_ret_adj.sum()),
            "avg_contagion":   avg_contagion,
            "max_contagion":   float(contagion_scores.max()),
            "w_Cash":          w_cash_val,
            **{f"w_{s}": sector_w.get(s, 0.0) for s in UNIQUE_SECTORS},
        })

    # ── 4. Compile equity curves ──────────────────────────────────────────────
    gnn_daily = pd.concat(gnn_records)
    ew_daily  = pd.concat(ew_records)
    spy_daily = pd.concat(spy_records)

    gnn_equity = INITIAL_CAPITAL * (1 + gnn_daily).cumprod()
    ew_equity  = INITIAL_CAPITAL * (1 + ew_daily).cumprod()
    spy_equity = INITIAL_CAPITAL * (1 + spy_daily).cumprod()

    results = pd.DataFrame({
        "GNN_Agent":  gnn_equity,
        "EqualWeight": ew_equity,
        "SPY":        spy_equity,
    }).dropna()
    results.to_csv("backtest_results.csv")

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    metrics = {}
    for name, ret_series, eq in [
        ("GNN_Agent",   gnn_daily, gnn_equity),
        ("EqualWeight", ew_daily,  ew_equity),
        ("SPY",         spy_daily, spy_equity),
    ]:
        ann_ret = float((1 + ret_series.mean()) ** 252 - 1)
        ann_vol = float(ret_series.std() * np.sqrt(252))
        metrics[name] = {
            "Annual Return (%)":  round(ann_ret * 100, 2),
            "Annual Volatility (%)": round(ann_vol * 100, 2),
            "Sharpe Ratio":       round(sharpe(ret_series.values), 3),
            "Sortino Ratio":      round(sortino(ret_series.values), 3),
            "Max Drawdown (%)":   round(max_drawdown(eq.values) * 100, 2),
            "Calmar Ratio":       round(calmar(ret_series.values, eq.values), 3),
            "Final Value ($)":    round(float(eq.iloc[-1]), 0),
        }

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv("backtest_metrics.csv")

    # Export performance metrics to JSON format
    import json
    metrics_json = {}
    for strategy, vals in metrics.items():
        metrics_json[strategy] = {
            "annual_return": vals["Annual Return (%)"],
            "annual_volatility": vals["Annual Volatility (%)"],
            "sharpe_ratio": vals["Sharpe Ratio"],
            "sortino_ratio": vals["Sortino Ratio"],
            "max_drawdown": vals["Max Drawdown (%)"],
            "calmar_ratio": vals["Calmar Ratio"],
            "final_value": vals["Final Value ($)"]
        }
    with open("backtest_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=4)
    print("[backtest] Metrics saved -> backtest_metrics.json")

    print("\n" + "=" * 65)
    print("  BACKTEST RESULTS SUMMARY")
    print("=" * 65)
    print(metrics_df.to_string())
    print("=" * 65)

    monthly_df = pd.DataFrame(monthly_log)
    monthly_df.to_csv("monthly_rebalance_log.csv", index=False)

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    _plot_backtest(results, monthly_df, metrics_df)
    return results, metrics_df, monthly_df


# ──────────────────────────────────────────────────────────────────────────────
# PUBLICATION-QUALITY BACKTEST CHART
# ──────────────────────────────────────────────────────────────────────────────
def _plot_backtest(results: pd.DataFrame,
                   monthly_df: pd.DataFrame,
                   metrics_df: pd.DataFrame):
    fig = plt.figure(figsize=(18, 14), facecolor="#0f172a")
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             hspace=0.4, wspace=0.35,
                             left=0.07, right=0.96,
                             top=0.93, bottom=0.07)

    DARK_BG   = "#0f172a"
    PANEL_BG  = "#1e293b"
    BORDER    = "#334155"
    COLORS    = {"GNN_Agent": "#10b981", "EqualWeight": "#f59e0b", "SPY": "#60a5fa"}
    TEXT_CLR  = "#e2e8f0"

    def style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        ax.grid(color=BORDER, linewidth=0.5, alpha=0.6)

    # ── Panel 1: Equity curves ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for col, clr in COLORS.items():
        if col in results.columns:
            ax1.plot(results.index, results[col],
                     label=col, color=clr, linewidth=2.0)
    ax1.set_title("Portfolio Equity Curve — Walk-Forward Backtest (2020–2024)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax1.legend(facecolor=PANEL_BG, edgecolor=BORDER,
                labelcolor=TEXT_CLR, fontsize=10)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    style_ax(ax1)

    # COVID crash annotation
    try:
        covid_start = pd.Timestamp("2020-02-19")
        covid_end   = pd.Timestamp("2020-03-23")
        ax1.axvspan(covid_start, covid_end, color="#ef4444", alpha=0.15)
        ax1.annotate("COVID\nCrash", xy=(covid_start, results["GNN_Agent"].max() * 0.9),
                      color="#ef4444", fontsize=8, fontstyle="italic")
    except Exception:
        pass

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for col, clr in COLORS.items():
        if col in results.columns:
            eq  = results[col].values
            cum = np.maximum.accumulate(eq)
            dd  = (eq - cum) / (cum + 1e-9) * 100
            ax2.fill_between(results.index, dd, 0, color=clr, alpha=0.35,
                              label=col)
            ax2.plot(results.index, dd, color=clr, linewidth=1.0)
    ax2.set_title("Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_CLR,
                fontsize=8)
    style_ax(ax2)

    # ── Panel 3: Rolling Sharpe (63-day) ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for col, clr in COLORS.items():
        if col in results.columns:
            daily = results[col].pct_change().dropna()
            roll_sharpe = (daily.rolling(63).mean() / daily.rolling(63).std()
                           * np.sqrt(252))
            ax3.plot(roll_sharpe.index, roll_sharpe.values,
                     label=col, color=clr, linewidth=1.2)
    ax3.axhline(0, color=BORDER, linewidth=0.8)
    ax3.set_title("Rolling 63-Day Sharpe Ratio", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Sharpe", fontsize=9)
    ax3.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_CLR,
                fontsize=8)
    style_ax(ax3)

    # ── Panel 4: Monthly contagion index ──────────────────────────────────
    if monthly_df is not None and "avg_contagion" in monthly_df.columns:
        ax4 = fig.add_subplot(gs[2, 0])
        mdates = pd.to_datetime(monthly_df["date"])
        ax4.fill_between(mdates, monthly_df["avg_contagion"],
                          color="#f87171", alpha=0.5, label="Avg contagion")
        ax4.plot(mdates, monthly_df["max_contagion"],
                 color="#ef4444", linewidth=1.2, label="Max contagion")
        ax4.set_title("GNN Contagion Index (Monthly)", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Contagion Score", fontsize=9)
        ax4.set_ylim(0, 1)
        ax4.legend(facecolor=PANEL_BG, edgecolor=BORDER,
                    labelcolor=TEXT_CLR, fontsize=8)
        style_ax(ax4)

    # ── Panel 5: Sector allocation heatmap ────────────────────────────────
    if monthly_df is not None:
        sector_cols = [c for c in monthly_df.columns if c.startswith("w_")]
        if sector_cols:
            ax5 = fig.add_subplot(gs[2, 1])
            sec_matrix = monthly_df[sector_cols].values.T
            sec_labels  = [c.replace("w_", "") for c in sector_cols]
            im = ax5.imshow(sec_matrix, aspect="auto", cmap="YlOrRd",
                             vmin=0, vmax=0.6)
            ax5.set_yticks(range(len(sec_labels)))
            ax5.set_yticklabels(sec_labels, fontsize=8, color=TEXT_CLR)
            ax5.set_title("Sector Weight Heatmap (Over Time)", fontsize=11,
                           fontweight="bold")
            ax5.set_xlabel("Month (index)", fontsize=9)
            ax5.tick_params(colors=TEXT_CLR)
            cbar = fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

    fig.suptitle("HTC-GNN Adaptive Portfolio — Backtest Report",
                 fontsize=16, fontweight="bold", color=TEXT_CLR, y=0.97)

    plt.savefig("backtest_chart.png", dpi=200, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print("\n[backtest] Chart saved -> backtest_chart.png")
    print("[backtest] Results saved -> backtest_results.csv, backtest_metrics.csv")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, metrics, monthly = run_backtest()
