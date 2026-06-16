"""
Layer 5: FastAPI Backend
Serves precomputed GNN + BL results to the React frontend.
All heavy computation is done offline; this just serves cached JSON.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import os

app = FastAPI(title="FinContagion API", version="1.0")

# Allow React dev server to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")

def load_json(filename):
    path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{filename} not found. Run the pipeline first.")
    with open(path) as f:
        return json.load(f)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "FinContagion API running"}


@app.get("/api/metadata")
def get_metadata():
    """Stock metadata: name, sector, color, vol, beta."""
    return load_json("metadata.json")


@app.get("/api/graph/latest")
def get_latest_graph():
    """Most recent graph snapshot (nodes + edges)."""
    snaps = load_json("graph_snapshots.json")
    return snaps[-1]


@app.get("/api/graph/snapshots")
def get_all_snapshots():
    """
    All graph snapshots (lightweight — just dates + edge/node counts).
    Use /api/graph/{index} for full detail.
    """
    snaps = load_json("graph_snapshots.json")
    return [{"date": s["date"], "node_count": s["node_count"],
             "edge_count": s["edge_count"]} for s in snaps]


@app.get("/api/graph/{index}")
def get_snapshot(index: int):
    """Full graph snapshot by index."""
    snaps = load_json("graph_snapshots.json")
    if index < 0 or index >= len(snaps):
        raise HTTPException(status_code=404, detail="Snapshot index out of range")
    return snaps[index]


@app.get("/api/risk/scores")
def get_risk_scores():
    """GNN risk scores for all snapshots."""
    return load_json("risk_scores.json")


@app.get("/api/risk/latest")
def get_latest_risk():
    """GNN risk scores for the most recent snapshot, merged with metadata."""
    risk_data = load_json("risk_scores.json")
    metadata  = load_json("metadata.json")
    snaps     = load_json("graph_snapshots.json")

    node_order    = risk_data["node_order"]
    latest_scores = risk_data["scores"][-1]
    latest_snap   = snaps[-1]

    result = []
    for i, nid in enumerate(node_order):
        meta = metadata.get(nid, {})
        result.append({
            "ticker":     nid,
            "name":       meta.get("name", nid),
            "sector":     meta.get("sector", "Unknown"),
            "color":      meta.get("color", "#888780"),
            "risk_score": round(float(latest_scores[i]), 4),
            "vol":        meta.get("vol", 0),
            "beta":       meta.get("beta", 1),
        })

    # Sort by risk descending
    result.sort(key=lambda x: -x["risk_score"])

    return {
        "date":   risk_data["dates"][-1],
        "assets": result,
        "systemic_risk_index": round(
            sum(r["risk_score"] for r in result) / len(result), 4
        ),
    }


@app.get("/api/risk/history/{ticker}")
def get_risk_history(ticker: str):
    """Time series of GNN risk scores for a single ticker."""
    risk_data  = load_json("risk_scores.json")
    node_order = risk_data["node_order"]
    if ticker not in node_order:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    idx = node_order.index(ticker)
    scores = [round(float(s[idx]), 4) for s in risk_data["scores"]]
    return {"ticker": ticker, "dates": risk_data["dates"], "scores": scores}


@app.get("/api/shock/{scenario}")
def get_shock_scenario(scenario: str):
    """
    GNN risk scores + BL weights for a crisis scenario.
    scenario: 2008_financial | 2020_covid | 2022_rate_hike
    """
    scenarios = load_json("shock_scenarios.json")
    bl_data   = load_json("scenario_bl.json")

    if scenario not in scenarios:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario. Choose from: {list(scenarios.keys())}"
        )

    return {
        "scenario":   scenario,
        "label":      scenarios[scenario]["label"],
        "node_risk":  scenarios[scenario]["node_risk"],
        "portfolio":  bl_data.get(scenario, {}),
    }


@app.get("/api/portfolio")
def get_portfolio():
    """Current BL + GNN portfolio vs equal-weight baseline."""
    return load_json("bl_result.json")


@app.get("/api/portfolio/sectors")
def get_sector_weights():
    """Sector-level weight comparison."""
    bl = load_json("bl_result.json")
    return bl.get("sector_weights", {})


@app.get("/api/backtest")
def get_backtest():
    """Backtest results: cumulative returns for all crisis periods."""
    return load_json("backtest.json")


@app.get("/api/backtest/{period}")
def get_backtest_period(period: str):
    """Backtest for a single crisis period (2020_covid | 2022_rate_hike)."""
    data = load_json("backtest.json")
    if period not in data:
        raise HTTPException(status_code=404, detail=f"Period {period} not found")
    return data[period]


@app.get("/api/network/sector")
def get_sector_network():
    """Sector-level contagion network (aggregated edge weights)."""
    return load_json("sector_adjacency.json")


@app.get("/api/summary")
def get_summary():
    """Dashboard summary: key numbers for the top metric cards."""
    risk    = load_json("risk_scores.json")
    bl      = load_json("bl_result.json")
    bt      = load_json("backtest.json")

    latest  = risk["scores"][-1]
    sri     = round(sum(latest) / len(latest), 4)
    n_high  = sum(1 for s in latest if s > 0.6)

    covid_bt = bt.get("2020_covid", {})
    drawdown_improvement = (
        round((covid_bt.get("max_dd_eq", 0) - covid_bt.get("max_dd_bl", 0)), 4)
        if covid_bt else 0
    )

    return {
        "systemic_risk_index": sri,
        "high_risk_assets":    n_high,
        "total_assets":        len(latest),
        "bl_sharpe":           bl["metrics_bl"]["sharpe"],
        "eq_sharpe":           bl["metrics_equal"]["sharpe"],
        "sharpe_improvement":  bl["improvement"]["sharpe_delta"],
        "drawdown_improvement":drawdown_improvement,
        "last_updated":        risk["dates"][-1],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
