"""
Layer 3: GNN Contagion Model
Architecture: Temporal GCN + GRU
- GCNConv learns spatial relationships (who is connected to whom)
- GRU learns temporal evolution (how risk evolves over time)
- Output: risk score per node (0 = safe, 1 = high contagion risk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
import numpy as np
import json
import pandas as pd
import os


class TemporalGNN(nn.Module):
    """
    Temporal GCN + GRU for financial contagion prediction.
    
    Architecture:
      Input node features (5-dim) 
        → GCN layer 1 (spatial message passing)
        → GCN layer 2 (deeper neighborhood aggregation)
        → GRU (temporal state across snapshots)
        → Linear head → risk score ∈ [0, 1]
    """
    def __init__(self, in_features=5, hidden_dim=64, gru_hidden=32, num_gcn_layers=2):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.gru_hidden  = gru_hidden

        # Spatial: Graph convolution layers
        self.gcn1 = GCNConv(in_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Optional attention layer for interpretability
        self.gat  = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=0.2)

        # Temporal: GRU processes sequence of GCN embeddings
        self.gru  = nn.GRU(hidden_dim, gru_hidden, batch_first=True, num_layers=2,
                           dropout=0.2)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid(),   # output in [0,1]
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward_snapshot(self, data):
        """Process a single graph snapshot. Returns node embeddings."""
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # GCN layers with residual-ish skip
        h = F.elu(self.gcn1(x, edge_index, edge_weight))
        h = self.norm1(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h = F.elu(self.gcn2(h, edge_index, edge_weight))
        h = self.norm2(h)

        return h   # [N, hidden_dim]

    def forward(self, snapshot_list):
        """
        snapshot_list: list of T PyG Data objects (time-ordered snapshots)
        Returns: risk_scores [N, 1] for each node at the last snapshot
        """
        # Build sequence of GCN embeddings per node
        # Assumes all snapshots have same node order (they do in our pipeline)
        embeddings = []
        for snap in snapshot_list:
            h = self.forward_snapshot(snap)   # [N, hidden_dim]
            embeddings.append(h.unsqueeze(1)) # [N, 1, hidden_dim]

        # [N, T, hidden_dim]
        seq = torch.cat(embeddings, dim=1)

        # GRU over time dimension
        out, _ = self.gru(seq)               # [N, T, gru_hidden]
        last   = out[:, -1, :]               # [N, gru_hidden]

        risk_scores = self.head(last)         # [N, 1]
        return risk_scores


def snapshot_to_pyg(snapshot, node_order):
    """Convert a graph snapshot dict → PyTorch Geometric Data object."""
    node_idx = {nid: i for i, nid in enumerate(node_order)}
    N = len(node_order)

    # Build feature matrix [N, 5]
    feat_map = {n["id"]: n["features"] for n in snapshot["nodes"]}
    X = []
    for nid in node_order:
        f = feat_map.get(nid, {"mu": 0, "vol": 0.2, "skew": 0, "kurt": 0, "beta": 1})
        X.append([f["mu"], f["vol"], f["skew"], f["kurt"], f["beta"]])
    X = torch.tensor(X, dtype=torch.float)

    # Build edge_index [2, E] and edge_attr [E]
    src, dst, weights = [], [], []
    for e in snapshot["edges"]:
        s = node_idx.get(e["source"])
        t = node_idx.get(e["target"])
        if s is None or t is None:
            continue
        src.append(s); dst.append(t); weights.append(e["weight"])
        if not e.get("directed", False):       # undirected → add reverse
            src.append(t); dst.append(s); weights.append(e["weight"])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(weights,     dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros(0,       dtype=torch.float)

    return Data(x=X, edge_index=edge_index, edge_attr=edge_attr)


def create_training_labels(returns, snapshots, node_order, forward_window=20, threshold=0.0):
    """
    Label = 1 if stock drops >threshold in the next forward_window days.
    This is our ground truth for "contagion / shock" events.
    """
    labels_per_snapshot = []
    dates = pd.to_datetime([s["date"] for s in snapshots])

    for snap_date in dates:
        # Find position in returns index (Updated for Pandas 2.0+)
        try:
            dt = pd.to_datetime(snap_date)
            pos = int(returns.index.get_indexer([dt], method="nearest")[0])
        except Exception as e:
            print(f"Error labeling date {snap_date}: {e}")
            labels_per_snapshot.append(np.zeros(len(node_order)))
            continue
        
        future_end = min(pos + forward_window, len(returns) - 1)
        if future_end <= pos:
            labels_per_snapshot.append(np.zeros(len(node_order)))
            continue

        future_ret = returns.iloc[pos:future_end].sum()  # cumulative log return
        labels = []
        for nid in node_order:
            r = future_ret.get(nid, 0.0)
            labels.append(1.0 if r < -threshold else 0.0)
        labels_per_snapshot.append(np.array(labels))

    return labels_per_snapshot   # list of [N] arrays


def train_model(snapshots, returns, epochs=40, seq_len=8, lr=1e-3, device="cpu"):
    """
    Train the Temporal GNN on historical graph snapshots.
    Uses sliding windows of seq_len snapshots as input sequences.
    """
    node_order = [n["id"] for n in snapshots[0]["nodes"]]
    N = len(node_order)

    # Convert all snapshots to PyG objects
    pyg_snaps = [snapshot_to_pyg(s, node_order) for s in snapshots]

    # Labels: one label vector [N] per snapshot
    all_labels = create_training_labels(returns, snapshots, node_order)

    model = TemporalGNN(in_features=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    print(f"Training on {N} nodes, {len(snapshots)} snapshots, {epochs} epochs...")
    print(f"Sequence length: {seq_len} snapshots per training sample")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for end_i in range(seq_len, len(pyg_snaps)):
            seq = pyg_snaps[end_i - seq_len: end_i]
            y   = torch.tensor(all_labels[end_i], dtype=torch.float).unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred = model(seq)           # [N, 1]
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch [{epoch+1:3d}/{epochs}]  loss: {avg_loss:.4f}  "
                  f"lr: {scheduler.get_last_lr()[0]:.5f}")

    return model, node_order, pyg_snaps


def generate_risk_scores(model, pyg_snaps, node_order, seq_len=8, device="cpu"):
    """Run inference on all snapshots. Returns risk score per node per snapshot."""
    model.eval()
    all_scores = []

    with torch.no_grad():
        for end_i in range(seq_len, len(pyg_snaps) + 1):
            seq = pyg_snaps[max(0, end_i - seq_len): end_i]
            scores = model(seq).squeeze(1).cpu().numpy()   # [N]
            all_scores.append(scores.tolist())

    return all_scores   # [T', N]


def generate_shock_scenarios(model, pyg_snaps, node_order, seq_len=8, device="cpu"):
    """
    Simulate shock injection: amplify edge weights for a sector,
    rerun inference to see how risk propagates.
    """
    scenarios = {}
    crises = {
        "2008_financial": {"date_hint": "2008", "shock_sectors": ["Banking", "Finance"], "mult": 2.5},
        "2020_covid":     {"date_hint": "2020", "shock_sectors": ["Auto", "Infra", "Energy"], "mult": 2.0},
        "2022_rate_hike": {"date_hint": "2022", "shock_sectors": ["IT", "Consumer"], "mult": 1.6},
    }

    for scenario_name, cfg in crises.items():
        # Use last available snapshot for shock injection
        seq = pyg_snaps[-seq_len:]

        # Amplify features for shocked nodes
        shocked_seq = []
        for snap in seq:
            snap_copy = snap.clone()
            # Increase volatility feature (index 1) for shocked nodes
            snap_copy.x[:, 1] = snap_copy.x[:, 1] * cfg["mult"]
            snap_copy.x[:, 0] = snap_copy.x[:, 0] - 0.05 * cfg["mult"]  # reduce returns
            shocked_seq.append(snap_copy)

        model.eval()
        with torch.no_grad():
            scores = model(shocked_seq).squeeze(1).cpu().numpy()

        scenarios[scenario_name] = {
            "node_risk": {nid: round(float(s), 4) for nid, s in zip(node_order, scores)},
            "label": scenario_name.replace("_", " ").title(),
        }

    return scenarios


if __name__ == "__main__":
    OUT_DIR = os.path.join(os.path.dirname(__file__), "../../outputs")

    print("Loading graph snapshots and returns...")
    with open(f"{OUT_DIR}/graph_snapshots.json") as f:
        snapshots = json.load(f)

    returns = pd.read_csv(f"{OUT_DIR}/returns.csv", index_col=0, parse_dates=True)
    with open(f"{OUT_DIR}/metadata.json") as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, node_order, pyg_snaps = train_model(
        snapshots, returns, epochs=40, seq_len=8, device=device
    )

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "node_order": node_order,
    }, f"{OUT_DIR}/gnn_model.pt")
    print(f"\nModel saved → gnn_model.pt")

    # Generate risk scores for all snapshots
    risk_scores = generate_risk_scores(model, pyg_snaps, node_order, seq_len=8)

    # Package risk scores with dates
    snap_dates = [s["date"] for s in snapshots[8:]]   # offset by seq_len
    risk_output = {
        "node_order": node_order,
        "dates": snap_dates,
        "scores": risk_scores,   # [T, N]
    }
    with open(f"{OUT_DIR}/risk_scores.json", "w") as f:
        json.dump(risk_output, f)
    print(f"Risk scores saved → risk_scores.json")

    # Generate crisis scenarios
    scenarios = generate_shock_scenarios(model, pyg_snaps, node_order)
    with open(f"{OUT_DIR}/shock_scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"Shock scenarios saved → shock_scenarios.json")

    # Print top-5 highest risk nodes at latest snapshot
    latest_scores = risk_scores[-1]
    ranked = sorted(zip(node_order, latest_scores), key=lambda x: -x[1])
    print(f"\nTop 5 highest-risk nodes (latest snapshot):")
    for nid, score in ranked[:5]:
        name = metadata.get(nid, {}).get("name", nid)
        print(f"  {name:20s}  risk={score:.3f}")
