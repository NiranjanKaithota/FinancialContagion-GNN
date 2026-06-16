"""
gnn_model.py
============
Publication-grade Graph Neural Network for financial contagion prediction.

Architecture: Hierarchical Temporal Contagion GNN (HTC-GNN)
─────────────────────────────────────────────────────────────
Layer 1  — Multi-head Graph Attention (firm→firm message passing)
Layer 2  — Second GATConv layer (deeper firm-level embedding)
Layer 3  — Sector super-node pooling (macro aggregation)
Layer 4  — Cross-level attention fusion (micro ⊕ macro)
Layer 5  — Temporal GRU cell (captures drift across snapshots)
Head     — Binary crash classifier with calibrated Sigmoid output

Why these choices:
  • GAT (vs GCN): attention weights identify which neighbour firms are the
    most contagious — directly interpretable as a contagion channel map.
  • Hierarchical pooling: mirrors the paper's micro/macro requirement.
  • GRU cell: lets the model track how contagion risk evolves month-to-month
    without needing a full sequence — one hidden state per ticker.
  • Class-weighted BCE: handles the natural imbalance (crashes are rare).
  • Focal Loss option: pushes the model to focus on hard crash samples.

Training uses a walk-forward split:
  train  2015-2018  |  val  2019  |  test  2020-2024 (incl. COVID crash)
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0. GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FEATURES = 13       # must match len(NUMERIC_FEATS) in data_loader
HIDDEN_DIM   = 64
GAT_HEADS    = 4        # multi-head attention
NUM_SECTORS  = 5
DROPOUT      = 0.3
LR           = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 300
PATIENCE     = 30       # early stopping
MODEL_PATH   = os.path.join("cache", "best_model.pt")
os.makedirs("cache", exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. FOCAL LOSS  (better than BCE for imbalanced crash labels)
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Lin et al. (2017) Focal Loss:  FL(p) = -α(1−p)^γ log(p)
    γ=2, α=0.75 pushes model to study the hard positive (crash) examples.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        fl = alpha_t * ((1 - pt) ** self.gamma) * bce
        return fl.mean()


# ──────────────────────────────────────────────────────────────────────────────
# 2. HTC-GNN  MODEL
# ──────────────────────────────────────────────────────────────────────────────
class HTCGNNModel(nn.Module):
    """
    Hierarchical Temporal Contagion GNN (HTC-GNN).

    forward() accepts:
        data        : PyG Data object (x, edge_index, edge_attr)
        sector_idx  : LongTensor of shape (N,) mapping each firm to its sector
        h_prev      : Optional FloatTensor (N, HIDDEN_DIM) — GRU hidden state
                      from the previous time step (pass None for first snapshot)

    Returns:
        logits      : (N, 1) raw logits (apply sigmoid for probability)
        h_new       : (N, HIDDEN_DIM) updated GRU hidden state
        attn_list   : list of attention-weight tensors for interpretability
    """
    def __init__(self,
                 num_features: int   = NUM_FEATURES,
                 hidden_dim: int     = HIDDEN_DIM,
                 gat_heads: int      = GAT_HEADS,
                 num_sectors: int    = NUM_SECTORS,
                 dropout: float      = DROPOUT):
        super().__init__()

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── Micro level: two GAT layers ────────────────────────────────────────
        self.gat1 = GATConv(hidden_dim, hidden_dim // gat_heads,
                             heads=gat_heads, dropout=dropout,
                             edge_dim=1, concat=True)   # out: hidden_dim

        self.gat2 = GATConv(hidden_dim, hidden_dim,
                             heads=1, dropout=dropout,
                             edge_dim=1, concat=False)  # out: hidden_dim

        self.micro_norm1 = nn.LayerNorm(hidden_dim)
        self.micro_norm2 = nn.LayerNorm(hidden_dim)

        # ── Macro level: sector pooling + sector MLP ──────────────────────────
        self.sector_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Cross-level attention fusion ──────────────────────────────────────
        #  Keys/queries come from micro embeddings, values from macro
        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_norm  = nn.LayerNorm(hidden_dim * 2)
        self.fusion_proj  = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Temporal GRU cell (stateful across monthly snapshots) ─────────────
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Residual skip ─────────────────────────────────────────────────────
        self.skip = nn.Linear(num_features, hidden_dim)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self,
                data,
                sector_idx: torch.Tensor,
                num_sectors: int,
                h_prev: torch.Tensor | None = None):

        x_raw, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # ── Input projection + residual ───────────────────────────────────────
        x      = self.input_proj(x_raw)
        skip_x = self.skip(x_raw)

        # ── Micro GAT layer 1 ─────────────────────────────────────────────────
        x1, attn1 = self.gat1(x, edge_index, edge_attr=edge_attr,
                               return_attention_weights=True)
        x1 = self.micro_norm1(F.elu(x1) + x)   # residual

        # ── Micro GAT layer 2 ─────────────────────────────────────────────────
        x2, attn2 = self.gat2(x1, edge_index, edge_attr=edge_attr,
                               return_attention_weights=True)
        x2 = self.micro_norm2(F.elu(x2) + x1)  # residual

        micro_embed = x2  # shape (N, hidden_dim)

        # ── Macro sector pooling ──────────────────────────────────────────────
        sector_nodes = []
        for s in range(num_sectors):
            mask = (sector_idx == s)
            if mask.any():
                pooled = micro_embed[mask].mean(dim=0)
            else:
                pooled = torch.zeros(micro_embed.size(1), device=DEVICE)
            sector_nodes.append(pooled)
        sector_embed = torch.stack(sector_nodes)               # (S, H)
        sector_embed = self.sector_mlp(sector_embed)           # (S, H)

        # ── Cross-level attention: each firm attends to its sector super-node ─
        q = self.cross_attn_q(micro_embed)                     # (N, H)
        k = self.cross_attn_k(sector_embed[sector_idx])        # (N, H)
        v = self.cross_attn_v(sector_embed[sector_idx])        # (N, H)
        scale = math.sqrt(q.size(-1))
        attn_w = torch.softmax((q * k).sum(-1, keepdim=True) / scale, dim=0)
        macro_ctx = attn_w * v                                  # (N, H)

        fused = self.fusion_norm(torch.cat([micro_embed, macro_ctx], dim=-1))
        fused = F.relu(self.fusion_proj(fused)) + skip_x       # (N, H)

        # ── Temporal GRU update ───────────────────────────────────────────────
        if h_prev is None:
            h_prev = torch.zeros_like(fused)
        h_new = self.gru_cell(fused, h_prev)                   # (N, H)
        h_new = self.temporal_norm(h_new)

        # ── Classification head ───────────────────────────────────────────────
        logits = self.head(h_new)                              # (N, 1)

        return logits, h_new, [attn1, attn2]


# ──────────────────────────────────────────────────────────────────────────────
# 3. GRAPH DATA BUILDER  (PyG Data object from snapshot)
# ──────────────────────────────────────────────────────────────────────────────
def build_pyg_data(X_norm: np.ndarray,
                   A_df: pd.DataFrame,
                   y: np.ndarray) -> Data:
    x         = torch.tensor(X_norm, dtype=torch.float)
    A_arr     = A_df.values.astype(np.float32)
    edge_idx  = torch.tensor(np.stack(np.where(A_arr > 0)), dtype=torch.long)
    edge_attr = torch.tensor(A_arr[A_arr > 0], dtype=torch.float)
    y_tensor  = torch.tensor(y, dtype=torch.float).unsqueeze(1)
    return Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=y_tensor)


def sector_idx_tensor(tickers: list, sector_map: dict,
                      unique_sectors: list) -> torch.Tensor:
    s2i = {s: i for i, s in enumerate(unique_sectors)}
    idxs = [s2i[sector_map[t]] for t in tickers]
    return torch.tensor(idxs, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAINING ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def train_model(snapshots_train: list[dict],
                snapshots_val:   list[dict],
                num_features: int = NUM_FEATURES,
                epochs: int       = EPOCHS,
                patience: int     = PATIENCE) -> HTCGNNModel:
    """
    snapshots_train / val: list of dicts
        {pyg_data: Data, sector_idx: Tensor, num_sectors: int, date: str}

    Uses walk-forward cross-validation: training set = all snapshots from
    the historical window, validation = the immediately following period.
    """
    model     = HTCGNNModel(num_features=num_features).to(DEVICE)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2)

    best_val_auc  = 0.0
    patience_ctr  = 0
    train_losses  = []
    val_aucs      = []

    print(f"\n[gnn_model] Training HTC-GNN on {DEVICE} …")
    print(f"  Train snapshots: {len(snapshots_train)}")
    print(f"  Val   snapshots: {len(snapshots_val)}")
    print(f"  Epochs: {epochs}  |  Patience: {patience}\n")

    for epoch in range(1, epochs + 1):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        h_state    = None   # reset temporal state each epoch

        for snap in snapshots_train:
            data      = snap["pyg_data"].to(DEVICE)
            sec_idx   = snap["sector_idx"].to(DEVICE)
            n_sectors = snap["num_sectors"]

            optimizer.zero_grad()
            logits, h_state, _ = model(data, sec_idx, n_sectors,
                                        h_prev=h_state.detach() if h_state is not None else None)
            loss = criterion(logits, data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(snapshots_train)
        train_losses.append(avg_loss)

        # ── Validation ───────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            all_probs, all_labels = [], []
            h_val = None
            with torch.no_grad():
                for snap in snapshots_val:
                    data    = snap["pyg_data"].to(DEVICE)
                    sec_idx = snap["sector_idx"].to(DEVICE)
                    logits, h_val, _ = model(data, sec_idx, snap["num_sectors"],
                                              h_prev=h_val)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    labs  = data.y.squeeze().cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(labs.tolist())

            if len(set(all_labels)) > 1:
                val_auc = roc_auc_score(all_labels, all_probs)
                val_ap  = average_precision_score(all_labels, all_probs)
            else:
                val_auc = val_ap = 0.5

            val_aucs.append(val_auc)
            print(f"  Epoch {epoch:3d} | Loss {avg_loss:.4f} | "
                  f"Val AUROC {val_auc:.4f} | Val AP {val_ap:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_ctr = 0
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"           ^ New best AUROC {best_val_auc:.4f} - model saved.")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stopping triggered at epoch {epoch}.")
                    break

    # Load best weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                     weights_only=True))
    print(f"\n[gnn_model] Best validation AUROC: {best_val_auc:.4f}")

    # Plot training curve
    _plot_training_curve(train_losses, val_aucs, epochs)

    return model


def _plot_training_curve(losses, aucs, total_epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses, color="#ef4444", linewidth=1.5)
    ax1.set_title("Training Loss (Focal Loss)", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)

    val_x = list(range(4, 4 + len(aucs) * 5, 5))[:len(aucs)]
    ax2.plot(val_x, aucs, color="#3b82f6", linewidth=1.5, marker="o", markersize=3)
    ax2.set_title("Validation AUROC", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUROC")
    ax2.set_ylim(0.4, 1.0)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[gnn_model] Training curve saved → training_curve.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION SUITE  (publishable metrics)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model: HTCGNNModel,
                   snapshots_test: list[dict],
                   tickers: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with per-ticker mean contagion scores over the test set,
    and prints AUROC, AP, F1, confusion matrix.
    """
    model.eval()
    all_probs, all_labels, all_tickers = [], [], []
    h_state = None

    with torch.no_grad():
        for snap in snapshots_test:
            data    = snap["pyg_data"].to(DEVICE)
            sec_idx = snap["sector_idx"].to(DEVICE)
            logits, h_state, _ = model(data, sec_idx,
                                        snap["num_sectors"], h_prev=h_state)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            labs  = data.y.squeeze().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())
            all_tickers.extend(snap["tickers"])

    results_df = pd.DataFrame({
        "ticker": all_tickers,
        "contagion_prob": all_probs,
        "true_label": all_labels,
    })

    preds_binary = (np.array(all_probs) >= 0.5).astype(int)
    auc = roc_auc_score(all_labels, all_probs)
    ap  = average_precision_score(all_labels, all_probs)

    print("\n" + "═" * 55)
    print("  HTC-GNN TEST SET EVALUATION")
    print("═" * 55)
    print(f"  AUROC              : {auc:.4f}")
    print(f"  Average Precision  : {ap:.4f}")
    print()
    print(classification_report(all_labels, preds_binary,
                                  target_names=["Safe", "Crash"],
                                  digits=4))
    cm = confusion_matrix(all_labels, preds_binary)
    print(f"  Confusion Matrix:\n{cm}")
    print("═" * 55 + "\n")

    # Per-ticker mean score
    summary = (results_df.groupby("ticker")
               .agg(mean_contagion=("contagion_prob", "mean"),
                    crash_rate=("true_label", "mean"))
               .sort_values("mean_contagion", ascending=False))
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 6. STANDALONE TRAINING  (run this file directly to train & save)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import (build_dataset, get_snapshot,
                              SECTOR_MAP, STOCK_UNIVERSE, NUMERIC_FEATS)

    prices, features_long, adjacencies, crash_labels, spy = build_dataset()

    UNIQUE_SECTORS = sorted(STOCK_UNIVERSE.keys())
    scaler         = None

    # Walk-forward split
    TRAIN_END = "2018-12-31"
    VAL_END   = "2019-12-31"

    snap_dates = sorted(adjacencies.keys())
    train_dates = [d for d in snap_dates if d <= TRAIN_END]
    val_dates   = [d for d in snap_dates if TRAIN_END < d <= VAL_END]
    test_dates  = [d for d in snap_dates if d > VAL_END]

    def make_snapshots(dates):
        snaps = []
        _scaler = None
        for d in dates:
            try:
                X, A, y, tks, _scaler = get_snapshot(
                    features_long, adjacencies, crash_labels, d, _scaler)
                si = sector_idx_tensor(tks, SECTOR_MAP, UNIQUE_SECTORS)
                snaps.append({
                    "pyg_data":   build_pyg_data(X, A, y),
                    "sector_idx": si,
                    "num_sectors": len(UNIQUE_SECTORS),
                    "tickers":    tks,
                    "date":       d,
                })
            except Exception as e:
                print(f"[warn] Skipping snapshot {d}: {e}")
        return snaps, _scaler

    train_snaps, scaler = make_snapshots(train_dates)
    val_snaps,   _      = make_snapshots(val_dates)
    test_snaps,  _      = make_snapshots(test_dates)

    model = train_model(train_snaps, val_snaps,
                        num_features=len(NUMERIC_FEATS))

    summary = evaluate_model(model, test_snaps,
                              tickers=[t for v in STOCK_UNIVERSE.values() for t in v])
    summary.to_csv("gnn_predictions.csv")
    print("\nContagion score summary (top 10):")
    print(summary.head(10))
