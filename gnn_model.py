import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

# =========================================================
# 1. LOAD DATA & PREPARE SECTOR INDICES
# =========================================================
print("Loading data from CSVs...")
X_df = pd.read_csv('node_features_X.csv', index_col=0)
A_df = pd.read_csv('edge_matrix_A.csv', index_col=0)

# Extract numeric features
features = X_df[['Return', 'Volatility', 'Momentum']].values
x = torch.tensor(features, dtype=torch.float)

# Map string sectors to integer indices for pooling
unique_sectors = X_df['Sector'].unique()
sector_to_idx = {sector: idx for idx, sector in enumerate(unique_sectors)}
X_df['Sector_Idx'] = X_df['Sector'].map(sector_to_idx)

# Create a tensor of sector assignments (e.g., [0, 1, 2, 2...])
sector_idx_tensor = torch.tensor(X_df['Sector_Idx'].values, dtype=torch.long)
num_sectors = len(unique_sectors)

# Extract Edges
A_tensor = torch.tensor(A_df.values, dtype=torch.float)
edge_index = (A_tensor > 0).nonzero(as_tuple=False).t()
edge_weight = A_tensor[edge_index[0], edge_index[1]]

# Create a Ground Truth Target (y) for Training
# For this simulation, we pretend a historical crash happened where 
# Tech (Sector 0) crashed heavily (1.0), and Energy survived (0.0).
# Your team will eventually replace this with actual historical drawdown data.
dummy_targets = []
for sector in X_df['Sector']:
    if sector == 'Tech':
        dummy_targets.append([1.0]) # 100% chance of crash
    elif sector == 'Finance':
        dummy_targets.append([0.5]) # 50% chance of crash
    else:
        dummy_targets.append([0.0]) # 0% chance of crash

y = torch.tensor(dummy_targets, dtype=torch.float)

# Build the PyG Data Object
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

print(f"Data Loaded. Found {num_sectors} distinct Macro Sectors: {list(unique_sectors)}")

# =========================================================
# 2. DEFINE THE HIERARCHICAL GNN ARCHITECTURE
# =========================================================
class HierarchicalContagionGNN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(HierarchicalContagionGNN, self).__init__()
        
        # Level 1: Firm-to-Firm Message Passing
        self.micro_conv = GCNConv(num_features, hidden_channels)
        
        # Level 2: Combine Firm Health with Sector Health
        # Input size is hidden_channels * 2 because we concatenate Firm + Sector embeddings
        self.macro_combine = nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Final Predictor Layer
        self.predictor = nn.Linear(hidden_channels, 1)

    def forward(self, data, sector_idx, total_sectors):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # --- PHASE 1: MICRO MESSAGE PASSING ---
        # Companies influence each other based on correlation edges
        firm_embeds = self.micro_conv(x, edge_index, edge_weight=edge_weight)
        firm_embeds = F.relu(firm_embeds)
        
        # --- PHASE 2: MACRO POOLING ---
        # Calculate Sector Super-Node embeddings by averaging the firms inside them
        # We loop through each sector ID, find its firms, and take the mean.
        sector_embeds_list = []
        for s in range(total_sectors):
            # Mask to find firms in sector 's'
            mask = (sector_idx == s)
            if mask.any():
                sector_mean = firm_embeds[mask].mean(dim=0)
            else:
                sector_mean = torch.zeros(firm_embeds.size(1), device=firm_embeds.device)
            sector_embeds_list.append(sector_mean)
            
        sector_embeds = torch.stack(sector_embeds_list)
        
        # --- PHASE 3: BROADCAST & COMBINE ---
        # Attach the Sector embedding back to each individual firm
        broadcasted_macro = sector_embeds[sector_idx]
        
        # Concatenate: [Firm Representation, Sector Representation]
        combined = torch.cat([firm_embeds, broadcasted_macro], dim=1)
        
        # Pass through the linear combination layer
        x = F.relu(self.macro_combine(combined))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # --- PHASE 4: FINAL PREDICTION ---
        # Sigmoid activation to squash output to a probability (0.0 to 1.0)
        contagion_score = torch.sigmoid(self.predictor(x))
        return contagion_score

# =========================================================
# 3. THE TRAINING LOOP
# =========================================================
print("\nInitializing Training Loop...")

# Initialize Model, Loss Function, and Optimizer
model = HierarchicalContagionGNN(num_features=3, hidden_channels=16)

# Binary Cross Entropy Loss is standard for probabilities between 0 and 1
criterion = nn.BCELoss() 

# Adam Optimizer with a standard learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training Loop
epochs = 100
model.train()

for epoch in range(1, epochs + 1):
    optimizer.zero_grad() # Clear old gradients
    
    # Forward Pass
    out = model(graph_data, sector_idx_tensor, num_sectors)
    
    # Calculate Loss against the target (y)
    loss = criterion(out, graph_data.y)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f}')

# =========================================================
# 4. EVALUATION (THE AGENT SIGNAL)
# =========================================================
print("\nTraining Complete! Running Final Contagion Inference...")
model.eval()
with torch.no_grad():
    final_scores = model(graph_data, sector_idx_tensor, num_sectors)

# Attach scores to the DataFrame to visualize the AI's output
X_df['Contagion_Risk'] = final_scores.numpy()
X_df['Target_Crash'] = graph_data.y.numpy()

print("\nFinal Output (The Signal for the Financial Agent):")
print(X_df[['Sector', 'Target_Crash', 'Contagion_Risk']])
X_df.to_csv('gnn_predictions.csv')