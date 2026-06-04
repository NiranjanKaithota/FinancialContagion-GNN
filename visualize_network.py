import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =========================================================
# 1. LOAD DATA
# =========================================================
print("Loading data for visualization...")
# We use the final predictions from the GNN and the Adjacency matrix
preds_df = pd.read_csv('gnn_predictions.csv', index_col=0)
A_df = pd.read_csv('edge_matrix_A.csv', index_col=0)

# =========================================================
# 2. BUILD THE NETWORKX GRAPH
# =========================================================
G = nx.Graph()

# Add Nodes with their Contagion Risk attributes
for ticker, row in preds_df.iterrows():
    G.add_node(ticker, risk=row['Contagion_Risk'], sector=row['Sector'])

# Add Edges (Only where correlation > 0)
for i in range(len(A_df.columns)):
    for j in range(i + 1, len(A_df.columns)):
        weight = A_df.iloc[i, j]
        if weight > 0:
            node_u = A_df.columns[i]
            node_v = A_df.columns[j]
            G.add_edge(node_u, node_v, weight=weight)

# =========================================================
# 3. CONFIGURE VISUAL LAYOUT
# =========================================================
# Use a spring layout to group highly correlated nodes together
pos = nx.spring_layout(G, k=0.8, seed=42)

# Extract node risks to determine colors
node_risks = [G.nodes[n]['risk'] for n in G.nodes()]

# Create a colormap: Green (0.0) -> Yellow (0.5) -> Red (1.0)
cmap = plt.cm.RdYlGn_r 

# Extract edge weights to determine line thickness
edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]

# =========================================================
# 4. DRAW THE GRAPH
# =========================================================
fig, ax = plt.subplots(figsize=(10, 8), dpi=300) # 300 DPI for publication quality

# Draw Edges
nx.draw_networkx_edges(
    G, pos, 
    ax=ax, 
    width=edge_weights, 
    edge_color="gray", 
    alpha=0.6
)

# Draw Nodes
nodes = nx.draw_networkx_nodes(
    G, pos, 
    ax=ax, 
    node_color=node_risks, 
    cmap=cmap, 
    node_size=1500, 
    vmin=0.0, 
    vmax=1.0,
    edgecolors="black",
    linewidths=1.5
)

# Draw Labels (Tickers)
nx.draw_networkx_labels(
    G, pos, 
    ax=ax, 
    font_size=10, 
    font_weight="bold", 
    font_color="black"
)

# =========================================================
# 5. ADD LEGEND & FORMATTING
# =========================================================
# Add a colorbar to explain the Heatmap
cbar = plt.colorbar(nodes, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('GNN Contagion Risk Probability', rotation=270, labelpad=20, weight='bold')

plt.title("GNN-Predicted Systemic Contagion Topology", fontsize=16, fontweight='bold', pad=20)
plt.axis("off") # Hide grid lines

# Save the figure as a high-res PNG for the research paper
plt.tight_layout()
plt.savefig("Contagion_Heatmap.png", format="png", bbox_inches="tight")
print("High-resolution graphic saved as 'Contagion_Heatmap.png'")

# Display the plot window
plt.show()