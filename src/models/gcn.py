import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

# Minimal GCN model stub for forward pass verification
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels=9, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)   # graph-level pooling
        return self.classifier(x)

# Build a synthetic batch of 4 graphs
synthetic_smiles = [
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "CCO",                           # ethanol
    "c1ccccc1",                      # benzene
    "CC(=O)Oc1ccccc1C(=O)O",        # aspirin
]
batch = Batch.from_data_list([smiles_to_graph(s) for s in synthetic_smiles])

model = GCNModel()
model.eval()
with torch.no_grad():
    out = model(batch.x, batch.edge_index, batch.batch)

print(f"Input batch:  {len(synthetic_smiles)} graphs")
print(f"Output shape: {out.shape}   expected: torch.Size([4, 2])")
print("Forward pass: OK" if out.shape == torch.Size([4, 2]) else "Forward pass: FAILED")
