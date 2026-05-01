##-----GCN Tuning-----##
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import numpy as np
from src.data.dataset import BBBPDataset
from src.data.splits import get_scaffold_split
from src.data.graph_dataset import BBBPGraphDataset, smiles_to_graph
from config import RAW_DATA_PATH, SEEDS, SEED
from src.models.gcn import GCNBBB
from torch_geometric.data import DataLoader

# Load dataset
base = BBBPDataset(RAW_DATA_PATH)
graph_dataset = BBBPGraphDataset(RAW_DATA_PATH)

# Get all graphs in same order as base.smiles
#all_graphs = [graph_dataset.get(i) for i in range(len(graph_dataset))]

# Random split
train_data, val_data, test_data = base.get_random_split(seed=SEED)

# Convert smiles lists to graph lists
def smiles_to_graph_list(smiles_list, label_list):
    graphs = []
    for smi, label in zip(smiles_list, label_list):
        g = smiles_to_graph(smi)
        g.y = torch.tensor([label], dtype=torch.long)
        graphs.append(g)
    return graphs

train_graphs = smiles_to_graph_list(train_data['smiles'], train_data['y'])
val_graphs   = smiles_to_graph_list(val_data['smiles'],   val_data['y'])
test_graphs  = smiles_to_graph_list(test_data['smiles'],  test_data['y'])

print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

# Train
gcn = GCNBBB(hidden_channels=64, num_layers=2, dropout=0.0, epochs=200, patience=20)
gcn.fit(train_graphs, val_graphs)

# Test AUC
test_auc = gcn._evaluate_auc(DataLoader(test_graphs, batch_size=32))
print(f"Test AUC: {test_auc:.4f}")
    

##-----Sweep hidden_dim in [64, 128, 256]-----##

results = {}

for hidden_dim in [64, 128, 256]:
    print(f"\n{'='*40}")
    print(f"Training with hidden_dim={hidden_dim}")
    print(f"{'='*40}")
    
    gcn = GCNBBB(
        hidden_channels=hidden_dim,
        num_layers=2,
        dropout=0.0,
        epochs=200,
        patience=20,
        checkpoint_path=f"best_gcn_hidden{hidden_dim}.pt"
    )
    gcn.fit(train_graphs, val_graphs)
    
    test_auc = gcn._evaluate_auc(DataLoader(test_graphs, batch_size=32))
    results[hidden_dim] = {
        'best_val_auc': max(auc for _, auc in gcn.history),
        'test_auc': test_auc,
        'history': gcn.history
    }
    print(f"Test AUC: {test_auc:.4f}")

# Summary
print(f"\n{'='*40}")
print("Summary:")
print(f"{'hidden_dim':>12} | {'Best Val AUC':>12} | {'Test AUC':>10}")
print(f"{'-'*40}")
for hd, r in results.items():
    print(f"{hd:>12} | {r['best_val_auc']:>12.4f} | {r['test_auc']:>10.4f}")


##-----Sweep num_layers + dropout-----##
import itertools

num_layers_list = [2, 3, 4]
dropout_list    = [0.1, 0.3, 0.5]

results_3b = {}

for num_layers, dropout in itertools.product(num_layers_list, dropout_list):
    key = f"layers{num_layers}_drop{dropout}"
    print(f"\n{'='*40}")
    print(f"num_layers={num_layers} | dropout={dropout}")
    print(f"{'='*40}")

    gcn = GCNBBB(
        hidden_channels=256,       # best from I3a
        num_layers=num_layers,
        dropout=dropout,
        epochs=200,
        patience=20,
        checkpoint_path=f"best_gcn_{key}.pt"
    )
    gcn.fit(train_graphs, val_graphs)

    test_auc = gcn._evaluate_auc(DataLoader(test_graphs, batch_size=32))
    best_val = max(auc for _, auc in gcn.history)
    results_3b[key] = {'num_layers': num_layers, 'dropout': dropout,
                       'best_val_auc': best_val, 'test_auc': test_auc}
    print(f"Test AUC: {test_auc:.4f}")

# Summary
print(f"\n{'='*50}")
print("Summary:")
print(f"{'num_layers':>10} | {'dropout':>7} | {'Best Val AUC':>12} | {'Test AUC':>10}")
print(f"{'-'*50}")
for r in results_3b.values():
    print(f"{r['num_layers']:>10} | {r['dropout']:>7} | {r['best_val_auc']:>12.4f} | {r['test_auc']:>10.4f}")