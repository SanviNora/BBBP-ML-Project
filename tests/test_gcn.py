
###-----small test-----###
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

## -----load dataset-----##
base = BBBPDataset(RAW_DATA_PATH)
graph_dataset = BBBPGraphDataset(RAW_DATA_PATH)

# Get all graphs in same order as base.smiles
#all_graphs = [graph_dataset.get(i) for i in range(len(graph_dataset))]

## ----random split test-----##
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

# ----- no class weight -----
gcn_random = GCNBBB(hidden_channels=64, num_layers=2, dropout=0.0,
                   epochs=20, patience=5, use_class_weight=False)
gcn_random.fit(train_graphs, val_graphs)
random_test_auc = gcn_random._evaluate_auc(DataLoader(test_graphs, batch_size=32))
print(f"[Random | No Weight] Test AUC: {random_test_auc:.4f}")

# ----- with class weight -----
gcn_random_bal = GCNBBB(hidden_channels=64, num_layers=2, dropout=0.0,
                       epochs=20, patience=5, use_class_weight=True)
gcn_random_bal.fit(train_graphs, val_graphs)
random_test_auc_bal = gcn_random_bal._evaluate_auc(DataLoader(test_graphs, batch_size=32))
print(f"[Random | Weighted] Test AUC: {random_test_auc_bal:.4f}")

## ----scaffold split test-----##

train_data, val_data, test_data = base.get_scaffold_split(seed=SEED)

train_graphs = smiles_to_graph_list(train_data["smiles"], train_data["y"])
val_graphs   = smiles_to_graph_list(val_data["smiles"], val_data["y"])
test_graphs  = smiles_to_graph_list(test_data["smiles"], test_data["y"])

print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")
print("Train positives:", sum(g.y.item() for g in train_graphs), "/", len(train_graphs))
print("Val positives:", sum(g.y.item() for g in val_graphs), "/", len(val_graphs))
print("Test positives:", sum(g.y.item() for g in test_graphs), "/", len(test_graphs))

# ----- no class weight -----
gcn_scaffold = GCNBBB(hidden_channels=64, num_layers=2, dropout=0.0,
                     epochs=20, patience=5, use_class_weight=False)
gcn_scaffold.fit(train_graphs, val_graphs)
scaffold_test_auc = gcn_scaffold._evaluate_auc(DataLoader(test_graphs, batch_size=32))
print(f"[Scaffold | No Weight] Test AUC: {scaffold_test_auc:.4f}")

# ----- with class weight -----
gcn_scaffold_bal = GCNBBB(hidden_channels=64, num_layers=2, dropout=0.0,
                         epochs=20, patience=5, use_class_weight=True)
gcn_scaffold_bal.fit(train_graphs, val_graphs)
scaffold_test_auc_bal = gcn_scaffold_bal._evaluate_auc(DataLoader(test_graphs, batch_size=32))
print(f"[Scaffold | Weighted] Test AUC: {scaffold_test_auc_bal:.4f}")

## ----summary-----##
print("\n===== Summary =====")
print(f"Random   | No Weight: {random_test_auc:.4f}")
print(f"Random   | Weighted : {random_test_auc_bal:.4f}")
print(f"Scaffold | No Weight: {scaffold_test_auc:.4f}")
print(f"Scaffold | Weighted : {scaffold_test_auc_bal:.4f}")