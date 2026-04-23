##-----GCN_Model-----##
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
import numpy as np

class GCNModel(torch.nn.Module):
    """
    Graph Convolutional Network for molecular property prediction.
    """
    def __init__(self, in_channels=9, hidden_channels=64, out_channels=2, 
                 num_layers=2, dropout=0.0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)  # shape [N]


class GCNBBB:
    """
    Wrapper around GCNModel with fit(), predict(), predict_proba().
    """
    def __init__(self, hidden_channels=64, num_layers=2, dropout=0.0,
                 lr=1e-3, epochs=200, patience=20, batch_size=32,
                 checkpoint_path="best_gcn.pt"):
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.model = None

    def fit(self, train_graphs, val_graphs):
        """
        Train the GCN model.

        Args:
            train_graphs: list of PyG Data objects (training set)
            val_graphs:   list of PyG Data objects (validation set)
        """
        self.model = GCNModel(
            in_channels=train_graphs[0].x.shape[1],
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        train_loader = DataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_graphs,   batch_size=self.batch_size, shuffle=False)

        best_val_auc = 0.0
        patience_counter = 0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            # Training
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(out, batch.y.float().squeeze())
                loss.backward()
                optimizer.step()

            # Validation AUC every 10 epochs
            if epoch % 10 == 0:
                val_auc = self._evaluate_auc(val_loader)
                self.history.append((epoch, val_auc))
                print(f"Epoch {epoch:3d} | Val AUC: {val_auc:.4f}")

                # Save best checkpoint
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience // 10:
                        print(f"Early stopping at epoch {epoch} | Best Val AUC: {best_val_auc:.4f}")
                        break

        # Load best checkpoint
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.eval()
        print(f"Training done. Best Val AUC: {best_val_auc:.4f}")

    def _evaluate_auc(self, loader):
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(out).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch.y.cpu().numpy())
        return roc_auc_score(all_labels, all_probs)

    def predict_proba(self, graphs):
        """
        Returns probability array of shape [N, 2].
        """
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(out).cpu().numpy()
                all_probs.extend(probs)
        all_probs = np.array(all_probs)
        return np.stack([1 - all_probs, all_probs], axis=1)  # shape [N, 2]

    def predict(self, graphs):
        """
        Returns predicted labels of shape [N,].
        """
        proba = self.predict_proba(graphs)
        return (proba[:, 1] >= 0.5).astype(int)


##-----Now we can use the GCNBBB class to train and evaluate on the BBBP dataset.-----##
import numpy as np
from src.data.dataset import BBBPDataset
from src.data.splits import get_scaffold_split

# Load dataset using Nora's loader
base = BBBPDataset("BBBP.csv")

# Get all graphs in same order as base.smiles
all_graphs = [dataset.get(i) for i in range(len(dataset))]

# Random split (reuse Nora's method)
train_data, val_data, test_data = base.get_random_split(seed=42)

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



##-----3 seeds × 2 splits → gcn_results.csv-----##
import os
import csv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Best config from I3a + I3b
BEST_CONFIG = dict(hidden_channels=256, num_layers=2, dropout=0.3, epochs=200, patience=20)
SEEDS = [42, 123, 7]

os.makedirs("results", exist_ok=True)

csv_rows = []
fp_smiles = []  # false positives under scaffold split
fn_smiles = []  # false negatives under scaffold split

def evaluate_graphs(gcn, graphs):
    """Returns proba, preds, labels as numpy arrays."""
    proba  = gcn.predict_proba(graphs)[:, 1]
    preds  = gcn.predict(graphs)
    labels = np.array([g.y.item() for g in graphs])
    return proba, preds, labels

def graphs_from_split(split_dict):
    return smiles_to_graph_list(split_dict['smiles'], split_dict['y'])

for seed in SEEDS:
    for split_name in ['random', 'scaffold']:
        print(f"\n{'='*40}")
        print(f"Seed={seed} | Split={split_name}")
        print(f"{'='*40}")

        # Get splits from Nora's functions
        if split_name == 'random':
            train_d, val_d, test_d = base.get_random_split(seed=seed)
        else:
            train_d, val_d, test_d = base.get_scaffold_split(seed=seed)

        train_g = graphs_from_split(train_d)
        val_g   = graphs_from_split(val_d)
        test_g  = graphs_from_split(test_d)

        # Train
        gcn = GCNBBB(**BEST_CONFIG,
                     checkpoint_path=f"best_gcn_{split_name}_seed{seed}.pt")
        gcn.fit(train_g, val_g)

        # Evaluate on test set
        proba, preds, labels = evaluate_graphs(gcn, test_g)

        auc  = roc_auc_score(labels, proba)
        f1   = f1_score(labels, preds)
        prec = precision_score(labels, preds)
        rec  = recall_score(labels, preds)

        print(f"AUC={auc:.4f} | F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")

        csv_rows.append({
            'model': 'GCN',
            'split': split_name,
            'seed': seed,
            'auc': round(auc, 4),
            'f1': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
        })

        # Collect FP and FN under scaffold split
        if split_name == 'scaffold':
            test_smiles = test_d['smiles']
            for smi, pred, label in zip(test_smiles, preds, labels):
                if pred == 1 and label == 0:
                    fp_smiles.append(smi)
                elif pred == 0 and label == 1:
                    fn_smiles.append(smi)

# Save results CSV
csv_path = "results/gcn_results.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['model','split','seed','auc','f1','precision','recall'])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nSaved results to {csv_path}")

# Save FP/FN SMILES for I4 error analysis
with open("results/gcn_fp_smiles.txt", 'w') as f:
    f.write('\n'.join(set(fp_smiles)))
with open("results/gcn_fn_smiles.txt", 'w') as f:
    f.write('\n'.join(set(fn_smiles)))

print(f"False positives: {len(set(fp_smiles))} unique molecules")
print(f"False negatives: {len(set(fn_smiles))} unique molecules")

##-----Final results summary-----##
import pandas as pd

df = pd.read_csv("results/gcn_results.csv")

print("\nFinal Results (mean ± std across 3 seeds):")
print(f"{'Split':>10} | {'AUC':>15} | {'F1':>15} | {'Precision':>15} | {'Recall':>15}")
print("-" * 80)

for split in ['random', 'scaffold']:
    sub = df[df['split'] == split]
    print(f"{split:>10} | "
          f"{sub['auc'].mean():.4f} ± {sub['auc'].std():.4f} | "
          f"{sub['f1'].mean():.4f} ± {sub['f1'].std():.4f} | "
          f"{sub['precision'].mean():.4f} ± {sub['precision'].std():.4f} | "
          f"{sub['recall'].mean():.4f} ± {sub['recall'].std():.4f}")