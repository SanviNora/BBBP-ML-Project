##-----run GCN final-----##
import sys
import os
from turtle import pd
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
    
##-----3 seeds × 2 splits → gcn_results.csv-----##
import os
import csv
from src.evaluation.metrics import compute_metrics
import pandas as pd

# Best config from I3a + I3b
BEST_CONFIG = dict(hidden_channels=256, num_layers=2, dropout=0.3, epochs=200, patience=20)

os.makedirs("results", exist_ok=True)

csv_rows = []
fp_smiles = []  # false positives under scaffold split
fn_smiles = []  # false negatives under scaffold split

def evaluate_graphs(gcn, graphs):
    """Returns proba, preds, labels as numpy arrays."""
    proba  = gcn.predict_proba(graphs)[:, 1]
    preds  = gcn.predict(graphs)
    labels = np.array([g.y.item() for g in graphs])
    metrics = compute_metrics(labels, preds, proba)
    return proba, preds, labels, metrics

def graphs_from_split(split_dict):
    return smiles_to_graph_list(split_dict['smiles'], split_dict['y'])
csv_rows = []
fp_smiles_all = {}  # key: (split, use_class_weight)
fn_smiles_all = {}

for seed in SEEDS:
    for split_name in ['random', 'scaffold']:
        for use_cw in [False, True]:
            print(f"\n{'='*50}")
            print(f"Seed={seed} | Split={split_name} | class_weight={use_cw}")
            print(f"{'='*50}")

            if split_name == 'random':
                train_d, val_d, test_d = base.get_random_split(seed=seed)
            else:
                train_d, val_d, test_d = base.get_scaffold_split(seed=seed)

            train_g = graphs_from_split(train_d)
            val_g   = graphs_from_split(val_d)
            test_g  = graphs_from_split(test_d)

            gcn = GCNBBB(**BEST_CONFIG,
                         use_class_weight=use_cw,
                         checkpoint_path=f"best_gcn_{split_name}_cw{use_cw}_seed{seed}.pt")
            gcn.fit(train_g, val_g)

            proba, preds, labels, metrics = evaluate_graphs(gcn, test_g)

            print(f"AUC={metrics['auc']:.4f} | F1={metrics['f1']:.4f} | "
                  f"Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f}")

            csv_rows.append({
                'model': 'GCN',
                'split': split_name,
                'use_class_weight': use_cw,
                'seed': seed,
                'auc':       round(metrics['auc'],       4),
                'f1':        round(metrics['f1'],        4),
                'precision': round(metrics['precision'], 4),
                'recall':    round(metrics['recall'],    4),
            })

            # Collect FP/FN under scaffold split
            if split_name == 'scaffold':
                key = (split_name, use_cw)
                fp_smiles_all.setdefault(key, [])
                fn_smiles_all.setdefault(key, [])
                for smi, pred, label in zip(test_d['smiles'], preds, labels):
                    if pred == 1 and label == 0:
                        fp_smiles_all[key].append(smi)
                    elif pred == 0 and label == 1:
                        fn_smiles_all[key].append(smi)
'''
for seed in SEEDS:
    for split_name in ['random', 'scaffold']:
        print(f"\n{'='*40}")
        print(f"Seed={seed} | Split={split_name}")
        print(f"{'='*40}")

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
'''
csv_path = "results/gcn_results.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'model','split','use_class_weight','seed','auc','f1','precision','recall'
    ])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nSaved results to {csv_path}")

# Save FP/FN per config
for (split_name, use_cw), fps in fp_smiles_all.items():
    tag = f"scaffold_cw{use_cw}"
    with open(f"results/gcn_fp_smiles_{tag}.txt", 'w') as f:
        f.write('\n'.join(set(fps)))
    fns = fn_smiles_all.get((split_name, use_cw), [])
    with open(f"results/gcn_fn_smiles_{tag}.txt", 'w') as f:
        f.write('\n'.join(set(fns)))
    print(f"[{tag}] FP: {len(set(fps))} | FN: {len(set(fns))}")
'''
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
'''
##-----Final results summary-----##
df = pd.read_csv("results/gcn_results_summary.csv")

print("\nFinal Results (mean ± std across 3 seeds):")
print(f"{'Split':>10} | {'ClassWeight':>12} | {'AUC':>15} | {'F1':>15} | {'Precision':>15} | {'Recall':>15}")
print("-" * 95)

for split in ['random', 'scaffold']:
    for use_cw in [False, True]:
        sub = df[(df['split'] == split) & (df['use_class_weight'] == use_cw)]
        print(f"{split:>10} | {str(use_cw):>12} | "
              f"{sub['auc'].mean():.4f} ± {sub['auc'].std():.4f} | "
              f"{sub['f1'].mean():.4f} ± {sub['f1'].std():.4f} | "
              f"{sub['precision'].mean():.4f} ± {sub['precision'].std():.4f} | "
              f"{sub['recall'].mean():.4f} ± {sub['recall'].std():.4f}")