import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from config import RAW_DATA_PATH, SEED
from src.data.dataset import BBBPDataset
from src.data.graph_dataset import smiles_to_graph
from src.models.gcn import GCNBBB, GCNModel

# ── 0. Load dataset & scaffold split ──────────────────────────────────────────
base = BBBPDataset(RAW_DATA_PATH)
train_d, val_d, test_d = base.get_scaffold_split(seed=SEED)

def smiles_to_graph_list(smiles_list, label_list):
    graphs = []
    for smi, label in zip(smiles_list, label_list):
        g = smiles_to_graph(smi)
        g.y = torch.tensor([label], dtype=torch.long)
        graphs.append(g)
    return graphs

test_graphs = smiles_to_graph_list(test_d['smiles'], test_d['y'])
test_smiles = list(test_d['smiles'])
test_labels = np.array(test_d['y'])

# ── 1. Load GCN checkpoint ────────────────────────────────────────────────────
CHECKPOINT = "best_gcn_scaffold_cwTrue_seed42.pt"
BEST_CONFIG = dict(hidden_channels=256, num_layers=2, dropout=0.3,
                   epochs=200, patience=20)

gcn = GCNBBB(**BEST_CONFIG, checkpoint_path=CHECKPOINT, use_class_weight=True)
gcn.model = GCNModel(
    in_channels=test_graphs[0].x.shape[1],
    hidden_channels=256, num_layers=2, dropout=0.3
)
gcn.model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
gcn.model.eval()

gcn_preds   = gcn.predict(test_graphs)
gcn_correct = (gcn_preds == test_labels)

# ── 2. Load RF FP/FN smiles ───────────────────────────────────────────────────
def load_smiles_set(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found, returning empty set")
        return set()
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

rf_fp    = load_smiles_set("results/rf_fp_smiles_scaffold.txt")
rf_fn    = load_smiles_set("results/rf_fn_smiles_scaffold.txt")
rf_wrong = rf_fp | rf_fn

# ── 3. Compare errors ─────────────────────────────────────────────────────────
gcn_only_correct = []
rf_only_correct  = []

for smi, pred, label, correct in zip(test_smiles, gcn_preds, test_labels, gcn_correct):
    gcn_right = bool(correct)
    rf_right  = smi not in rf_wrong
    if gcn_right and not rf_right:
        gcn_only_correct.append({'smiles': smi, 'true_label': label, 'gcn_pred': pred})
    elif rf_right and not gcn_right:
        rf_only_correct.append({'smiles': smi, 'true_label': label, 'gcn_pred': pred})

print(f"\n[I4a] GCN correct & RF wrong: {len(gcn_only_correct)} molecules")
print(f"[I4a] RF correct & GCN wrong: {len(rf_only_correct)} molecules")

# ── 4. Describe molecules ─────────────────────────────────────────────────────
def describe_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    return {
        'smiles':  smi,
        'MW':      round(Descriptors.MolWt(mol), 2),
        'LogP':    round(Descriptors.MolLogP(mol), 2),
        'HBD':     Descriptors.NumHDonors(mol),
        'HBA':     Descriptors.NumHAcceptors(mol),
        'rings':   Descriptors.RingCount(mol),
    }

print("\n── GCN correct, RF wrong (top 4) ──")
for item in gcn_only_correct[:4]:
    desc = describe_mol(item['smiles'])
    print(f"  True={item['true_label']} | MW={desc.get('MW')} "
          f"LogP={desc.get('LogP')} HBD={desc.get('HBD')} "
          f"HBA={desc.get('HBA')} rings={desc.get('rings')}")
    print(f"  SMILES: {item['smiles']}")

print("\n── RF correct, GCN wrong (top 4) ──")
for item in rf_only_correct[:4]:
    desc = describe_mol(item['smiles'])
    print(f"  True={item['true_label']} | MW={desc.get('MW')} "
          f"LogP={desc.get('LogP')} HBD={desc.get('HBD')} "
          f"HBA={desc.get('HBA')} rings={desc.get('rings')}")
    print(f"  SMILES: {item['smiles']}")

# ── 5. Save CSVs ──────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

pd.DataFrame([describe_mol(x['smiles']) | {'true_label': x['true_label']}
              for x in gcn_only_correct]).to_csv("results/gcn_only_correct.csv", index=False)
pd.DataFrame([describe_mol(x['smiles']) | {'true_label': x['true_label']}
              for x in rf_only_correct]).to_csv("results/rf_only_correct.csv", index=False)

print("\nSaved results/gcn_only_correct.csv and results/rf_only_correct.csv")