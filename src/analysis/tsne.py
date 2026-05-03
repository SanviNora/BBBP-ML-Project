import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from config import RAW_DATA_PATH, SEED
from src.data.dataset import BBBPDataset
from src.data.graph_dataset import smiles_to_graph
from src.models.gcn import GCNBBB, GCNModel

# ── 0. Load dataset & scaffold split ──────────────────────────────────────────
base = BBBPDataset(RAW_DATA_PATH)
_, _, test_d = base.get_scaffold_split(seed=SEED)

def smiles_to_graph_list(smiles_list, label_list):
    graphs = []
    for smi, label in zip(smiles_list, label_list):
        g = smiles_to_graph(smi)
        g.y = torch.tensor([label], dtype=torch.long)
        graphs.append(g)
    return graphs

test_graphs = smiles_to_graph_list(test_d['smiles'], test_d['y'])
test_labels = np.array(test_d['y'])

# ── 1. Load GCN checkpoint ────────────────────────────────────────────────────
CHECKPOINT = "best_gcn_scaffold_cwFalse_seed42.pt"

gcn = GCNBBB(hidden_channels=256, num_layers=2, dropout=0.3,
             epochs=200, patience=20,
             checkpoint_path=CHECKPOINT, use_class_weight=False)
gcn.model = GCNModel(
    in_channels=test_graphs[0].x.shape[1],
    hidden_channels=256, num_layers=2, dropout=0.3
)
gcn.model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
gcn.model.eval()

gcn_preds   = gcn.predict(test_graphs)
gcn_correct = (gcn_preds == test_labels)

# ── 2. Extract embeddings & run t-SNE ─────────────────────────────────────────
print("[I4b] Extracting GCN embeddings...")
embeddings = gcn.get_embeddings(test_graphs)   # [N, 256]

print("Running t-SNE...")
tsne  = TSNE(n_components=2, random_state=SEED, perplexity=30)
emb2d = tsne.fit_transform(embeddings)         # [N, 2]

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

colors = {0: '#4C9BE8', 1: '#E8724C'}
labels_str = {0: 'Non-BBB (0)', 1: 'BBB (1)'}

for label in [0, 1]:
    mask = (test_labels == label) & gcn_correct
    ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
               c=colors[label], alpha=0.6, s=30,
               label=f'{labels_str[label]} correct')

ax.scatter(emb2d[~gcn_correct, 0], emb2d[~gcn_correct, 1],
           c='black', marker='x', s=50, linewidths=1.2,
           label='GCN error')

ax.set_title("t-SNE of GCN Graph Embeddings (scaffold test set)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(loc='best', fontsize=9)
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/gcn_tsne.png", dpi=150)
plt.close()
print("Saved results/gcn_tsne.png")

# ── 4. ECFP4 collision search ─────────────────────────────────────────────────
print("\n[I4b] Searching for ECFP4 collision (Tanimoto=1.0, different labels)...")

all_smiles = list(base.smiles)
all_labels = list(base.y)

fps, valid_idx = [], []
for i, smi in enumerate(all_smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fps.append(GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        valid_idx.append(i)

collision_found = False
for threshold, label in [(1.0, "Exact"), (0.99, "Near")]:
    if collision_found:
        break
    if threshold < 1.0:
        print(f"  No exact collision found, relaxing to Tanimoto >= {threshold}...")
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            li  = all_labels[valid_idx[i]]
            lj  = all_labels[valid_idx[j]]
            if sim >= threshold and li != lj:
                print(f"  {label} collision found! Tanimoto={sim:.4f}")
                print(f"  Mol A: label={li} | {all_smiles[valid_idx[i]]}")
                print(f"  Mol B: label={lj} | {all_smiles[valid_idx[j]]}")
                collision_found = True
                break
        if collision_found:
            break

if not collision_found:
    print("  No collision found in this dataset.")

from rdkit.Chem import Draw

# ── 5. Draw ECFP4 collision molecules ────────────────────────────────────────
if collision_found:
    smi_a = all_smiles[valid_idx[i]]
    smi_b = all_smiles[valid_idx[j]]
    
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    
    img = Draw.MolsToGridImage(
        [mol_a, mol_b],
        molsPerRow=2,
        subImgSize=(400, 300),
        legends=[
            f"Mol A | label={all_labels[valid_idx[i]]} (BBB+)",
            f"Mol B | label={all_labels[valid_idx[j]]} (non-BBB)",
        ]
    )
    img.save("results/ecfp4_collision.png")
    print("Saved results/ecfp4_collision.png")