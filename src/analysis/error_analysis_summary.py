'''
import pandas as pd

gcn_correct = pd.read_csv("results/gcn_only_correct.csv")  
rf_correct  = pd.read_csv("results/rf_only_correct.csv")   

print(gcn_correct[['MW','LogP','HBD','HBA','rings']].mean())
print(rf_correct[['MW','LogP','HBD','HBA','rings']].mean())
'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── 0. Load data ──────────────────────────────────────────────────────────────
gcn_wins = pd.read_csv("results/gcn_only_correct.csv")  
rf_wins  = pd.read_csv("results/rf_only_correct.csv")   

gcn_wins['group'] = 'GCN correct\nRF wrong'
rf_wins['group']  = 'RF correct\nGCN wrong'
combined = pd.concat([gcn_wins, rf_wins], ignore_index=True)

DESCRIPTORS = ['MW', 'LogP', 'HBD', 'HBA', 'rings']
COLORS = {'GCN correct\nRF wrong': '#4C9BE8', 'RF correct\nGCN wrong': '#E8724C'}

os.makedirs("results", exist_ok=True)

# ── 1. Summary statistics ─────────────────────────────────────────────────────
print("=" * 60)
print("GCN correct & RF wrong:")
print(gcn_wins[DESCRIPTORS].describe().round(2))
print(f"\nN = {len(gcn_wins)} | true_label distribution:")
print(gcn_wins['true_label'].value_counts().to_string())

print("\n" + "=" * 60)
print("RF correct & GCN wrong:")
print(rf_wins[DESCRIPTORS].describe().round(2))
print(f"\nN = {len(rf_wins)} | true_label distribution:")
print(rf_wins['true_label'].value_counts().to_string())

print("\n" + "=" * 60)
print("Mean comparison:")
comparison = pd.DataFrame({
    'GCN_wins': gcn_wins[DESCRIPTORS].mean(),
    'RF_wins':  rf_wins[DESCRIPTORS].mean(),
})
comparison['diff'] = comparison['GCN_wins'] - comparison['RF_wins']
print(comparison.round(3))

# ── 2. Boxplots for each descriptor ──────────────────────────────────────────
fig, axes = plt.subplots(1, len(DESCRIPTORS), figsize=(14, 5))

for ax, desc in zip(axes, DESCRIPTORS):
    data_gcn = gcn_wins[desc].dropna()
    data_rf  = rf_wins[desc].dropna()

    bp = ax.boxplot([data_gcn, data_rf],
                    patch_artist=True,
                    widths=0.5,
                    medianprops=dict(color='black', linewidth=2))

    bp['boxes'][0].set_facecolor('#4C9BE8')
    bp['boxes'][1].set_facecolor('#E8724C')

    ax.set_title(desc, fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['GCN✓\nRF✗', 'RF✓\nGCN✗'], fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# shared legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4C9BE8', label='GCN correct / RF wrong'),
    Patch(facecolor='#E8724C', label='RF correct / GCN wrong'),
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=9)
fig.suptitle("Structural Descriptor Distributions: GCN vs RF Error Groups",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("results/i4a_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved results/i4a_boxplots.png")

# ── 3. MW vs LogP scatter ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for group, df_g in combined.groupby('group'):
    ax.scatter(df_g['MW'], df_g['LogP'],
               c=COLORS[group], label=group,
               alpha=0.7, s=50, edgecolors='white', linewidths=0.5)

ax.set_xlabel("Molecular Weight (MW)", fontsize=11)
ax.set_ylabel("LogP", fontsize=11)
ax.set_title("MW vs LogP: GCN vs RF Error Groups", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.axvline(500, color='gray', linestyle='--', alpha=0.5, label='Lipinski MW=500')
ax.axhline(5,   color='gray', linestyle=':',  alpha=0.5, label='Lipinski LogP=5')
plt.tight_layout()
plt.savefig("results/i4a_mw_logp_scatter.png", dpi=150)
plt.close()
print("Saved results/i4a_mw_logp_scatter.png")

# ── 4. Ring count histogram ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

max_rings = int(combined['rings'].max()) + 1
bins = range(0, max_rings + 1)

ax.hist(gcn_wins['rings'], bins=bins, alpha=0.6,
        color='#4C9BE8', label='GCN correct / RF wrong',
        align='left', rwidth=0.4)
ax.hist(rf_wins['rings'],  bins=bins, alpha=0.6,
        color='#E8724C', label='RF correct / GCN wrong',
        align='mid', rwidth=0.4)

ax.set_xlabel("Number of Rings", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Ring Count Distribution", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/i4a_ring_histogram.png", dpi=150)
plt.close()
print("Saved results/i4a_ring_histogram.png")

# ── 5. Concrete examples ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3-4 Concrete examples: GCN correct, RF wrong")
print("=" * 60)
for _, row in gcn_wins.head(4).iterrows():
    print(f"  Label={int(row['true_label'])} | MW={row['MW']} | LogP={row['LogP']} | "
          f"HBD={int(row['HBD'])} | HBA={int(row['HBA'])} | rings={int(row['rings'])}")
    print(f"  SMILES: {row['smiles']}\n")

print("=" * 60)
print("3-4 Concrete examples: RF correct, GCN wrong")
print("=" * 60)
for _, row in rf_wins.head(4).iterrows():
    print(f"  Label={int(row['true_label'])} | MW={row['MW']} | LogP={row['LogP']} | "
          f"HBD={int(row['HBD'])} | HBA={int(row['HBA'])} | rings={int(row['rings'])}")
    print(f"  SMILES: {row['smiles']}\n")