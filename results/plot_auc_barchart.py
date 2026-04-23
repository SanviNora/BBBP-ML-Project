"""
Grouped bar chart: AUC by model and split type.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from fingerprint_results.csv
models = ['LR', 'SVM', 'MLP']
random_auc =   [0.9032, 0.9002, 0.8828]
random_std =   [0.0245, 0.0031, 0.0063]
scaffold_auc = [0.7617, 0.7464, 0.7659]
scaffold_std = [0.0000, 0.0000, 0.0092]

x = np.arange(len(models))
width = 0.30

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, random_auc, width, yerr=random_std,
               label='Random Split', color='#4C72B0', capsize=4, edgecolor='white')
bars2 = ax.bar(x + width/2, scaffold_auc, width, yerr=scaffold_std,
               label='Scaffold Split', color='#DD8452', capsize=4, edgecolor='white')

ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('Fingerprint Model Performance: Random vs Scaffold Split', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0.60, 1.00)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0.80, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/auc_barchart.png', dpi=150, bbox_inches='tight')
print("Saved to results/auc_barchart.png")
plt.show()
