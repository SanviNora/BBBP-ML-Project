"""
Grouped bar chart: AUC by model and split type (all 6 models).
"""

import matplotlib.pyplot as plt
import numpy as np

models = ['Lipinski', 'LR', 'SVM', 'MLP', 'RF', 'GCN']

random_auc = [0.6295, 0.9043, 0.9018, 0.8983, 0.9106, 0.8483]
random_std = [0.0166, 0.0221, 0.0045, 0.0134, 0.0233, 0.0245]

scaffold_auc = [0.5516, 0.7558, 0.7378, 0.7745, 0.7640, 0.5172]
scaffold_std = [0.0000, 0.0000, 0.0001, 0.0057, 0.0032, 0.0132]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width/2, random_auc, width, yerr=random_std,
               label='Random Split', color='#4C72B0', capsize=4, edgecolor='white')
bars2 = ax.bar(x + width/2, scaffold_auc, width, yerr=scaffold_std,
               label='Scaffold Split', color='#DD8452', capsize=4, edgecolor='white')

ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('Model Performance: Random vs Scaffold Split (All Models)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0.40, 1.02)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0.80, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='AUC=0.80')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/auc_barchart_final.png', dpi=150, bbox_inches='tight')
print("Saved to results/auc_barchart_final.png")
