"""
Fingerprint model comparison: LR, SVM, MLP
Runs all models over 3 seeds × 2 splits × 4 metrics
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from src.data.dataset import BBBPDataset
from src.models.fingerprint_models import LogisticRegressionBBB, SVMBBB, MLPBBB
from config import SEEDS, RAW_DATA_PATH


def compute_metrics(y_true, y_pred, y_proba):
    """Compute all 4 evaluation metrics."""
    return {
        'AUC':       roc_auc_score(y_true, y_proba),
        'F1':        f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred),
    }


def run_model_on_split(ModelClass, ds, split_type, seed):
    """Train and evaluate a single model on a single seed and split."""
    if split_type == 'random':
        train, val, test = ds.get_random_split(seed)
    else:
        train, val, test = ds.get_scaffold_split(seed)

    model = ModelClass(seed=seed)
    model.fit(train['X'], train['y'], val['X'], val['y'])

    preds = model.predict(test['X'])
    proba = model.predict_proba(test['X'])[:, 1]

    return compute_metrics(test['y'], preds, proba)


def main():
    os.makedirs('results', exist_ok=True)
    ds = BBBPDataset(RAW_DATA_PATH)

    models = {
        'LR':  LogisticRegressionBBB,
        'SVM': SVMBBB,
        'MLP': MLPBBB,
    }
    splits = ['random', 'scaffold']
    metrics_list = ['AUC', 'F1', 'Precision', 'Recall']

    rows = []
    for model_name, ModelClass in models.items():
        for split in splits:
            seed_results = {m: [] for m in metrics_list}
            for seed in SEEDS:
                print(f"  Running {model_name} | {split} | seed={seed}")
                metrics = run_model_on_split(ModelClass, ds, split, seed)
                for m in metrics_list:
                    seed_results[m].append(metrics[m])

            # Compute mean ± std across seeds
            row = {'Model': model_name, 'Split': split}
            for m in metrics_list:
                vals = seed_results[m]
                row[f'{m}_mean'] = np.mean(vals)
                row[f'{m}_std']  = np.std(vals)
                row[m] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv('results/fingerprint_results.csv', index=False)

    # Print a nice table
    print("\n" + "=" * 70)
    print("FINGERPRINT MODEL RESULTS (mean ± std over 3 seeds)")
    print("=" * 70)
    display_cols = ['Model', 'Split', 'AUC', 'F1', 'Precision', 'Recall']
    print(df[display_cols].to_string(index=False))
    print("=" * 70)

    return df


if __name__ == '__main__':
    main()
