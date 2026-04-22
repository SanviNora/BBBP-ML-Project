import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from config import SEEDS


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = float('nan')
    else:
        metrics['auc'] = float('nan')
    return metrics


def run_multi_seed(model_fn, dataset, split_type='random', seeds=SEEDS) -> dict:
    all_metrics = []
    for seed in seeds:
        if split_type == 'random':
            train, val, test = dataset.get_random_split(seed=seed)
        else:
            train, val, test = dataset.get_scaffold_split(seed=seed)
        model = model_fn(train, val, seed)
        preds = model.predict(test['X'])
        proba = model.predict_proba(test['X'])
        p1d   = proba[:, 1] if proba.ndim == 2 else proba
        all_metrics.append(compute_metrics(test['y'], preds, p1d))
    agg = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        agg[k] = (float(np.mean(vals)), float(np.std(vals)))
    return agg


def format_results_table(results: dict) -> str:
    header = '| Model | Split | AUC | F1 | Precision | Recall |'
    sep    = '|-------|-------|-----|----|-----------|---------'
    rows   = [header, sep]
    for (model_name, split), metrics in results.items():
        def fmt(k): return f"{metrics[k][0]:.3f}±{metrics[k][1]:.3f}"
        rows.append(f'| {model_name} | {split} | {fmt("auc")} | {fmt("f1")} | {fmt("precision")} | {fmt("recall")} |')
    return '\n'.join(rows)
