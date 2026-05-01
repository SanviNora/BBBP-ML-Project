# src/analysis/error_analysis.py
import sys
sys.path.insert(0, '/Users/nora/BBBP-ML-Project')
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from src.data.dataset import BBBPDataset
from src.models.baselines import BBBRandomForest, LipinskiClassifier


def get_physicochemical_profile(smiles_list):
    """Get MW, LogP, HBD, HBA for a list of SMILES."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append({
            'smiles': smi,
            'MW':   Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD':  Descriptors.NumHDonors(mol),
            'HBA':  Descriptors.NumHAcceptors(mol),
        })
    return pd.DataFrame(rows)


def analyze_errors(model_name, y_true, y_pred, smiles_list):
    """Analyze false positives and false negatives for a model."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]
    tp_idx = np.where((y_pred == 1) & (y_true == 1))[0]

    fp_smiles = [smiles_list[i] for i in fp_idx]
    fn_smiles = [smiles_list[i] for i in fn_idx]
    tp_smiles = [smiles_list[i] for i in tp_idx]

    fp_profile = get_physicochemical_profile(fp_smiles)
    fn_profile = get_physicochemical_profile(fn_smiles)
    tp_profile = get_physicochemical_profile(tp_smiles)

    print(f"\n{'='*50}")
    print(f"Error Analysis: {model_name}")
    print(f"{'='*50}")
    print(f"False Positives (predicted BBB+, actually BBB-): {len(fp_idx)}")
    print(f"False Negatives (predicted BBB-, actually BBB+): {len(fn_idx)}")
    print(f"True Positives (correctly predicted BBB+):       {len(tp_idx)}")

    if len(fp_profile) > 0:
        print(f"\nFalse Positive avg properties:")
        print(f"  MW={fp_profile['MW'].mean():.1f}  LogP={fp_profile['LogP'].mean():.2f}  HBD={fp_profile['HBD'].mean():.1f}  HBA={fp_profile['HBA'].mean():.1f}")

    if len(fn_profile) > 0:
        print(f"False Negative avg properties:")
        print(f"  MW={fn_profile['MW'].mean():.1f}  LogP={fn_profile['LogP'].mean():.2f}  HBD={fn_profile['HBD'].mean():.1f}  HBA={fn_profile['HBA'].mean():.1f}")

    if len(tp_profile) > 0:
        print(f"True Positive avg properties (reference):")
        print(f"  MW={tp_profile['MW'].mean():.1f}  LogP={tp_profile['LogP'].mean():.2f}  HBD={tp_profile['HBD'].mean():.1f}  HBA={tp_profile['HBA'].mean():.1f}")

    return {
        'model': model_name,
        'n_fp': len(fp_idx),
        'n_fn': len(fn_idx),
        'fp_profile': fp_profile,
        'fn_profile': fn_profile,
        'tp_profile': tp_profile,
        'fp_smiles': fp_smiles,
        'fn_smiles': fn_smiles,
    }


def run_error_analysis():
    """Run error analysis for RF and Lipinski under scaffold split."""
    ds = BBBPDataset('data/raw/BBBP.csv')
    train, val, test = ds.get_scaffold_split(seed=42)

    results = {}

    # RF error analysis
    print("\nRunning RF error analysis on scaffold split...")
    rf = BBBRandomForest(seed=42)
    rf.fit(train['X'], train['y'], val['X'], val['y'])
    rf_preds = rf.predict(test['X'])
    results['RF'] = analyze_errors('RF (scaffold)', test['y'], rf_preds, test['smiles'])

    # Lipinski error analysis
    print("\nRunning Lipinski error analysis on scaffold split...")
    clf = LipinskiClassifier()
    lip_preds = clf.predict(test['smiles'])
    results['Lipinski'] = analyze_errors('Lipinski (scaffold)', test['y'], lip_preds, test['smiles'])

    # Summary comparison
    print(f"\n{'='*50}")
    print("SUMMARY: False Negative comparison")
    print(f"{'='*50}")
    for name, r in results.items():
        fn = r['fn_profile']
        if len(fn) > 0:
            print(f"{name}: {r['n_fn']} FN | avg MW={fn['MW'].mean():.1f} LogP={fn['LogP'].mean():.2f}")

    # Save summary to CSV
    summary_rows = []
    for name, r in results.items():
        fp = r['fp_profile']
        fn = r['fn_profile']
        tp = r['tp_profile']
        summary_rows.append({
            'Model': name,
            'Split': 'scaffold',
            'N_FP': r['n_fp'],
            'N_FN': r['n_fn'],
            'FP_avg_MW':   round(fp['MW'].mean(), 1)   if len(fp) > 0 else '',
            'FP_avg_LogP': round(fp['LogP'].mean(), 2) if len(fp) > 0 else '',
            'FP_avg_HBD':  round(fp['HBD'].mean(), 1)  if len(fp) > 0 else '',
            'FP_avg_HBA':  round(fp['HBA'].mean(), 1)  if len(fp) > 0 else '',
            'FN_avg_MW':   round(fn['MW'].mean(), 1)   if len(fn) > 0 else '',
            'FN_avg_LogP': round(fn['LogP'].mean(), 2) if len(fn) > 0 else '',
            'TP_avg_MW':   round(tp['MW'].mean(), 1)   if len(tp) > 0 else '',
            'TP_avg_LogP': round(tp['LogP'].mean(), 2) if len(tp) > 0 else '',
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('results/error_analysis_summary.csv', index=False)
    print("\nSaved to results/error_analysis_summary.csv")
    
    return results


if __name__ == '__main__':
    results = run_error_analysis()